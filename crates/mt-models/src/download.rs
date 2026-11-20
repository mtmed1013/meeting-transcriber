//! Resumable, checksum-verified model downloads with atomic install.
//!
//! Files are written to `<models_dir>/.mt-downloading/<folder>/<name>.part`, verified,
//! renamed to `<name>`, and the whole staging folder is renamed to `<models_dir>/<folder>`.

use std::fs::{self, File, OpenOptions};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use reqwest::StatusCode;
use reqwest::header::{CONTENT_RANGE, RANGE};
use sha2::{Digest, Sha256};

use crate::error::ModelHubError;
use crate::hub::HubClient;
use crate::plan::{ModelPlan, PlannedFile};

/// Hidden folder under the models directory that holds in-progress downloads.
pub const STAGING_DIR: &str = ".mt-downloading";

/// Minimum time between two throttled progress reports.
const PROGRESS_INTERVAL: Duration = Duration::from_millis(150);

/// Cooperative cancellation flag. Clones share state.
#[derive(Debug, Clone, Default)]
pub struct CancelToken(Arc<AtomicBool>);

impl CancelToken {
    pub fn new() -> Self {
        Self::default()
    }

    /// Requests cancellation. Takes effect at the next chunk boundary.
    pub fn cancel(&self) {
        self.0.store(true, Ordering::SeqCst);
    }

    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }
}

/// Cumulative progress across the whole plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Progress {
    pub downloaded: u64,
    pub total: u64,
}

/// Installs model plans into a models directory.
#[derive(Debug, Clone)]
pub struct Downloader {
    hub: HubClient,
    models_dir: PathBuf,
}

impl Downloader {
    pub fn new(hub: HubClient, models_dir: impl Into<PathBuf>) -> Self {
        Self {
            hub,
            models_dir: models_dir.into(),
        }
    }

    /// Downloads every file in `plan` and atomically installs the folder.
    /// Returns the final folder path. On cancel or network failure, `.part` files are kept for resume.
    pub async fn install<F>(
        &self,
        plan: &ModelPlan,
        cancel: &CancelToken,
        mut progress: F,
    ) -> Result<PathBuf, ModelHubError>
    where
        F: FnMut(Progress),
    {
        validate_component(&plan.folder)?;
        for file in &plan.files {
            validate_component(&file.name)?;
        }
        let final_dir = self.models_dir.join(&plan.folder);
        if final_dir.exists() {
            return Err(already_installed(&final_dir));
        }

        let staging_root = self.models_dir.join(STAGING_DIR);
        let staging = staging_root.join(&plan.folder);
        fs::create_dir_all(&staging)?;

        let mut done = 0;
        for file in &plan.files {
            if cancel.is_cancelled() {
                return Err(ModelHubError::Cancelled);
            }
            let dest = staging.join(&file.name);
            let len = match already_verified(&dest, file).await? {
                Some(len) => len,
                None => {
                    let part = part_path(&staging, file);
                    if !is_complete_part(&part, file)? {
                        self.download_part(plan, file, &part, cancel, done, &mut progress)
                            .await?;
                    }
                    publish(file, &part, &dest).await?;
                    file_len(&dest)?
                }
            };
            done += len;
            progress(Progress {
                downloaded: done,
                total: plan.total_bytes,
            });
        }

        if final_dir.exists() {
            return Err(already_installed(&final_dir));
        }
        fs::rename(&staging, &final_dir)?;
        let _ = fs::remove_dir(&staging_root);
        Ok(final_dir)
    }

    /// Appends the missing bytes of `file` to `part`, resuming with `Range` when possible.
    /// `base` is the number of bytes already completed by earlier files in the plan.
    async fn download_part<F>(
        &self,
        plan: &ModelPlan,
        file: &PlannedFile,
        part: &Path,
        cancel: &CancelToken,
        base: u64,
        progress: &mut F,
    ) -> Result<(), ModelHubError>
    where
        F: FnMut(Progress),
    {
        let have = file_len(part)?;
        let mut request = self.hub.http().get(self.hub.file_url(&plan.repo, &file.name));
        if have > 0 {
            request = request.header(RANGE, format!("bytes={have}-"));
        }
        let mut response = request.send().await?;

        let resumed = match response.status() {
            StatusCode::PARTIAL_CONTENT if have > 0 => {
                if content_range_start(&response) != Some(have) {
                    remove_if_exists(part)?;
                    return Err(ModelHubError::BadResponse(format!(
                        "Content-Range no coincide al reanudar {}",
                        file.name
                    )));
                }
                true
            }
            StatusCode::OK => false,
            status => {
                if status == StatusCode::RANGE_NOT_SATISFIABLE {
                    remove_if_exists(part)?;
                }
                return Err(status_error(status, &plan.repo.repo_id));
            }
        };

        let handle = if resumed {
            OpenOptions::new().append(true).open(part)?
        } else {
            File::create(part)?
        };
        let mut out = BufWriter::new(handle);
        let mut written = if resumed { have } else { 0 };
        let mut last_report: Option<Instant> = None;

        while let Some(chunk) = response.chunk().await? {
            if cancel.is_cancelled() {
                out.flush()?;
                return Err(ModelHubError::Cancelled);
            }
            out.write_all(&chunk)?;
            written += chunk.len() as u64;
            if last_report.is_none_or(|at| at.elapsed() >= PROGRESS_INTERVAL) {
                progress(Progress {
                    downloaded: base + written,
                    total: plan.total_bytes,
                });
                last_report = Some(Instant::now());
            }
        }

        out.flush()?;
        out.into_inner()
            .map_err(|err| err.into_error())?
            .sync_all()?;
        Ok(())
    }
}

/// Removes the in-progress staging folder for `folder`. Missing folders are not an error.
pub fn discard_partial(models_dir: &Path, folder: &str) -> Result<(), ModelHubError> {
    validate_component(folder)?;
    let staging_root = models_dir.join(STAGING_DIR);
    match fs::remove_dir_all(staging_root.join(folder)) {
        Ok(()) => {}
        Err(err) if err.kind() == io::ErrorKind::NotFound => {}
        Err(err) => return Err(err.into()),
    }
    // Best effort: fails harmlessly when other staging folders remain.
    let _ = fs::remove_dir(&staging_root);
    Ok(())
}

/// Rejects names that could escape their parent folder.
fn validate_component(name: &str) -> Result<(), ModelHubError> {
    if name.is_empty() || name == "." || name.contains(['/', '\\']) || name.contains("..") {
        return Err(ModelHubError::InvalidReference(format!(
            "nombre de archivo o carpeta no seguro: {name:?}"
        )));
    }
    Ok(())
}

fn already_installed(path: &Path) -> ModelHubError {
    ModelHubError::AlreadyInstalled(path.display().to_string())
}

fn part_path(staging: &Path, file: &PlannedFile) -> PathBuf {
    staging.join(format!("{}.part", file.name))
}

/// Length of `path`, or `0` when it does not exist.
fn file_len(path: &Path) -> io::Result<u64> {
    match fs::metadata(path) {
        Ok(meta) => Ok(meta.len()),
        Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(0),
        Err(err) => Err(err),
    }
}

fn remove_if_exists(path: &Path) -> io::Result<()> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(err) => Err(err),
    }
}

/// True when `part` already holds the full expected size, so no request is needed.
fn is_complete_part(part: &Path, file: &PlannedFile) -> io::Result<bool> {
    Ok(file.size_bytes > 0 && file_len(part)? >= file.size_bytes)
}

/// Length of `dest` when it already matches the planned size and SHA-256; `None` otherwise.
/// Files with neither a known size nor a checksum can never be verified, so they are fetched again.
async fn already_verified(dest: &Path, file: &PlannedFile) -> Result<Option<u64>, ModelHubError> {
    if file.size_bytes == 0 && file.sha256.is_none() {
        return Ok(None);
    }
    let len = match fs::metadata(dest) {
        Ok(meta) if meta.is_file() => meta.len(),
        Ok(_) => return Ok(None),
        Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(err) => return Err(err.into()),
    };
    if file.size_bytes > 0 && len != file.size_bytes {
        return Ok(None);
    }
    if let Some(expected) = &file.sha256
        && !sha256_file(dest.to_path_buf()).await?.eq_ignore_ascii_case(expected)
    {
        return Ok(None);
    }
    Ok(Some(len))
}

/// Verifies a fully downloaded `part` and renames it to `dest`. Failed verification deletes `part`.
async fn publish(file: &PlannedFile, part: &Path, dest: &Path) -> Result<(), ModelHubError> {
    let len = file_len(part)?;
    if file.size_bytes > 0 && len != file.size_bytes {
        remove_if_exists(part)?;
        return Err(ModelHubError::BadResponse(format!(
            "{} pesa {len} bytes; se esperaban {}",
            file.name, file.size_bytes
        )));
    }
    if let Some(expected) = &file.sha256 {
        let actual = sha256_file(part.to_path_buf()).await?;
        if !actual.eq_ignore_ascii_case(expected) {
            remove_if_exists(part)?;
            return Err(ModelHubError::ChecksumMismatch {
                file: file.name.clone(),
                expected: expected.clone(),
                actual,
            });
        }
    }
    fs::rename(part, dest)?;
    Ok(())
}

/// Lowercase hex SHA-256 of a file, hashed on the blocking pool.
async fn sha256_file(path: PathBuf) -> Result<String, ModelHubError> {
    let digest = tokio::task::spawn_blocking(move || -> io::Result<String> {
        let mut hasher = Sha256::new();
        io::copy(&mut File::open(path)?, &mut hasher)?;
        Ok(hasher
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect())
    })
    .await
    .map_err(io::Error::other)??;
    Ok(digest)
}

/// Start offset of a `Content-Range: bytes <start>-<end>/<total>` header.
fn content_range_start(response: &reqwest::Response) -> Option<u64> {
    response
        .headers()
        .get(CONTENT_RANGE)?
        .to_str()
        .ok()?
        .strip_prefix("bytes ")?
        .split('-')
        .next()?
        .parse()
        .ok()
}

/// Maps a non-success file response to the crate's error variants.
fn status_error(status: StatusCode, repo_id: &str) -> ModelHubError {
    match status {
        StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN => ModelHubError::Gated(repo_id.to_string()),
        StatusCode::NOT_FOUND => ModelHubError::RepoNotFound(repo_id.to_string()),
        s if s.is_server_error() || s == StatusCode::TOO_MANY_REQUESTS => {
            ModelHubError::Network(format!("HTTP {s} al descargar {repo_id}"))
        }
        s => ModelHubError::BadResponse(format!("HTTP {s} al descargar {repo_id}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::ModelFormat;
    use crate::reference::RepoRef;
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    const MODEL: &[u8] = b"0123456789abcdef";
    const CONFIG: &[u8] = b"{\"model_type\":\"whisper\"}";

    fn sha_hex(bytes: &[u8]) -> String {
        Sha256::digest(bytes).iter().map(|byte| format!("{byte:02x}")).collect()
    }

    fn plan(model_sha: Option<String>) -> ModelPlan {
        let files = vec![
            PlannedFile {
                name: "model.gguf".into(),
                size_bytes: MODEL.len() as u64,
                sha256: model_sha,
            },
            PlannedFile {
                name: "config.json".into(),
                size_bytes: CONFIG.len() as u64,
                sha256: None,
            },
        ];
        ModelPlan {
            repo: RepoRef { repo_id: "org/model".into(), revision: "main".into() },
            format: ModelFormat::Gguf,
            folder: "org--model".into(),
            quant: Some("Q4_K_M".into()),
            total_bytes: (MODEL.len() + CONFIG.len()) as u64,
            files,
        }
    }

    async fn serve(server: &MockServer, name: &str, body: &'static [u8]) {
        Mock::given(method("GET"))
            .and(path(format!("/org/model/resolve/main/{name}").as_str()))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(body))
            .mount(server)
            .await;
    }

    fn downloader(server: &MockServer, models: &Path) -> Downloader {
        Downloader::new(HubClient::with_base_url(&server.uri()).unwrap(), models)
    }

    #[tokio::test]
    async fn install_is_atomic_verified_and_refuses_a_second_install() {
        let server = MockServer::start().await;
        serve(&server, "model.gguf", MODEL).await;
        serve(&server, "config.json", CONFIG).await;
        let models = tempfile::tempdir().unwrap();
        let downloader = downloader(&server, models.path());
        let plan = plan(Some(sha_hex(MODEL)));

        let mut events = Vec::new();
        let dir = downloader
            .install(&plan, &CancelToken::new(), |p| events.push(p))
            .await
            .unwrap();

        assert_eq!(dir, models.path().join("org--model"));
        assert_eq!(fs::read(dir.join("model.gguf")).unwrap(), MODEL);
        assert_eq!(fs::read(dir.join("config.json")).unwrap(), CONFIG);
        assert!(!models.path().join(STAGING_DIR).exists(), "staging folder must be gone");
        let total = plan.total_bytes;
        assert_eq!(events.last(), Some(&Progress { downloaded: total, total }));

        let err = downloader
            .install(&plan, &CancelToken::new(), |_| {})
            .await
            .unwrap_err();
        assert!(matches!(err, ModelHubError::AlreadyInstalled(_)), "{err}");
    }

    #[tokio::test]
    async fn resumes_partial_download_with_range_header() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/org/model/resolve/main/model.gguf"))
            .and(header("range", "bytes=6-"))
            .respond_with(
                ResponseTemplate::new(206)
                    .insert_header("content-range", "bytes 6-15/16")
                    .set_body_bytes(&MODEL[6..]),
            )
            .expect(1)
            .mount(&server)
            .await;
        serve(&server, "config.json", CONFIG).await;
        let models = tempfile::tempdir().unwrap();
        let staging = models.path().join(STAGING_DIR).join("org--model");
        fs::create_dir_all(&staging).unwrap();
        fs::write(staging.join("model.gguf.part"), &MODEL[..6]).unwrap();

        let dir = downloader(&server, models.path())
            .install(&plan(Some(sha_hex(MODEL))), &CancelToken::new(), |_| {})
            .await
            .unwrap();

        assert_eq!(fs::read(dir.join("model.gguf")).unwrap(), MODEL);
    }

    #[tokio::test]
    async fn checksum_mismatch_discards_part_then_cancel_and_discard_partial_clean_up() {
        let server = MockServer::start().await;
        serve(&server, "model.gguf", MODEL).await;
        serve(&server, "config.json", CONFIG).await;
        let models = tempfile::tempdir().unwrap();
        let downloader = downloader(&server, models.path());
        let staging = models.path().join(STAGING_DIR).join("org--model");

        let err = downloader
            .install(&plan(Some(sha_hex(b"not the model"))), &CancelToken::new(), |_| {})
            .await
            .unwrap_err();
        assert!(
            matches!(err, ModelHubError::ChecksumMismatch { ref file, .. } if file == "model.gguf"),
            "{err}"
        );
        assert!(!staging.join("model.gguf.part").exists(), ".part must be discarded");
        assert!(!staging.join("model.gguf").exists());
        assert!(!models.path().join("org--model").exists(), "nothing may be installed");

        // Cancel right after the first file completes: the second file is never fetched.
        let cancel = CancelToken::new();
        let trigger = cancel.clone();
        let err = downloader
            .install(&plan(Some(sha_hex(MODEL))), &cancel, move |p| {
                if p.downloaded > 0 {
                    trigger.cancel();
                }
            })
            .await
            .unwrap_err();
        assert!(matches!(err, ModelHubError::Cancelled), "{err}");
        assert!(staging.join("model.gguf").exists(), "completed file is kept for resume");
        assert!(!staging.join("config.json").exists());
        assert!(!models.path().join("org--model").exists());

        discard_partial(models.path(), "org--model").unwrap();
        assert!(!staging.exists());
        assert!(!models.path().join(STAGING_DIR).exists());
        discard_partial(models.path(), "org--model").unwrap();
        let err = discard_partial(models.path(), "../escape").unwrap_err();
        assert!(matches!(err, ModelHubError::InvalidReference(_)), "{err}");
    }
}
