//! Public (account-free) Hugging Face hub client.

use std::fmt::Write as _;
use std::time::Duration;

use reqwest::StatusCode;
use serde::Deserialize;

use crate::error::ModelHubError;
use crate::reference::RepoRef;

/// Default hub origin.
pub const HUB_BASE_URL: &str = "https://huggingface.co";

/// Largest `config.json` accepted from the hub, in bytes.
const MAX_CONFIG_BYTES: usize = 2 * 1024 * 1024;
/// Largest `tokenizer_config.json` accepted from the hub, in bytes.
const MAX_TOKENIZER_CONFIG_BYTES: usize = 2 * 1024 * 1024;
/// Largest `generation_config.json` accepted from the hub, in bytes.
const MAX_GENERATION_CONFIG_BYTES: usize = 2 * 1024 * 1024;
/// Leading bytes of a GGUF file requested as its metadata header, in bytes.
pub(crate) const GGUF_HEADER_SAMPLE_BYTES: usize = 64 * 1024;
/// User-agent sent with every hub request.
const USER_AGENT: &str = "MeetingTranscriber";
/// Time allowed to establish a connection.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(20);

/// One file listed by the hub for a revision.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Sibling {
    pub rfilename: String,
    #[serde(default)]
    pub size: Option<u64>,
    #[serde(default)]
    pub lfs: Option<LfsInfo>,
}

/// Git-LFS metadata published for large files.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct LfsInfo {
    #[serde(default)]
    pub sha256: Option<String>,
    #[serde(default)]
    pub size: Option<u64>,
}

impl Sibling {
    /// Size in bytes, preferring the LFS size; `0` when unknown.
    pub fn size_bytes(&self) -> u64 {
        self.size.or(self.lfs.as_ref().and_then(|l| l.size)).unwrap_or(0)
    }

    /// Published SHA-256 when the hub provides one.
    pub fn sha256(&self) -> Option<&str> {
        self.lfs.as_ref().and_then(|l| l.sha256.as_deref())
    }
}

/// Repository metadata for one revision.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct RepoInfo {
    #[serde(rename = "id", default)]
    pub repo_id: String,
    #[serde(default)]
    pub sha: Option<String>,
    /// Raw `gated` value from the hub: `false` for open repositories.
    #[serde(default)]
    pub gated: serde_json::Value,
    #[serde(default)]
    pub siblings: Vec<Sibling>,
    /// Hub tags such as `text-generation` or `tool-use`.
    #[serde(default)]
    pub tags: Vec<String>,
}

/// Anonymous client for the public Hugging Face API.
#[derive(Debug, Clone)]
pub struct HubClient {
    http: reqwest::Client,
    base: url::Url,
}

impl HubClient {
    /// Client pointed at `https://huggingface.co`.
    pub fn new() -> Result<Self, ModelHubError> {
        Self::with_base_url(HUB_BASE_URL)
    }

    /// Client pointed at an arbitrary origin (used by tests).
    pub fn with_base_url(base: &str) -> Result<Self, ModelHubError> {
        let base = url::Url::parse(base)
            .map_err(|err| ModelHubError::Network(format!("URL del hub inválida: {err}")))?;
        let http = reqwest::Client::builder()
            .user_agent(USER_AGENT)
            .connect_timeout(CONNECT_TIMEOUT)
            .build()?;
        Ok(Self { http, base })
    }

    /// Public URL of one file in the reference's revision (`.../resolve/{rev}/{name}`).
    pub fn file_url(&self, reference: &RepoRef, name: &str) -> String {
        format!(
            "{}/{}/resolve/{}/{}",
            self.origin(),
            encode_path(&reference.repo_id),
            encode_segment(&reference.revision),
            encode_path(name)
        )
    }

    /// Underlying HTTP client, shared with the downloader.
    pub(crate) fn http(&self) -> &reqwest::Client {
        &self.http
    }

    fn origin(&self) -> &str {
        self.base.as_str().trim_end_matches('/')
    }

    /// Fetches `/api/models/{repo}/revision/{rev}?blobs=true`.
    /// Rejects anything but `gated: false` with [`ModelHubError::Gated`].
    pub async fn repo_info(&self, reference: &RepoRef) -> Result<RepoInfo, ModelHubError> {
        let url = format!(
            "{}/api/models/{}/revision/{}?blobs=true",
            self.origin(),
            encode_path(&reference.repo_id),
            encode_segment(&reference.revision)
        );
        let response = ensure_success(self.http.get(&url).send().await?, &reference.repo_id)?;
        let info: RepoInfo = response
            .json()
            .await
            .map_err(|err| ModelHubError::BadResponse(err.to_string()))?;
        if !info.gated.is_null() && info.gated != serde_json::Value::Bool(false) {
            return Err(ModelHubError::Gated(reference.repo_id.clone()));
        }
        Ok(info)
    }

    /// Fetches the optional `config.json`; `Ok(None)` when the repository has none.
    pub async fn fetch_config(
        &self,
        reference: &RepoRef,
    ) -> Result<Option<serde_json::Value>, ModelHubError> {
        self.fetch_json_file(reference, "config.json", MAX_CONFIG_BYTES).await
    }

    /// Fetches the optional `tokenizer_config.json`; `Ok(None)` when the repository has none.
    pub async fn fetch_tokenizer_config(
        &self,
        reference: &RepoRef,
    ) -> Result<Option<serde_json::Value>, ModelHubError> {
        self.fetch_json_file(reference, "tokenizer_config.json", MAX_TOKENIZER_CONFIG_BYTES)
            .await
    }

    /// Fetches the optional `generation_config.json`; `Ok(None)` when the repository has none.
    pub async fn fetch_generation_config(
        &self,
        reference: &RepoRef,
    ) -> Result<Option<serde_json::Value>, ModelHubError> {
        self.fetch_json_file(reference, "generation_config.json", MAX_GENERATION_CONFIG_BYTES)
            .await
    }

    /// Fetches the leading [`GGUF_HEADER_SAMPLE_BYTES`] of one GGUF file, where the chat template lives.
    /// `Ok(None)` when the hub refuses the range (for example 416 on a tiny file) or answers with a
    /// non-2xx status. At most the sample size is read, even when the hub ignores the range request.
    pub async fn fetch_gguf_header_sample(
        &self,
        reference: &RepoRef,
        filename: &str,
    ) -> Result<Option<Vec<u8>>, ModelHubError> {
        let range = format!("bytes=0-{}", GGUF_HEADER_SAMPLE_BYTES - 1);
        let mut response = self
            .http
            .get(self.file_url(reference, filename))
            .header(reqwest::header::RANGE, range)
            .send()
            .await?;
        if !response.status().is_success() {
            return Ok(None);
        }
        let mut sample = Vec::with_capacity(GGUF_HEADER_SAMPLE_BYTES);
        while let Some(chunk) = response.chunk().await? {
            let room = GGUF_HEADER_SAMPLE_BYTES - sample.len();
            if chunk.len() >= room {
                sample.extend_from_slice(&chunk[..room]);
                break;
            }
            sample.extend_from_slice(&chunk);
        }
        Ok(Some(sample))
    }

    /// Shared reader for small optional JSON files at the repository root.
    async fn fetch_json_file(
        &self,
        reference: &RepoRef,
        name: &str,
        max_bytes: usize,
    ) -> Result<Option<serde_json::Value>, ModelHubError> {
        let response = self.http.get(self.file_url(reference, name)).send().await?;
        if response.status() == StatusCode::NOT_FOUND {
            return Ok(None);
        }
        let mut response = ensure_success(response, &reference.repo_id)?;
        let too_large = || ModelHubError::BadResponse(format!("{name} supera {max_bytes} bytes"));
        if response
            .content_length()
            .is_some_and(|len| len > max_bytes as u64)
        {
            return Err(too_large());
        }
        let mut body = Vec::new();
        while let Some(chunk) = response.chunk().await? {
            if body.len() + chunk.len() > max_bytes {
                return Err(too_large());
            }
            body.extend_from_slice(&chunk);
        }
        serde_json::from_slice(&body)
            .map(Some)
            .map_err(|err| ModelHubError::BadResponse(format!("{name} no es JSON: {err}")))
    }
}

/// Maps non-2xx hub answers to the crate's error variants.
fn ensure_success(
    response: reqwest::Response,
    repo_id: &str,
) -> Result<reqwest::Response, ModelHubError> {
    let status = response.status();
    if status.is_success() {
        return Ok(response);
    }
    Err(match status {
        StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN | StatusCode::NOT_FOUND => {
            ModelHubError::RepoNotFound(repo_id.to_string())
        }
        s if s.is_server_error() => ModelHubError::Network(format!("HTTP {s} desde el hub")),
        s => ModelHubError::BadResponse(format!("HTTP {s} para {repo_id}")),
    })
}

/// Percent-encodes one path segment; everything outside the URL unreserved set is escaped.
fn encode_segment(segment: &str) -> String {
    let mut out = String::with_capacity(segment.len());
    for byte in segment.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'.' | b'_' | b'~') {
            out.push(char::from(byte));
        } else {
            let _ = write!(out, "%{byte:02X}");
        }
    }
    out
}

/// Encodes each `/`-separated part of a path, keeping the separators.
fn encode_path(path: &str) -> String {
    path.split('/').map(encode_segment).collect::<Vec<_>>().join("/")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use wiremock::matchers::{header, method, path, query_param};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn reference(repo_id: &str) -> RepoRef {
        RepoRef { repo_id: repo_id.into(), revision: "main".into() }
    }

    #[tokio::test]
    async fn repo_info_parses_siblings_for_open_repo() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/models/org/model/revision/main"))
            .and(query_param("blobs", "true"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "org/model",
                "sha": "abc",
                "gated": false,
                "siblings": [{ "rfilename": "model.gguf", "size": 10 }]
            })))
            .mount(&server)
            .await;

        let client = HubClient::with_base_url(&server.uri()).unwrap();
        let info = client.repo_info(&reference("org/model")).await.unwrap();
        assert_eq!(info.sha.as_deref(), Some("abc"));
        assert_eq!(info.siblings[0].rfilename, "model.gguf");
        assert_eq!(info.siblings[0].size_bytes(), 10);
        assert_eq!(
            client.file_url(&reference("org/model"), "config.json"),
            format!("{}/org/model/resolve/main/config.json", server.uri())
        );
    }

    #[tokio::test]
    async fn repo_info_rejects_gated_and_maps_missing_repo_to_not_found() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/models/org/gated/revision/main"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({ "id": "org/gated", "gated": "manual" })))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/api/models/org/missing/revision/main"))
            .respond_with(ResponseTemplate::new(404))
            .mount(&server)
            .await;

        let client = HubClient::with_base_url(&server.uri()).unwrap();
        let gated = client.repo_info(&reference("org/gated")).await.unwrap_err();
        assert!(matches!(gated, ModelHubError::Gated(ref r) if r == "org/gated"), "{gated}");
        let missing = client.repo_info(&reference("org/missing")).await.unwrap_err();
        assert!(matches!(missing, ModelHubError::RepoNotFound(_)), "{missing}");
    }

    #[tokio::test]
    async fn fetch_config_parses_json_none_when_missing_and_caps_size() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/org/model/resolve/main/config.json"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({ "model_type": "whisper" })))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/org/none/resolve/main/config.json"))
            .respond_with(ResponseTemplate::new(404))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/org/huge/resolve/main/config.json"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(vec![b' '; MAX_CONFIG_BYTES + 1]))
            .mount(&server)
            .await;

        let client = HubClient::with_base_url(&server.uri()).unwrap();
        let config = client.fetch_config(&reference("org/model")).await.unwrap();
        assert_eq!(config, Some(json!({ "model_type": "whisper" })));
        assert_eq!(client.fetch_config(&reference("org/none")).await.unwrap(), None);
        let huge = client.fetch_config(&reference("org/huge")).await.unwrap_err();
        assert!(matches!(huge, ModelHubError::BadResponse(_)), "{huge}");
    }

    #[tokio::test]
    async fn repo_info_parses_tags_and_defaults_them_to_empty_when_omitted() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/models/org/tagged/revision/main"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "org/tagged",
                "gated": false,
                "tags": ["text-generation", "tool-use"]
            })))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/api/models/org/untagged/revision/main"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({ "id": "org/untagged", "gated": false })))
            .mount(&server)
            .await;

        let client = HubClient::with_base_url(&server.uri()).unwrap();
        let tagged = client.repo_info(&reference("org/tagged")).await.unwrap();
        assert_eq!(tagged.tags, vec!["text-generation".to_string(), "tool-use".to_string()]);
        let untagged = client.repo_info(&reference("org/untagged")).await.unwrap();
        assert!(untagged.tags.is_empty());
    }

    #[tokio::test]
    async fn fetch_tokenizer_config_parses_json_none_when_missing_and_caps_size() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/org/model/resolve/main/tokenizer_config.json"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({ "chat_template": "{{ tools }}" })))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/org/none/resolve/main/tokenizer_config.json"))
            .respond_with(ResponseTemplate::new(404))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/org/huge/resolve/main/tokenizer_config.json"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(vec![b' '; MAX_TOKENIZER_CONFIG_BYTES + 1]))
            .mount(&server)
            .await;

        let client = HubClient::with_base_url(&server.uri()).unwrap();
        let config = client.fetch_tokenizer_config(&reference("org/model")).await.unwrap();
        assert_eq!(config, Some(json!({ "chat_template": "{{ tools }}" })));
        assert_eq!(client.fetch_tokenizer_config(&reference("org/none")).await.unwrap(), None);
        let huge = client.fetch_tokenizer_config(&reference("org/huge")).await.unwrap_err();
        assert!(matches!(huge, ModelHubError::BadResponse(_)), "{huge}");
    }

    #[tokio::test]
    async fn fetch_generation_config_parses_json_none_when_missing_and_caps_size() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/org/model/resolve/main/generation_config.json"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({ "temperature": 0.7, "top_k": 20 })))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/org/none/resolve/main/generation_config.json"))
            .respond_with(ResponseTemplate::new(404))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/org/huge/resolve/main/generation_config.json"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(vec![b' '; MAX_GENERATION_CONFIG_BYTES + 1]))
            .mount(&server)
            .await;

        let client = HubClient::with_base_url(&server.uri()).unwrap();
        let config = client.fetch_generation_config(&reference("org/model")).await.unwrap();
        assert_eq!(config, Some(json!({ "temperature": 0.7, "top_k": 20 })));
        assert_eq!(client.fetch_generation_config(&reference("org/none")).await.unwrap(), None);
        let huge = client.fetch_generation_config(&reference("org/huge")).await.unwrap_err();
        assert!(matches!(huge, ModelHubError::BadResponse(_)), "{huge}");
    }

    #[tokio::test]
    async fn fetch_gguf_header_sample_asks_for_the_leading_range_and_treats_refusals_as_absent() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/org/model/resolve/main/model.gguf"))
            .and(header("range", "bytes=0-65535"))
            .respond_with(ResponseTemplate::new(206).set_body_bytes(b"GGUF-header".to_vec()))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/org/full/resolve/main/model.gguf"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(vec![7u8; GGUF_HEADER_SAMPLE_BYTES + 4096]))
            .mount(&server)
            .await;
        for (route, status) in [
            ("/org/refused/resolve/main/model.gguf", 416),
            ("/org/none/resolve/main/model.gguf", 404),
            ("/org/broken/resolve/main/model.gguf", 500),
        ] {
            Mock::given(method("GET"))
                .and(path(route))
                .respond_with(ResponseTemplate::new(status))
                .mount(&server)
                .await;
        }

        let client = HubClient::with_base_url(&server.uri()).unwrap();
        let sample = client
            .fetch_gguf_header_sample(&reference("org/model"), "model.gguf")
            .await
            .unwrap();
        assert_eq!(sample, Some(b"GGUF-header".to_vec()));
        let capped = client
            .fetch_gguf_header_sample(&reference("org/full"), "model.gguf")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(capped.len(), GGUF_HEADER_SAMPLE_BYTES);
        for repo in ["org/refused", "org/none", "org/broken"] {
            let absent = client
                .fetch_gguf_header_sample(&reference(repo), "model.gguf")
                .await
                .unwrap();
            assert_eq!(absent, None, "{repo}");
        }
    }
}
