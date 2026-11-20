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
/// Growing leading windows of a GGUF file searched for its embedded chat template, in bytes.
const GGUF_CHAT_TEMPLATE_WINDOWS: [usize; 5] = [
    GGUF_HEADER_SAMPLE_BYTES,
    512 * 1024,
    4 * 1024 * 1024,
    16 * 1024 * 1024,
    24 * 1024 * 1024,
];
/// Largest `tokenizer.chat_template` value read from a GGUF file, in bytes.
const MAX_GGUF_CHAT_TEMPLATE_BYTES: usize = 32 * 1024;
/// GGUF metadata key holding the embedded chat template.
const GGUF_CHAT_TEMPLATE_KEY: &[u8] = b"tokenizer.chat_template";
/// GGUF metadata value type code for strings.
const GGUF_TYPE_STRING: u32 = 8;
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

    /// Reads the chat template embedded in one GGUF file's metadata (`tokenizer.chat_template`).
    ///
    /// Requests growing leading windows of the file and stops at the first window that contains the key.
    /// `Ok(None)` when no readable template is found: refused ranges (404/416), other non-success
    /// statuses, read errors, a missing or invalid length, or a value cut short by the window. Each window
    /// replaces the previous buffer, so at most one window is held in memory.
    pub async fn fetch_gguf_chat_template(
        &self,
        reference: &RepoRef,
        filename: &str,
    ) -> Result<Option<String>, ModelHubError> {
        let url = self.file_url(reference, filename);
        for window in GGUF_CHAT_TEMPLATE_WINDOWS {
            let Some(bytes) = self.fetch_leading_bytes(&url, window).await else {
                return Ok(None);
            };
            match scan_gguf_chat_template(&bytes) {
                TemplateScan::Found(template) => return Ok(Some(template)),
                TemplateScan::Invalid => return Ok(None),
                // The file ended inside this window, so a larger window cannot contain more of it.
                TemplateScan::Absent | TemplateScan::Truncated if bytes.len() < window => {
                    return Ok(None);
                }
                // Key not seen yet, or its value crosses this window's end: try the next, larger window.
                TemplateScan::Absent | TemplateScan::Truncated => {}
            }
        }
        Ok(None)
    }

    /// Reads at most `window` leading bytes of `url`. `None` when the hub refuses the range, answers with a
    /// non-2xx status or the transfer fails. Reading stops at `window` even if the hub ignores the range.
    async fn fetch_leading_bytes(&self, url: &str, window: usize) -> Option<Vec<u8>> {
        let range = format!("bytes=0-{}", window - 1);
        let mut response = self
            .http
            .get(url)
            .header(reqwest::header::RANGE, range)
            .send()
            .await
            .ok()?;
        if !response.status().is_success() {
            return None;
        }
        let mut body = Vec::new();
        while body.len() < window {
            match response.chunk().await {
                Ok(Some(chunk)) => {
                    let room = window - body.len();
                    body.extend_from_slice(&chunk[..chunk.len().min(room)]);
                }
                Ok(None) => break,
                Err(_) => return None,
            }
        }
        Some(body)
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

/// Result of scanning one window of GGUF metadata for `tokenizer.chat_template`.
#[derive(Debug, PartialEq, Eq)]
enum TemplateScan {
    /// The key does not appear (completely) in this window.
    Absent,
    /// The key is present, but its value runs past the end of this window.
    Truncated,
    /// The key is present with a zero length, so there is no template to read.
    Invalid,
    /// The template value, capped at [`MAX_GGUF_CHAT_TEMPLATE_BYTES`].
    Found(String),
}

/// Parses the `tokenizer.chat_template` string entry in a metadata window. The key must be preceded by its
/// own u64 length and followed by the string type code, so values and other keys that merely contain the
/// text (for example `tokenizer.chat_template.tool_use`) are skipped.
fn scan_gguf_chat_template(window: &[u8]) -> TemplateScan {
    const PREFIX_LEN: usize = 8;
    let key_len = GGUF_CHAT_TEMPLATE_KEY.len();
    let key_prefix = (key_len as u64).to_le_bytes();
    for (at, candidate) in window.windows(key_len).enumerate() {
        if candidate != GGUF_CHAT_TEMPLATE_KEY || at < PREFIX_LEN {
            continue;
        }
        if window[at - PREFIX_LEN..at] != key_prefix {
            continue;
        }
        let rest = &window[at + key_len..];
        let Some((kind, rest)) = rest.split_first_chunk::<4>() else {
            return TemplateScan::Absent;
        };
        if u32::from_le_bytes(*kind) != GGUF_TYPE_STRING {
            continue;
        }
        let Some((len, payload)) = rest.split_first_chunk::<8>() else {
            return TemplateScan::Absent;
        };
        let len = u64::from_le_bytes(*len);
        if len == 0 {
            return TemplateScan::Invalid;
        }
        let wanted = usize::try_from(len).unwrap_or(usize::MAX).min(MAX_GGUF_CHAT_TEMPLATE_BYTES);
        if payload.len() < wanted {
            return TemplateScan::Truncated;
        }
        return TemplateScan::Found(String::from_utf8_lossy(&payload[..wanted]).into_owned());
    }
    TemplateScan::Absent
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

    /// One GGUF string metadata entry: u64 key length, key, u32 type 8, u64 value length, value.
    fn gguf_string_entry(key: &str, value: &[u8]) -> Vec<u8> {
        let mut entry = (key.len() as u64).to_le_bytes().to_vec();
        entry.extend_from_slice(key.as_bytes());
        entry.extend_from_slice(&8u32.to_le_bytes());
        entry.extend_from_slice(&(value.len() as u64).to_le_bytes());
        entry.extend_from_slice(value);
        entry
    }

    const GEMMA_TEMPLATE: &str = "{%- macro format_parameters(p) %}<|\"|>{{ p }}<|\"|>{%- endmacro %}{{ tools }}";

    #[tokio::test]
    async fn chat_template_past_the_first_window_is_found_on_the_second_request() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/org/model/resolve/main/model.gguf"))
            .and(header("range", "bytes=0-65535"))
            .respond_with(ResponseTemplate::new(206).set_body_bytes(vec![0u8; 64 * 1024]))
            .expect(1)
            .mount(&server)
            .await;
        let mut second = vec![0u8; 100 * 1024];
        second.extend(gguf_string_entry("tokenizer.chat_template", GEMMA_TEMPLATE.as_bytes()));
        Mock::given(method("GET"))
            .and(path("/org/model/resolve/main/model.gguf"))
            .and(header("range", "bytes=0-524287"))
            .respond_with(ResponseTemplate::new(206).set_body_bytes(second))
            .expect(1)
            .mount(&server)
            .await;

        let client = HubClient::with_base_url(&server.uri()).unwrap();
        let template = client
            .fetch_gguf_chat_template(&reference("org/model"), "model.gguf")
            .await
            .unwrap();
        assert_eq!(template.as_deref(), Some(GEMMA_TEMPLATE));
        // Each window is requested at most once and the search stops at the first hit.
        assert_eq!(server.received_requests().await.unwrap().len(), 2);
    }

    #[tokio::test]
    async fn chat_template_bytes_are_extracted_exactly_and_capped_at_32_kib() {
        let server = MockServer::start().await;
        let mut body = b"GGUF\x03\x00\x00\x00".to_vec();
        body.extend(gguf_string_entry("general.name", b"demo"));
        body.extend(gguf_string_entry("tokenizer.chat_template", GEMMA_TEMPLATE.as_bytes()));
        body.extend(gguf_string_entry("tokenizer.ggml.model", b"llama"));
        Mock::given(method("GET"))
            .and(path("/org/exact/resolve/main/model.gguf"))
            .respond_with(ResponseTemplate::new(206).set_body_bytes(body))
            .mount(&server)
            .await;
        let huge_template = "t".repeat(MAX_GGUF_CHAT_TEMPLATE_BYTES + 100);
        let capped = gguf_string_entry("tokenizer.chat_template", huge_template.as_bytes());
        Mock::given(method("GET"))
            .and(path("/org/capped/resolve/main/model.gguf"))
            .respond_with(ResponseTemplate::new(206).set_body_bytes(capped))
            .mount(&server)
            .await;

        let client = HubClient::with_base_url(&server.uri()).unwrap();
        let exact = client
            .fetch_gguf_chat_template(&reference("org/exact"), "model.gguf")
            .await
            .unwrap();
        assert_eq!(exact.as_deref(), Some(GEMMA_TEMPLATE));
        let cut = client
            .fetch_gguf_chat_template(&reference("org/capped"), "model.gguf")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(cut.len(), MAX_GGUF_CHAT_TEMPLATE_BYTES);
        assert!(cut.bytes().all(|b| b == b't'));
    }

    #[tokio::test]
    async fn refused_or_failed_gguf_reads_yield_no_template() {
        let server = MockServer::start().await;
        for (route, status) in [
            ("/org/none/resolve/main/model.gguf", 404),
            ("/org/refused/resolve/main/model.gguf", 416),
            ("/org/broken/resolve/main/model.gguf", 500),
        ] {
            Mock::given(method("GET"))
                .and(path(route))
                .respond_with(ResponseTemplate::new(status))
                .mount(&server)
                .await;
        }

        let client = HubClient::with_base_url(&server.uri()).unwrap();
        for repo in ["org/none", "org/refused", "org/broken"] {
            let template = client
                .fetch_gguf_chat_template(&reference(repo), "model.gguf")
                .await
                .unwrap();
            assert_eq!(template, None, "{repo}");
        }
    }

    #[tokio::test]
    async fn key_with_a_bogus_length_yields_no_template_instead_of_an_error() {
        let server = MockServer::start().await;
        // Zero length, then a length longer than the bytes the file has left.
        let mut zero = b"GGUF".to_vec();
        zero.extend(gguf_string_entry("tokenizer.chat_template", b""));
        let mut overrun = b"GGUF".to_vec();
        overrun.extend(("tokenizer.chat_template".len() as u64).to_le_bytes());
        overrun.extend_from_slice(b"tokenizer.chat_template");
        overrun.extend_from_slice(&8u32.to_le_bytes());
        overrun.extend_from_slice(&(1u64 << 40).to_le_bytes());
        overrun.extend_from_slice(b"short");
        // The key is the last thing in the file: no length at all.
        let mut missing = b"GGUF".to_vec();
        missing.extend(("tokenizer.chat_template".len() as u64).to_le_bytes());
        missing.extend_from_slice(b"tokenizer.chat_template");
        for (repo, body) in [("org/zero", zero), ("org/overrun", overrun), ("org/missing", missing)] {
            Mock::given(method("GET"))
                .and(path(format!("/{repo}/resolve/main/model.gguf")))
                .respond_with(ResponseTemplate::new(206).set_body_bytes(body))
                .mount(&server)
                .await;
        }

        let client = HubClient::with_base_url(&server.uri()).unwrap();
        for repo in ["org/zero", "org/overrun", "org/missing"] {
            let template = client
                .fetch_gguf_chat_template(&reference(repo), "model.gguf")
                .await
                .unwrap();
            assert_eq!(template, None, "{repo}");
        }
    }

    #[test]
    fn scan_skips_longer_keys_and_non_string_entries_with_the_same_text() {
        let mut window = gguf_string_entry("tokenizer.chat_template.tool_use", b"plain");
        window.extend(("tokenizer.chat_template".len() as u64).to_le_bytes());
        window.extend_from_slice(b"tokenizer.chat_template");
        window.extend_from_slice(&4u32.to_le_bytes());
        window.extend_from_slice(&[0; 12]);
        window.extend(gguf_string_entry("tokenizer.chat_template", b"{{ tools }}"));
        assert_eq!(scan_gguf_chat_template(&window), TemplateScan::Found("{{ tools }}".into()));
    }
}
