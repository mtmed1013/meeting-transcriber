//! OpenAI-compatible chat client. Enforces the input budget before any network call and sends
//! `chat_template_kwargs.enable_thinking` only to loopback endpoints.

use std::fmt;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::budget::estimate_messages_tokens;
use crate::error::AiError;

/// Connection and sampling settings for one chat model.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ModelConfig {
    /// Base URL such as `http://127.0.0.1:8080/v1`; `/chat/completions` is appended.
    pub endpoint: String,
    /// Bearer token. Empty means no `Authorization` header.
    pub api_key: String,
    pub model: String,
    pub timeout_seconds: u64,
    pub temperature: f32,
    /// Nucleus sampling cutoff. Omitted from the request when `None`.
    pub top_p: Option<f32>,
    /// Top-k sampling cutoff. Only sent to loopback endpoints, and only when `Some`.
    pub top_k: Option<u32>,
    /// Repetition penalty. Only sent to loopback endpoints, and only when `Some`.
    pub repetition_penalty: Option<f32>,
    /// Presence penalty. Only sent to loopback endpoints, and only when `Some`.
    pub presence_penalty: Option<f32>,
    /// Total context window of the model, in tokens.
    pub context_limit_tokens: usize,
    /// Tokens reserved for the reply; the rest of the window is the input budget.
    pub max_output_tokens: usize,
    /// Requests the model's thinking mode. Only sent to loopback endpoints, and only when the
    /// model declares support for the switch.
    pub thinking: bool,
    /// Whether the model's chat template understands `enable_thinking`.
    #[serde(default)]
    pub supports_thinking_toggle: bool,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            endpoint: "http://127.0.0.1:8080/v1".to_string(),
            api_key: String::new(),
            model: String::new(),
            timeout_seconds: 120,
            // Safer fallback for small models when no HF profile is present.
            temperature: 0.2,
            top_p: None,
            top_k: None,
            repetition_penalty: None,
            presence_penalty: None,
            context_limit_tokens: 32768,
            max_output_tokens: 4096,
            thinking: false,
            supports_thinking_toggle: false,
        }
    }
}

impl ModelConfig {
    /// Tokens available for the prompt: the context window minus the reserved reply tokens.
    pub fn input_budget_tokens(&self) -> usize {
        self.context_limit_tokens.saturating_sub(self.max_output_tokens)
    }
}

impl fmt::Debug for ModelConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ModelConfig")
            .field("endpoint", &self.endpoint)
            .field("api_key", &if self.api_key.is_empty() { "<empty>" } else { "<redacted>" })
            .field("model", &self.model)
            .field("timeout_seconds", &self.timeout_seconds)
            .field("temperature", &self.temperature)
            .field("top_p", &self.top_p)
            .field("top_k", &self.top_k)
            .field("repetition_penalty", &self.repetition_penalty)
            .field("presence_penalty", &self.presence_penalty)
            .field("context_limit_tokens", &self.context_limit_tokens)
            .field("max_output_tokens", &self.max_output_tokens)
            .field("thinking", &self.thinking)
            .finish()
    }
}

/// Role of one chat message.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ChatRole {
    System,
    User,
    Assistant,
}

/// One tool-calling reply: the assistant message plus the server's finish reason.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolReply {
    pub message: Value,
    pub finish_reason: Option<String>,
}

impl ToolReply {
    /// Reply text; empty when the server sent only tool calls.
    pub fn content(&self) -> &str {
        self.message
            .get("content")
            .and_then(Value::as_str)
            .unwrap_or_default()
    }

    /// Native tool calls, when the server emitted structured ones.
    pub fn tool_calls(&self) -> &[Value] {
        self.message
            .get("tool_calls")
            .and_then(Value::as_array)
            .map(Vec::as_slice)
            .unwrap_or_default()
    }
}

/// One message of a chat completion request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self { role: ChatRole::System, content: content.into() }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self { role: ChatRole::User, content: content.into() }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self { role: ChatRole::Assistant, content: content.into() }
    }
}

/// Client for an OpenAI-compatible `/chat/completions` endpoint.
#[derive(Debug)]
pub struct OpenAiCompatibleProvider {
    config: ModelConfig,
    client: reqwest::Client,
    completions_url: String,
    loopback: bool,
}

impl OpenAiCompatibleProvider {
    /// Validates the endpoint and builds the HTTP client. Performs no network I/O.
    pub fn new(config: ModelConfig) -> Result<Self, AiError> {
        let parsed = url::Url::parse(&config.endpoint)
            .map_err(|err| AiError::EndpointUnreachable(format!("endpoint inválido: {err}")))?;
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(config.timeout_seconds.max(1)))
            .build()
            .map_err(map_reqwest_error)?;
        Ok(Self {
            loopback: is_loopback_host(parsed.host_str()),
            completions_url: format!("{}/chat/completions", config.endpoint.trim_end_matches('/')),
            config,
            client,
        })
    }

    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Sends `messages` and returns the assistant reply text.
    ///
    /// Fails with [`AiError::ContextOverflow`] before any network call when the estimated prompt
    /// exceeds [`ModelConfig::input_budget_tokens`].
    pub async fn chat_complete(&self, messages: &[ChatMessage]) -> Result<String, AiError> {
        let estimated_tokens = estimate_messages_tokens(messages.iter().map(|m| m.content.as_str()));
        let budget_tokens = self.config.input_budget_tokens();
        if estimated_tokens > budget_tokens {
            return Err(AiError::ContextOverflow { estimated_tokens, budget_tokens });
        }

        let body = request_body(&self.config, messages, self.loopback);
        let text = self.post_completion(&body).await?;

        let parsed: ChatResponse = serde_json::from_str(&text)
            .map_err(|err| AiError::InvalidResponse(format!("JSON de completions: {err}")))?;
        parsed
            .choices
            .into_iter()
            .find_map(|choice| choice.message.content)
            .ok_or_else(|| AiError::InvalidResponse("la respuesta no trae contenido".to_string()))
    }

    /// Sends raw OpenAI-style `messages` with function `tools` and returns the assistant message
    /// (`choices[0].message`), so the caller can read its `tool_calls`.
    ///
    /// Fails with [`AiError::ContextOverflow`] before any network call when the estimated size of
    /// the message contents, the serialized `tools`, and the serialized `tool_choice` exceeds
    /// [`ModelConfig::input_budget_tokens`].
    pub async fn chat_with_tools(
        &self,
        messages: &[Value],
        tools: &[Value],
        tool_choice: Value,
    ) -> Result<ToolReply, AiError> {
        let contents = messages.iter().filter_map(|m| m.get("content").and_then(Value::as_str));
        // The tool schema and tool choice travel in the same request, so they count against the budget.
        let tool_texts: Vec<String> = tools.iter().map(Value::to_string).collect();
        let choice_text = tool_choice.to_string();
        let parts = contents
            .chain(tool_texts.iter().map(String::as_str))
            .chain(std::iter::once(choice_text.as_str()));
        let estimated_tokens = estimate_messages_tokens(parts);
        let budget_tokens = self.config.input_budget_tokens();
        if estimated_tokens > budget_tokens {
            return Err(AiError::ContextOverflow { estimated_tokens, budget_tokens });
        }

        let mut body = tools_request_body(&self.config, messages, tools, tool_choice, self.loopback);
        let reply = self.send_tool_request(&body).await?;
        // A reply cut off at `max_tokens` usually loses its tool call: give it one wider attempt.
        if reply.finish_reason.as_deref() == Some("length") {
            let widened = self
                .config
                .max_output_tokens
                .saturating_mul(2)
                .min(MAX_RETRY_OUTPUT_TOKENS);
            if widened > self.config.max_output_tokens {
                body["max_tokens"] = json!(widened);
                if let Ok(retry) = self.send_tool_request(&body).await {
                    return Ok(retry);
                }
            }
        }
        Ok(reply)
    }

    /// Sends one tool request and splits the assistant message from its finish reason.
    async fn send_tool_request(&self, body: &Value) -> Result<ToolReply, AiError> {
        let text = self.post_completion(body).await?;
        let parsed: Value = serde_json::from_str(&text)
            .map_err(|err| AiError::InvalidResponse(format!("JSON de completions: {err}")))?;
        let message = parsed
            .pointer("/choices/0/message")
            .filter(|message| message.is_object())
            .cloned()
            .ok_or_else(|| {
                AiError::InvalidResponse("la respuesta no trae choices[0].message".to_string())
            })?;
        let finish_reason = parsed
            .pointer("/choices/0/finish_reason")
            .and_then(Value::as_str)
            .map(str::to_string);
        Ok(ToolReply { message, finish_reason })
    }

    /// POSTs `body` to the completions URL and returns the response text of a successful reply.
    async fn post_completion(&self, body: &Value) -> Result<String, AiError> {
        let mut request = self.client.post(&self.completions_url).json(body);
        if !self.config.api_key.is_empty() {
            request = request.bearer_auth(&self.config.api_key);
        }
        let response = request.send().await.map_err(map_reqwest_error)?;
        let status = response.status();
        let text = response.text().await.map_err(map_reqwest_error)?;
        if !status.is_success() {
            let snippet: String = text.chars().take(300).collect();
            return Err(AiError::RequestFailed(format!("HTTP {}: {snippet}", status.as_u16())));
        }
        Ok(text)
    }
}

/// Hard ceiling for the one wider retry after a truncated reply.
const MAX_RETRY_OUTPUT_TOKENS: usize = 8192;

/// Whether `host` is a loopback address whose server accepts `chat_template_kwargs`.
fn is_loopback_host(host: Option<&str>) -> bool {
    matches!(
        host.map(str::to_ascii_lowercase).as_deref(),
        Some("127.0.0.1" | "localhost" | "[::1]" | "::1")
    )
}

fn map_reqwest_error(err: reqwest::Error) -> AiError {
    if err.is_connect() || err.is_timeout() {
        AiError::EndpointUnreachable(err.to_string())
    } else {
        AiError::RequestFailed(err.to_string())
    }
}

#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: &'a [ChatMessage],
    temperature: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    repetition_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    presence_penalty: Option<f32>,
    max_tokens: usize,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    chat_template_kwargs: Option<ChatTemplateKwargs>,
}

#[derive(Serialize)]
struct ChatTemplateKwargs {
    enable_thinking: bool,
}

/// Body of a tool-calling request. `top_p` is added when set. The loopback-only fields
/// (`top_k`, `repetition_penalty`, `presence_penalty`, `chat_template_kwargs`) are added only for
/// loopback endpoints.
fn tools_request_body(
    config: &ModelConfig,
    messages: &[Value],
    tools: &[Value],
    tool_choice: Value,
    loopback: bool,
) -> Value {
    let mut body = json!({
        "model": config.model,
        "messages": messages,
        "tools": tools,
        "tool_choice": tool_choice,
        "temperature": config.temperature,
        "max_tokens": config.max_output_tokens,
        "stream": false,
    });
    if let Some(top_p) = config.top_p {
        body["top_p"] = json!(top_p);
    }
    if loopback {
        if let Some(top_k) = config.top_k {
            body["top_k"] = json!(top_k);
        }
        if let Some(repetition_penalty) = config.repetition_penalty {
            body["repetition_penalty"] = json!(repetition_penalty);
        }
        if let Some(presence_penalty) = config.presence_penalty {
            body["presence_penalty"] = json!(presence_penalty);
        }
        if config.supports_thinking_toggle {
            body["chat_template_kwargs"] = json!({ "enable_thinking": config.thinking });
        }
    }
    body
}

fn request_body(config: &ModelConfig, messages: &[ChatMessage], loopback: bool) -> Value {
    let request = ChatRequest {
        model: &config.model,
        messages,
        temperature: config.temperature,
        top_p: config.top_p,
        top_k: loopback.then_some(config.top_k).flatten(),
        repetition_penalty: loopback.then_some(config.repetition_penalty).flatten(),
        presence_penalty: loopback.then_some(config.presence_penalty).flatten(),
        max_tokens: config.max_output_tokens,
        stream: false,
        chat_template_kwargs: (loopback && config.supports_thinking_toggle)
            .then_some(ChatTemplateKwargs { enable_thinking: config.thinking }),
    };
    serde_json::to_value(request).expect("serializing a chat request cannot fail")
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Deserialize)]
struct ChatChoice {
    message: ChatResponseMessage,
}

#[derive(Deserialize)]
struct ChatResponseMessage {
    content: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use wiremock::matchers::{body_partial_json, header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn config_for(server: &MockServer) -> ModelConfig {
        ModelConfig {
            endpoint: format!("{}/v1", server.uri()),
            model: "test-model".to_string(),
            ..ModelConfig::default()
        }
    }

    fn completion(content: &str) -> ResponseTemplate {
        ResponseTemplate::new(200).set_body_json(json!({
            "choices": [{ "message": { "role": "assistant", "content": content } }]
        }))
    }

    #[test]
    fn loopback_hosts_are_recognized_and_others_are_not() {
        for endpoint in [
            "http://127.0.0.1:8080/v1",
            "http://localhost:1234/v1",
            "http://LOCALHOST:1234/v1",
            "http://[::1]:8080/v1",
        ] {
            let parsed = url::Url::parse(endpoint).unwrap();
            assert!(is_loopback_host(parsed.host_str()), "{endpoint}");
        }
        for endpoint in [
            "https://api.example.com/v1",
            "http://127.0.0.1.evil.example/v1",
            "http://192.168.1.10:8080/v1",
        ] {
            let parsed = url::Url::parse(endpoint).unwrap();
            assert!(!is_loopback_host(parsed.host_str()), "{endpoint}");
        }
    }

    #[test]
    fn enable_thinking_is_only_in_the_body_for_loopback_endpoints() {
        let config = ModelConfig {
            thinking: true,
            supports_thinking_toggle: true,
            model: "m".to_string(),
            ..ModelConfig::default()
        };
        let messages = [ChatMessage::user("hola")];

        let loopback = request_body(&config, &messages, true);
        assert_eq!(loopback["chat_template_kwargs"], json!({ "enable_thinking": true }));
        let unsupported = ModelConfig { supports_thinking_toggle: false, ..config.clone() };
        assert!(request_body(&unsupported, &messages, true).get("chat_template_kwargs").is_none());

        let remote = request_body(&config, &messages, false);
        assert!(remote.get("chat_template_kwargs").is_none(), "{remote}");
        assert_eq!(remote["model"], "m");
        assert_eq!(remote["stream"], false);
        assert_eq!(remote["max_tokens"], 4096);
        assert_eq!(remote["messages"][0], json!({ "role": "user", "content": "hola" }));
    }

    #[test]
    fn input_budget_reserves_output_tokens_and_saturates() {
        let config = ModelConfig { context_limit_tokens: 1000, max_output_tokens: 300, ..ModelConfig::default() };
        assert_eq!(config.input_budget_tokens(), 700);
        let tight = ModelConfig { context_limit_tokens: 100, max_output_tokens: 300, ..ModelConfig::default() };
        assert_eq!(tight.input_budget_tokens(), 0);
    }

    #[test]
    fn default_limits_match_app_settings() {
        let defaults = ModelConfig::default();
        let app = mt_core::AppSettings::default();
        assert_eq!(defaults.context_limit_tokens, app.context_limit_tokens as usize);
        assert_eq!(defaults.max_output_tokens, app.max_output_tokens as usize);
    }

    #[test]
    fn temperature_defaults_to_0_2_and_sampling_overrides_default_to_none() {
        let defaults = ModelConfig::default();
        assert_eq!(defaults.temperature, 0.2);
        assert_eq!(defaults.top_p, None);
        assert_eq!(defaults.top_k, None);
        assert_eq!(defaults.repetition_penalty, None);
        assert_eq!(defaults.presence_penalty, None);
    }

    #[test]
    fn debug_output_includes_sampling_overrides() {
        let config = ModelConfig {
            top_p: Some(0.9),
            top_k: Some(40),
            repetition_penalty: Some(1.1),
            presence_penalty: Some(0.5),
            ..ModelConfig::default()
        };
        let shown = format!("{config:?}");
        assert!(shown.contains("top_p: Some(0.9)"), "{shown}");
        assert!(shown.contains("top_k: Some(40)"), "{shown}");
        assert!(shown.contains("repetition_penalty: Some(1.1)"), "{shown}");
        assert!(shown.contains("presence_penalty: Some(0.5)"), "{shown}");
    }

    #[test]
    fn top_p_is_sent_on_remote_and_loopback_bodies_only_when_set() {
        let messages = [ChatMessage::user("hola")];
        let tool_messages = [json!({ "role": "user", "content": "hola" })];
        let tools = [save_report_tool()];
        let choice = json!("auto");
        let set = ModelConfig { top_p: Some(0.95), model: "m".to_string(), ..ModelConfig::default() };
        let unset = ModelConfig { model: "m".to_string(), ..ModelConfig::default() };

        for loopback in [true, false] {
            let body = request_body(&set, &messages, loopback);
            let top_p = body["top_p"].as_f64().expect("top_p present");
            assert!((top_p - 0.95).abs() < 1e-6, "{body}");
            assert!(request_body(&unset, &messages, loopback).get("top_p").is_none());

            let body = tools_request_body(&set, &tool_messages, &tools, choice.clone(), loopback);
            let top_p = body["top_p"].as_f64().expect("top_p present");
            assert!((top_p - 0.95).abs() < 1e-6, "{body}");
            assert!(tools_request_body(&unset, &tool_messages, &tools, choice.clone(), loopback).get("top_p").is_none());
        }
    }

    #[test]
    fn top_k_repetition_and_presence_penalties_are_sent_only_to_loopback() {
        let messages = [ChatMessage::user("hola")];
        let tool_messages = [json!({ "role": "user", "content": "hola" })];
        let tools = [save_report_tool()];
        let choice = json!("auto");
        let config = ModelConfig {
            top_k: Some(40),
            repetition_penalty: Some(1.1),
            presence_penalty: Some(0.5),
            supports_thinking_toggle: true,
            model: "m".to_string(),
            ..ModelConfig::default()
        };

        let local = request_body(&config, &messages, true);
        assert_eq!(local["top_k"], 40, "{local}");
        assert!((local["repetition_penalty"].as_f64().unwrap() - 1.1).abs() < 1e-6, "{local}");
        assert!((local["presence_penalty"].as_f64().unwrap() - 0.5).abs() < 1e-6, "{local}");
        assert_eq!(local["chat_template_kwargs"], json!({ "enable_thinking": false }));

        let remote = request_body(&config, &messages, false);
        for key in ["top_k", "repetition_penalty", "presence_penalty", "chat_template_kwargs"] {
            assert!(remote.get(key).is_none(), "{key} leaked to remote: {remote}");
        }

        let local = tools_request_body(&config, &tool_messages, &tools, choice.clone(), true);
        assert_eq!(local["top_k"], 40, "{local}");
        assert!((local["repetition_penalty"].as_f64().unwrap() - 1.1).abs() < 1e-6, "{local}");
        assert!((local["presence_penalty"].as_f64().unwrap() - 0.5).abs() < 1e-6, "{local}");

        let remote = tools_request_body(&config, &tool_messages, &tools, choice, false);
        for key in ["top_k", "repetition_penalty", "presence_penalty"] {
            assert!(remote.get(key).is_none(), "{key} leaked to remote: {remote}");
        }
    }

    #[test]
    fn debug_output_never_shows_the_api_key() {
        let config = ModelConfig { api_key: "sk-secret".to_string(), ..ModelConfig::default() };
        let shown = format!("{config:?}");
        assert!(!shown.contains("sk-secret"), "{shown}");
        assert!(shown.contains("<redacted>"), "{shown}");
    }

    #[tokio::test]
    async fn loopback_request_carries_enable_thinking_and_bearer_auth() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .and(header("authorization", "Bearer secret"))
            .and(body_partial_json(json!({
                "chat_template_kwargs": { "enable_thinking": true },
                "messages": [{ "role": "user", "content": "hola" }]
            })))
            .respond_with(completion("respuesta"))
            .expect(1)
            .mount(&server)
            .await;

        let provider = OpenAiCompatibleProvider::new(ModelConfig {
            api_key: "secret".to_string(),
            thinking: true,
            supports_thinking_toggle: true,
            ..config_for(&server)
        })
        .unwrap();
        let reply = provider.chat_complete(&[ChatMessage::user("hola")]).await.unwrap();
        assert_eq!(reply, "respuesta");
    }

    #[tokio::test]
    async fn context_overflow_is_reported_before_any_network_call() {
        let server = MockServer::start().await;
        Mock::given(method("POST")).respond_with(completion("nunca")).expect(0).mount(&server).await;

        let provider = OpenAiCompatibleProvider::new(ModelConfig {
            context_limit_tokens: 100,
            max_output_tokens: 50,
            ..config_for(&server)
        })
        .unwrap();
        let big = "x".repeat(600);
        let err = provider.chat_complete(&[ChatMessage::user(big)]).await.unwrap_err();
        assert!(
            matches!(err, AiError::ContextOverflow { budget_tokens: 50, .. }),
            "{err:?}"
        );
    }

    #[tokio::test]
    async fn error_status_becomes_request_failed() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(500).set_body_string("boom"))
            .mount(&server)
            .await;
        let provider = OpenAiCompatibleProvider::new(config_for(&server)).unwrap();
        let err = provider.chat_complete(&[ChatMessage::user("hola")]).await.unwrap_err();
        assert!(matches!(&err, AiError::RequestFailed(msg) if msg.contains("500")), "{err:?}");
    }

    #[tokio::test]
    async fn reply_without_content_is_invalid_response() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({ "choices": [] })))
            .mount(&server)
            .await;
        let provider = OpenAiCompatibleProvider::new(config_for(&server)).unwrap();
        let err = provider.chat_complete(&[ChatMessage::user("hola")]).await.unwrap_err();
        assert!(matches!(err, AiError::InvalidResponse(_)), "{err:?}");
    }

    #[tokio::test]
    async fn refused_connection_is_endpoint_unreachable() {
        // Bind then drop a listener to obtain a loopback port with nothing listening on it.
        let port = std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
        let provider = OpenAiCompatibleProvider::new(ModelConfig {
            endpoint: format!("http://127.0.0.1:{port}/v1"),
            ..ModelConfig::default()
        })
        .unwrap();
        let err = provider.chat_complete(&[ChatMessage::user("hola")]).await.unwrap_err();
        assert!(matches!(err, AiError::EndpointUnreachable(_)), "{err:?}");
    }

    fn save_report_tool() -> serde_json::Value {
        json!({"type": "function", "function": {"name": "save_meeting_report", "parameters": {"type": "object"}}})
    }

    #[test]
    fn tool_request_body_sends_tools_and_enable_thinking_only_for_loopback() {
        let config = ModelConfig {
            thinking: true,
            supports_thinking_toggle: true,
            model: "m".to_string(),
            ..ModelConfig::default()
        };
        let messages = [json!({ "role": "user", "content": "hola" })];
        let tools = [save_report_tool()];
        let choice = json!({ "type": "function", "function": { "name": "save_meeting_report" } });

        let loopback = tools_request_body(&config, &messages, &tools, choice.clone(), true);
        assert_eq!(loopback["chat_template_kwargs"], json!({ "enable_thinking": true }));
        assert_eq!(loopback["tool_choice"], choice);
        assert_eq!(loopback["tools"], json!(tools));

        let remote = tools_request_body(&config, &messages, &tools, choice, false);
        assert!(remote.get("chat_template_kwargs").is_none(), "{remote}");
        assert_eq!(remote["stream"], false);
        assert_eq!(remote["max_tokens"], 4096);
    }

    #[tokio::test]
    async fn tool_request_sends_tools_tool_choice_and_loopback_thinking_and_returns_the_message() {
        let server = MockServer::start().await;
        let tool_choice = json!({ "type": "function", "function": { "name": "save_meeting_report" } });
        let message = json!({
            "role": "assistant",
            "content": null,
            "tool_calls": [{ "id": "call_1", "type": "function", "function": { "name": "save_meeting_report", "arguments": "{}" } }]
        });
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .and(body_partial_json(json!({
                "model": "test-model",
                "tools": [save_report_tool()],
                "tool_choice": tool_choice,
                "chat_template_kwargs": { "enable_thinking": true },
                "messages": [{ "role": "user", "content": "hola" }]
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({ "choices": [{ "message": message }] })))
            .expect(1)
            .mount(&server)
            .await;

        let provider = OpenAiCompatibleProvider::new(ModelConfig {
            thinking: true,
            supports_thinking_toggle: true,
            ..config_for(&server)
        })
        .unwrap();
        let reply = provider
            .chat_with_tools(
                &[json!({ "role": "user", "content": "hola" })],
                &[save_report_tool()],
                tool_choice.clone(),
            )
            .await
            .unwrap();
        assert_eq!(reply.message, message);
    }

    #[tokio::test]
    async fn tool_request_over_budget_fails_before_any_network_call() {
        let server = MockServer::start().await;
        Mock::given(method("POST")).respond_with(completion("nunca")).expect(0).mount(&server).await;
        let provider = OpenAiCompatibleProvider::new(ModelConfig {
            context_limit_tokens: 100,
            max_output_tokens: 50,
            ..config_for(&server)
        })
        .unwrap();
        let big = json!({ "role": "user", "content": "x".repeat(600) });
        let err = provider
            .chat_with_tools(&[big], &[save_report_tool()], json!("auto"))
            .await
            .unwrap_err();
        assert!(
            matches!(err, AiError::ContextOverflow { budget_tokens: 50, .. }),
            "{err:?}"
        );
    }

    #[tokio::test]
    async fn tool_schema_counts_against_the_input_budget_before_any_network_call() {
        let server = MockServer::start().await;
        Mock::given(method("POST")).respond_with(completion("nunca")).expect(0).mount(&server).await;
        let provider = OpenAiCompatibleProvider::new(ModelConfig {
            context_limit_tokens: 100,
            max_output_tokens: 50,
            ..config_for(&server)
        })
        .unwrap();
        let huge_tool = json!({
            "type": "function",
            "function": { "name": "save_meeting_report", "description": "x".repeat(600), "parameters": { "type": "object" } }
        });
        let err = provider
            .chat_with_tools(&[json!({ "role": "user", "content": "hola" })], &[huge_tool], json!("auto"))
            .await
            .unwrap_err();
        assert!(
            matches!(err, AiError::ContextOverflow { budget_tokens: 50, estimated_tokens } if estimated_tokens > 50),
            "{err:?}"
        );
    }

    #[tokio::test]
    async fn tool_choice_counts_against_the_input_budget_before_any_network_call() {
        let server = MockServer::start().await;
        Mock::given(method("POST")).respond_with(completion("nunca")).expect(0).mount(&server).await;
        let provider = OpenAiCompatibleProvider::new(ModelConfig {
            context_limit_tokens: 100,
            max_output_tokens: 50,
            ..config_for(&server)
        })
        .unwrap();
        let huge_choice = json!({ "type": "function", "function": { "name": "x".repeat(600) } });
        let err = provider
            .chat_with_tools(&[json!({ "role": "user", "content": "hola" })], &[], huge_choice)
            .await
            .unwrap_err();
        assert!(matches!(err, AiError::ContextOverflow { budget_tokens: 50, .. }), "{err:?}");
    }

    #[tokio::test]
    async fn tool_reply_without_choices_message_is_invalid_response() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({ "choices": [] })))
            .mount(&server)
            .await;
        let provider = OpenAiCompatibleProvider::new(config_for(&server)).unwrap();
        let err = provider
            .chat_with_tools(&[json!({ "role": "user", "content": "hola" })], &[], json!("auto"))
            .await
            .unwrap_err();
        assert!(matches!(err, AiError::InvalidResponse(_)), "{err:?}");
    }
}
