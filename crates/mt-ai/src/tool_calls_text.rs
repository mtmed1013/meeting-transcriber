//! Tool-call recovery from plain-text model replies.
//!
//! Some servers (`mlx_lm.server` among them) drop the `tools` payload when a model's chat template
//! declares no tool-call markers, so the model answers in text using the call syntax of its family.
//! This module recovers those calls. It is pure: no I/O and no execution. Only tool names listed in
//! `allowed` survive, so a call to an unknown tool is never dispatched.

use serde_json::{Map, Value, json};

/// One tool call recovered from text.
#[derive(Debug, Clone, PartialEq)]
pub struct ParsedCall {
    pub name: String,
    pub arguments: serde_json::Value,
}

/// Allowed calls found in a reply, plus the reply text with the call markup removed.
#[derive(Debug, Clone, PartialEq)]
pub struct ParsedCalls {
    pub calls: Vec<ParsedCall>,
    pub cleaned_text: String,
}

/// Recovers the tool calls in `content` whose names are listed in `allowed`.
///
/// Detectors run in a fixed order and the first one that recognises call markup wins. Explicit
/// markup (fenced `tool_call`, Hermes, LFM2, Mistral, python tag) is removed from the reply even when
/// its tool name is unknown, and the unknown calls are dropped. Unmarked calls (pythonic list, JSON
/// array, bare JSON object) are only recognised when they name an allowed tool, so quoted examples
/// and invented syntax stay as text.
pub fn parse_tool_calls(content: &str, allowed: &[&str]) -> ParsedCalls {
    let masked = mask_code_fences(content);
    for detect in DETECTORS {
        if let Some(found) = detect(content, &masked, allowed) {
            return ParsedCalls {
                calls: found.calls,
                cleaned_text: remove_spans(content, &found.spans).trim().to_string(),
            };
        }
    }
    ParsedCalls {
        calls: Vec::new(),
        cleaned_text: content.trim().to_string(),
    }
}

/// Byte ranges of recognised call markup, removed from the visible reply.
type Spans = Vec<(usize, usize)>;

/// Calls and markup spans recovered by one detector.
#[derive(Default)]
struct Found {
    calls: Vec<ParsedCall>,
    spans: Spans,
}

/// A detector gets the raw reply and a copy with code fences blanked out. The copy has the same
/// byte offsets, so spans found in it can be removed from the raw reply.
type Detector = fn(raw: &str, masked: &str, allowed: &[&str]) -> Option<Found>;

const DETECTORS: [Detector; 8] = [
    detect_fenced_tool_call,
    detect_hermes,
    detect_lfm2,
    detect_mistral,
    detect_python_tag,
    detect_pythonic_list,
    detect_json_array,
    detect_bare_json,
];

fn detect_fenced_tool_call(raw: &str, _masked: &str, allowed: &[&str]) -> Option<Found> {
    let mut found = Found::default();
    for fence in code_fences(raw)
        .into_iter()
        .filter(|fence| fence.closed && fence.info == "tool_call")
    {
        if let Some(calls) = calls_from_body(fence.body) {
            found.calls.extend(calls);
            found.spans.push((fence.start, fence.end));
        }
    }
    explicit(found, allowed)
}

fn detect_hermes(raw: &str, _masked: &str, allowed: &[&str]) -> Option<Found> {
    let found = tagged_blocks(raw, "<tool_call>", "</tool_call>", calls_from_body);
    explicit(found, allowed)
}

fn detect_lfm2(raw: &str, _masked: &str, allowed: &[&str]) -> Option<Found> {
    let found = tagged_blocks(raw, "<|tool_call_start|>", "<|tool_call_end|>", lfm2_body);
    explicit(found, allowed)
}

fn detect_mistral(raw: &str, _masked: &str, allowed: &[&str]) -> Option<Found> {
    let found = marked_json(raw, "[TOOL_CALLS]");
    explicit(found, allowed)
}

fn detect_python_tag(raw: &str, _masked: &str, allowed: &[&str]) -> Option<Found> {
    let found = marked_json(raw, "<|python_tag|>");
    explicit(found, allowed)
}

fn detect_pythonic_list(_raw: &str, masked: &str, allowed: &[&str]) -> Option<Found> {
    let mut found = Found::default();
    let mut at = 0;
    while let Some(rel) = masked[at..].find('[') {
        let start = at + rel;
        let parsed = closing_bracket(masked, start)
            .and_then(|end| pythonic_list(&masked[start..=end]).map(|calls| (end, calls)));
        match parsed {
            Some((end, calls)) => {
                keep_allowed(&mut found, calls, (start, end + 1), allowed);
                at = end + 1;
            }
            None => at = start + 1,
        }
    }
    implicit(found)
}

fn detect_json_array(_raw: &str, masked: &str, allowed: &[&str]) -> Option<Found> {
    let mut found = Found::default();
    for (start, end, value) in json_values_in(masked) {
        let Value::Array(items) = &value else {
            continue;
        };
        if let Some(calls) = calls_from_array(items) {
            keep_allowed(&mut found, calls, (start, end), allowed);
        }
    }
    implicit(found)
}

fn detect_bare_json(_raw: &str, masked: &str, allowed: &[&str]) -> Option<Found> {
    let mut found = Found::default();
    for (start, end, value) in json_values_in(masked) {
        if let Some(call) = call_from_object(&value) {
            keep_allowed(&mut found, vec![call], (start, end), allowed);
        }
    }
    implicit(found)
}

/// Explicit markup is recognised whatever the tool name. Unknown calls are dropped afterwards.
fn explicit(mut found: Found, allowed: &[&str]) -> Option<Found> {
    if found.spans.is_empty() {
        return None;
    }
    found
        .calls
        .retain(|call| allowed.contains(&call.name.as_str()));
    Some(found)
}

/// Unmarked calls are recognised only when they name an allowed tool.
fn implicit(found: Found) -> Option<Found> {
    (!found.spans.is_empty()).then_some(found)
}

fn keep_allowed(found: &mut Found, calls: Vec<ParsedCall>, span: (usize, usize), allowed: &[&str]) {
    let kept: Vec<ParsedCall> = calls
        .into_iter()
        .filter(|call| allowed.contains(&call.name.as_str()))
        .collect();
    if !kept.is_empty() {
        found.calls.extend(kept);
        found.spans.push(span);
    }
}

/// Every `open ... close` block in `raw`. `parse` decides whether a body holds calls.
fn tagged_blocks(
    raw: &str,
    open: &str,
    close: &str,
    parse: fn(&str) -> Option<Vec<ParsedCall>>,
) -> Found {
    let mut found = Found::default();
    let mut at = 0;
    while let Some(rel) = raw[at..].find(open) {
        let start = at + rel;
        let body_start = start + open.len();
        let Some(close_rel) = raw[body_start..].find(close) else {
            break;
        };
        let body_end = body_start + close_rel;
        let end = body_end + close.len();
        if let Some(calls) = parse(raw[body_start..body_end].trim()) {
            found.calls.extend(calls);
            found.spans.push((start, end));
        }
        at = end;
    }
    found
}

/// LFM2 bodies hold a pythonic list or JSON.
fn lfm2_body(body: &str) -> Option<Vec<ParsedCall>> {
    pythonic_list(body).or_else(|| calls_from_body(body))
}

/// A marker followed by a JSON array or object, as in `[TOOL_CALLS] [...]`.
fn marked_json(raw: &str, marker: &str) -> Found {
    let mut found = Found::default();
    let mut at = 0;
    while let Some(rel) = raw[at..].find(marker) {
        let start = at + rel;
        let after = start + marker.len();
        let payload = raw[after..].trim_start();
        let payload_start = raw.len() - payload.len();
        let parsed = json_prefix(payload)
            .and_then(|(len, value)| calls_from_value(&value).map(|calls| (len, calls)));
        match parsed {
            Some((len, calls)) => {
                found.calls.extend(calls);
                found.spans.push((start, payload_start + len));
                at = payload_start + len;
            }
            None => at = after,
        }
    }
    found
}

/// A call object, or an array of call objects, as the whole of a JSON body.
fn calls_from_body(body: &str) -> Option<Vec<ParsedCall>> {
    let mut calls = Vec::new();
    for item in serde_json::Deserializer::from_str(body.trim()).into_iter::<Value>() {
        calls.extend(calls_from_value(&item.ok()?)?);
    }
    (!calls.is_empty()).then_some(calls)
}

fn calls_from_value(value: &Value) -> Option<Vec<ParsedCall>> {
    match value {
        Value::Object(_) => call_from_object(value).map(|call| vec![call]),
        Value::Array(items) => calls_from_array(items),
        _ => None,
    }
}

/// An array counts as calls only when every item is a call object.
fn calls_from_array(items: &[Value]) -> Option<Vec<ParsedCall>> {
    if items.is_empty() {
        return None;
    }
    items.iter().map(call_from_object).collect()
}

/// `{"name": ..., "arguments"|"parameters": ...}`. Arguments may be an object or a JSON string
/// that holds an object. Anything else is malformed.
fn call_from_object(value: &Value) -> Option<ParsedCall> {
    let object = value.as_object()?;
    let name = object.get("name")?.as_str()?.trim();
    if name.is_empty() {
        return None;
    }
    let raw_arguments = object
        .get("arguments")
        .or_else(|| object.get("parameters"))?;
    let arguments = match raw_arguments {
        Value::Object(_) => raw_arguments.clone(),
        Value::String(text) => match serde_json::from_str::<Value>(text).ok()? {
            object @ Value::Object(_) => object,
            _ => return None,
        },
        _ => return None,
    };
    Some(ParsedCall {
        name: name.to_string(),
        arguments,
    })
}

/// Parses one JSON value at the start of `text`, returning its byte length.
fn json_prefix(text: &str) -> Option<(usize, Value)> {
    let mut stream = serde_json::Deserializer::from_str(text).into_iter::<Value>();
    let value = stream.next()?.ok()?;
    Some((stream.byte_offset(), value))
}

/// Every JSON object or array in `text`, as `(start, end, value)` byte offsets.
fn json_values_in(text: &str) -> Vec<(usize, usize, Value)> {
    let mut found = Vec::new();
    let mut at = 0;
    while let Some(rel) = text[at..].find(['{', '[']) {
        let start = at + rel;
        match json_prefix(&text[start..]) {
            Some((len, value)) => {
                found.push((start, start + len, value));
                at = start + len;
            }
            None => at = start + 1,
        }
    }
    found
}

/// One Markdown code fence. An unclosed fence runs to the end of the reply.
struct Fence<'a> {
    start: usize,
    end: usize,
    info: &'a str,
    body: &'a str,
    closed: bool,
}

/// Pairs ``` markers in order. The info string is the rest of the opening line.
fn code_fences(text: &str) -> Vec<Fence<'_>> {
    let mut fences = Vec::new();
    let mut at = 0;
    while let Some(rel) = text[at..].find("```") {
        let start = at + rel;
        let info_start = start + 3;
        let line_end = text[info_start..]
            .find('\n')
            .map_or(text.len(), |i| info_start + i);
        let info = text[info_start..line_end].trim();
        let body_start = (line_end + 1).min(text.len());
        let (body_end, end, closed) = match text[body_start..].find("```") {
            Some(close_rel) => (body_start + close_rel, body_start + close_rel + 3, true),
            None => (text.len(), text.len(), false),
        };
        fences.push(Fence {
            start,
            end,
            info,
            body: &text[body_start..body_end],
            closed,
        });
        at = end;
    }
    fences
}

/// Copy of `content` with every code fence blanked to spaces, keeping byte offsets.
fn mask_code_fences(content: &str) -> String {
    let mut masked = content.to_string();
    for fence in code_fences(content) {
        masked.replace_range(fence.start..fence.end, &" ".repeat(fence.end - fence.start));
    }
    masked
}

fn remove_spans(text: &str, spans: &[(usize, usize)]) -> String {
    let mut sorted = spans.to_vec();
    sorted.sort_unstable();
    let mut out = String::with_capacity(text.len());
    let mut at = 0;
    for (start, end) in sorted {
        if start > at {
            out.push_str(&text[at..start]);
        }
        at = at.max(end);
    }
    if at < text.len() {
        out.push_str(&text[at..]);
    }
    out
}

/// Tracks whether a byte is code or inside a string literal.
#[derive(Default)]
struct Quotes {
    open: Option<u8>,
    escaped: bool,
}

impl Quotes {
    /// Returns true when `byte` is code, outside any string literal.
    fn step(&mut self, byte: u8) -> bool {
        if let Some(quote) = self.open {
            match byte {
                _ if self.escaped => self.escaped = false,
                b'\\' => self.escaped = true,
                _ if byte == quote => self.open = None,
                _ => {}
            }
            return false;
        }
        if byte == b'"' || byte == b'\'' {
            self.open = Some(byte);
            return false;
        }
        true
    }
}

/// Index of the bracket that closes the one at `open_at`, skipping string literals.
fn closing_bracket(text: &str, open_at: usize) -> Option<usize> {
    let mut depth = 0i32;
    let mut quotes = Quotes::default();
    for (index, &byte) in text.as_bytes().iter().enumerate().skip(open_at) {
        if !quotes.step(byte) {
            continue;
        }
        match byte {
            b'(' | b'[' | b'{' => depth += 1,
            b')' | b']' | b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(index);
                }
            }
            _ => {}
        }
    }
    None
}

/// Splits on commas that sit outside brackets and string literals.
fn split_top_level(text: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut depth = 0i32;
    let mut quotes = Quotes::default();
    let mut start = 0;
    for (index, &byte) in text.as_bytes().iter().enumerate() {
        if !quotes.step(byte) {
            continue;
        }
        match byte {
            b'(' | b'[' | b'{' => depth += 1,
            b')' | b']' | b'}' => depth -= 1,
            b',' if depth == 0 => {
                parts.push(&text[start..index]);
                start = index + 1;
            }
            _ => {}
        }
    }
    parts.push(&text[start..]);
    parts
}

/// `[name(key="value", n=1), other()]`. Every entry must be a call, or the whole list is rejected.
fn pythonic_list(text: &str) -> Option<Vec<ParsedCall>> {
    let inner = text.trim().strip_prefix('[')?.strip_suffix(']')?;
    let mut calls = Vec::new();
    for entry in split_top_level(inner) {
        let entry = entry.trim();
        if entry.is_empty() {
            continue;
        }
        let open = entry.find('(')?;
        let name = entry[..open].trim();
        if !is_identifier(name) {
            return None;
        }
        let args = entry[open + 1..].strip_suffix(')')?;
        calls.push(ParsedCall {
            name: name.to_string(),
            arguments: pythonic_kwargs(args)?,
        });
    }
    (!calls.is_empty()).then_some(calls)
}

/// `key="value", n=1`. Positional arguments are rejected.
fn pythonic_kwargs(args: &str) -> Option<Value> {
    let mut map = Map::new();
    for part in split_top_level(args) {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let (key, raw) = part.split_once('=')?;
        let key = key.trim();
        if !is_identifier(key) {
            return None;
        }
        map.insert(key.to_string(), python_scalar(raw.trim()));
    }
    Some(Value::Object(map))
}

fn is_identifier(name: &str) -> bool {
    !name.is_empty() && name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_')
}

fn python_scalar(raw: &str) -> Value {
    let quoted = raw.len() >= 2
        && ((raw.starts_with('"') && raw.ends_with('"'))
            || (raw.starts_with('\'') && raw.ends_with('\'')));
    match raw {
        "True" | "true" => Value::Bool(true),
        "False" | "false" => Value::Bool(false),
        "None" | "null" => Value::Null,
        _ if quoted => Value::String(unescape(&raw[1..raw.len() - 1])),
        _ => {
            if let Ok(n) = raw.parse::<i64>() {
                json!(n)
            } else if let Some(f) = raw.parse::<f64>().ok().filter(|f| f.is_finite()) {
                json!(f)
            } else if raw.starts_with('[') || raw.starts_with('{') {
                serde_json::from_str(raw).unwrap_or_else(|_| Value::String(raw.to_string()))
            } else {
                Value::String(raw.to_string())
            }
        }
    }
}

/// Resolves the escapes a Python string literal can carry. Unknown escapes keep their backslash.
fn unescape(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    let mut escaped = false;
    for ch in raw.chars() {
        if !escaped {
            if ch == '\\' {
                escaped = true;
            } else {
                out.push(ch);
            }
            continue;
        }
        escaped = false;
        match ch {
            'n' => out.push('\n'),
            't' => out.push('\t'),
            'r' => out.push('\r'),
            '\\' | '"' | '\'' => out.push(ch),
            _ => {
                out.push('\\');
                out.push(ch);
            }
        }
    }
    if escaped {
        out.push('\\');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOOLS: &[&str] = &["read_file", "list_directory", "create_folder", "open_app"];

    fn parse(content: &str) -> ParsedCalls {
        parse_tool_calls(content, TOOLS)
    }

    fn call(name: &str, arguments: serde_json::Value) -> ParsedCall {
        ParsedCall {
            name: name.to_string(),
            arguments,
        }
    }

    #[test]
    fn fenced_tool_call_block_is_parsed() {
        let out = parse(
            "Voy a leerlo.\n```tool_call\n{\"name\": \"read_file\", \"arguments\": {\"path\": \"a.md\"}}\n```",
        );
        assert_eq!(out.calls, vec![call("read_file", json!({"path": "a.md"}))]);
        assert_eq!(out.cleaned_text, "Voy a leerlo.");
    }

    #[test]
    fn hermes_tool_call_block_is_parsed() {
        let out = parse(
            "Listo.\n<tool_call>{\"name\":\"read_file\",\"arguments\":{\"path\":\"a.md\"}}</tool_call>",
        );
        assert_eq!(out.calls, vec![call("read_file", json!({"path": "a.md"}))]);
        assert_eq!(out.cleaned_text, "Listo.");
    }

    #[test]
    fn lfm2_pythonic_call_is_parsed_with_typed_arguments() {
        let out = parse(
            "<|tool_call_start|>[read_file(path=\"a.md\", depth=2, dry=True)]<|tool_call_end|>",
        );
        assert_eq!(
            out.calls,
            vec![call(
                "read_file",
                json!({"path": "a.md", "depth": 2, "dry": true})
            )]
        );
        assert!(out.cleaned_text.is_empty());
    }

    #[test]
    fn lfm2_json_call_is_parsed() {
        let out = parse(
            "<|tool_call_start|>[{\"name\":\"list_directory\",\"arguments\":{\"path\":\"~/\"}}]<|tool_call_end|>",
        );
        assert_eq!(
            out.calls,
            vec![call("list_directory", json!({"path": "~/"}))]
        );
        assert!(out.cleaned_text.is_empty());
    }

    #[test]
    fn mistral_marker_is_parsed() {
        let out =
            parse("[TOOL_CALLS] [{\"name\":\"open_app\",\"arguments\":{\"name\":\"Safari\"}}]");
        assert_eq!(out.calls, vec![call("open_app", json!({"name": "Safari"}))]);
        assert!(out.cleaned_text.is_empty());
    }

    #[test]
    fn python_tag_payload_is_parsed() {
        let out =
            parse("<|python_tag|>{\"name\":\"read_file\",\"parameters\":{\"path\":\"a.md\"}}");
        assert_eq!(out.calls, vec![call("read_file", json!({"path": "a.md"}))]);
        assert!(out.cleaned_text.is_empty());
    }

    #[test]
    fn pythonic_list_is_parsed() {
        let out = parse("[read_file(path=\"a.md\"), list_directory(path=\"~/\", depth=1)]");
        assert_eq!(
            out.calls,
            vec![
                call("read_file", json!({"path": "a.md"})),
                call("list_directory", json!({"path": "~/", "depth": 1})),
            ]
        );
        assert!(out.cleaned_text.is_empty());
    }

    #[test]
    fn json_array_of_calls_is_parsed() {
        let out = parse(
            "[{\"name\":\"create_folder\",\"arguments\":{\"path\":\"~/Proyectos\"}},{\"name\":\"list_directory\",\"arguments\":{\"path\":\"~/\"}}]",
        );
        assert_eq!(out.calls.len(), 2);
        assert_eq!(out.calls[0].name, "create_folder");
        assert_eq!(out.calls[1].name, "list_directory");
        assert!(out.cleaned_text.is_empty());
    }

    #[test]
    fn bare_json_object_is_parsed_and_prose_is_kept() {
        let out =
            parse("Te paso el archivo: {\"name\":\"read_file\",\"arguments\":{\"path\":\"a.md\"}}");
        assert_eq!(out.calls, vec![call("read_file", json!({"path": "a.md"}))]);
        assert_eq!(out.cleaned_text, "Te paso el archivo:");
    }

    #[test]
    fn unknown_tool_name_is_dropped_in_every_family() {
        let hermes =
            parse("<tool_call>{\"name\":\"rm_rf\",\"arguments\":{\"path\":\"/\"}}</tool_call>");
        assert!(hermes.calls.is_empty());
        assert!(hermes.cleaned_text.is_empty());

        let pythonic = parse("[rm_rf(path=\"/\")]");
        assert!(pythonic.calls.is_empty());

        let bare = parse("{\"name\":\"rm_rf\",\"arguments\":{}}");
        assert!(bare.calls.is_empty());
        assert_eq!(bare.cleaned_text, "{\"name\":\"rm_rf\",\"arguments\":{}}");
    }

    #[test]
    fn unknown_call_is_dropped_while_known_call_in_same_block_survives() {
        let out = parse(
            "<tool_call>{\"name\":\"rm_rf\",\"arguments\":{}}</tool_call>\n<tool_call>{\"name\":\"read_file\",\"arguments\":{\"path\":\"a.md\"}}</tool_call>",
        );
        assert_eq!(out.calls, vec![call("read_file", json!({"path": "a.md"}))]);
    }

    #[test]
    fn json_example_in_a_json_fence_is_ignored() {
        let text =
            "Ejemplo:\n```json\n{\"name\":\"read_file\",\"arguments\":{\"path\":\"x\"}}\n```";
        let out = parse(text);
        assert!(out.calls.is_empty());
        assert_eq!(out.cleaned_text, text);
    }

    #[test]
    fn pythonic_example_in_a_plain_fence_is_ignored() {
        let text = "```\n[read_file(path=\"a.md\")]\n```";
        let out = parse(text);
        assert!(out.calls.is_empty());
        assert_eq!(out.cleaned_text, text);
    }

    #[test]
    fn malformed_json_is_ignored() {
        let hermes = "<tool_call>{\"name\":\"read_file\",\"arguments\":{\"path\":</tool_call>";
        let out = parse(hermes);
        assert!(out.calls.is_empty());
        assert_eq!(out.cleaned_text, hermes);

        let fenced = "```tool_call\n{\"name\":\"read_file\",\"arguments\":{\"path\"\n```";
        let out = parse(fenced);
        assert!(out.calls.is_empty());

        let bare = "{\"name\":\"read_file\",\"arguments\":{";
        let out = parse(bare);
        assert!(out.calls.is_empty());
        assert_eq!(out.cleaned_text, bare);
    }

    #[test]
    fn text_without_calls_returns_no_calls_and_keeps_the_text() {
        let out = parse("  Hola, ¿qué tal? Todo bien.  ");
        assert!(out.calls.is_empty());
        assert_eq!(out.cleaned_text, "Hola, ¿qué tal? Todo bien.");
    }

    #[test]
    fn two_calls_in_one_block_are_both_parsed() {
        let out = parse(
            "```tool_call\n{\"name\":\"read_file\",\"arguments\":{\"path\":\"a.md\"}}\n{\"name\":\"list_directory\",\"arguments\":{\"path\":\"~/\"}}\n```",
        );
        assert_eq!(
            out.calls,
            vec![
                call("read_file", json!({"path": "a.md"})),
                call("list_directory", json!({"path": "~/"})),
            ]
        );
        assert!(out.cleaned_text.is_empty());
    }

    #[test]
    fn arguments_as_json_string_and_parameters_alias_are_accepted() {
        let out = parse(
            "<tool_call>{\"name\":\"read_file\",\"parameters\":\"{\\\"path\\\":\\\"a.md\\\"}\"}</tool_call>",
        );
        assert_eq!(out.calls, vec![call("read_file", json!({"path": "a.md"}))]);
    }
}
