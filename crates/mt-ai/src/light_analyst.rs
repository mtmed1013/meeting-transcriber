//! Light, tool-driven meeting analysis.
//!
//! The transcript is pulled by the model with `read_transcript`, the skill arrives through
//! `read_meeting_context`, and the report is delivered by calling `save_meeting_report` with one
//! Markdown string per section. Small local models follow this far better than a large prompt
//! carrying the transcript and a deeply nested JSON schema, and servers that drop the `tools`
//! payload still work because plain-text tool calls are parsed (see `tool_calls_text`).

use mt_core::report::MarkdownSections;
use mt_core::{
    compact_transcript_for_analysis, count_spoken_words, default_report_skill_prompt,
    strip_existing_report,
};
use serde_json::{Value, json};

use crate::budget::{chunk_transcript, estimate_messages_tokens, max_bytes_for_tokens};
use crate::error::AiError;
use crate::provider::{ChatMessage, OpenAiCompatibleProvider};
use crate::tool_calls_text::parse_tool_calls;

/// Transcripts with fewer spoken words than this skip the model entirely.
pub const MIN_SPOKEN_WORDS: usize = 20;
/// Name of the tool that delivers the finished report.
pub const SAVE_REPORT_TOOL_NAME: &str = "save_meeting_report";
/// Tool that hands the skill and the chunk map to the model.
pub const READ_CONTEXT_TOOL_NAME: &str = "read_meeting_context";
/// Tool that pages through the transcript.
pub const READ_TRANSCRIPT_TOOL_NAME: &str = "read_transcript";

const MAX_ROUNDS: usize = 12;
const MAX_ROUND_CAP: usize = 40;
/// Maximum tokens per transcript chunk (~7.5 KB), keeping every chunk well below
/// `MAX_TOOL_RESULT_CHARS` so `read_transcript` never truncates a chunk silently.
const MAX_CHUNK_TOKENS: usize = 2_500;
/// Longest tool result handed back to the model.
const MAX_TOOL_RESULT_CHARS: usize = 12_000;
/// Longest single report section accepted from the model.
const MAX_FIELD_CHARS: usize = 12_000;
const ROUNDING_SLACK_TOKENS: usize = 8;
const NO_REPORT_MESSAGE: &str =
    "el modelo no llamó a save_meeting_report (revisa que soporte tool calling)";

/// The instructions that teach a small model how to deliver the report. English on purpose:
/// small models follow a short English prompt better, whatever the output language is.
pub fn analysis_system_prompt(output_language: &str) -> String {
    format!(
        "You are meeting-report-analyst. You read a meeting transcript and deliver the report by \
calling tools.\n\
Order: call read_meeting_context first; then call read_transcript for EVERY chunk 0, 1, … up to \
the last chunk; only after reading all chunks, call save_meeting_report once.\n\
Tools: read_meeting_context() returns the skill and the chunk count; \
read_transcript(chunk: integer) returns one chunk of the transcript; \
save_meeting_report(executive_summary, key_decisions, topics_and_discussion, action_items, \
key_questions, open_issues, important_facts) writes the report, each argument rich Markdown.\n\
Rules: use only what the transcript states; never invent owners, dates, decisions or numbers; \
write Not specified (in OUTPUT_LANGUAGE) when data is missing; write informative narrative and \
detailed workflow/process explanations rather than shallow one-line bullets; \
OUTPUT_LANGUAGE = {output_language} applies to the report content only, never to these instructions.\n\
Never say you will do something without calling the tool.\n\
Use your native tool-call format when the server supports it; otherwise reply ONLY with this \
block, then wait for the result:\n\
```tool_call\n\
{{\"name\": \"save_meeting_report\", \"arguments\": {{\"executive_summary\": \"...\"}}}}\n\
```"
    )
}

/// OpenAI function tool that saves the final meeting report.
fn save_report_tool() -> Value {
    let section = |description: &str| json!({ "type": "string", "description": description });
    json!({
        "type": "function",
        "function": {
            "name": SAVE_REPORT_TOOL_NAME,
            "description": "Saves the final meeting report. Every argument is rich Markdown text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "executive_summary": section("1-2 clear narrative paragraphs covering purpose, participants, main topics/workflows, decisions, and next steps."),
                    "key_decisions": section("Explicit decisions and agreements with their reason/context when discussed."),
                    "topics_and_discussion": section("Chronological topic chapters (### Title — HH:MM:SS) with substantive explanations, including step-by-step workflows/processes discussed."),
                    "action_items": section("Detailed checklist items (- [ ] ...) with owner, specific task, priority/status if stated, due date, and timestamp if available."),
                    "key_questions": section("Material questions with their answer (or Open/Partially resolved status) and why they matter."),
                    "open_issues": section("Unresolved problems, bugs, blockers, or risks with impact and next step when stated."),
                    "important_facts": section("Exact dates, deadlines, rules, laws, endpoints, environments, limits, numbers, and technical constraints.")
                },
                "required": [
                    "executive_summary",
                    "key_decisions",
                    "topics_and_discussion",
                    "action_items",
                    "key_questions",
                    "open_issues",
                    "important_facts"
                ],
                "additionalProperties": false
            }
        }
    })
}

/// OpenAI function tools for the report flow.
pub fn meeting_tools() -> Vec<Value> {
    vec![
        json!({
            "type": "function",
            "function": {
                "name": READ_CONTEXT_TOOL_NAME,
                "description": "Returns the report skill, the output language and the transcript chunk count.",
                "parameters": { "type": "object", "properties": {}, "required": [] }
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": READ_TRANSCRIPT_TOOL_NAME,
                "description": "Returns one chunk of the meeting transcript.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "chunk": { "type": "integer", "description": "Zero-based chunk index." }
                    },
                    "required": ["chunk"],
                    "additionalProperties": false
                }
            }
        }),
        save_report_tool(),
    ]
}

/// Everything the tools read while the model works.
struct ReportContext {
    skill: String,
    output_language: String,
    chunks: Vec<String>,
}

impl ReportContext {
    fn tool_names() -> Vec<&'static str> {
        vec![
            READ_CONTEXT_TOOL_NAME,
            READ_TRANSCRIPT_TOOL_NAME,
            SAVE_REPORT_TOOL_NAME,
        ]
    }

    /// Runs one tool call and returns the text handed back to the model.
    fn call(&self, name: &str, args: &Value) -> String {
        match name {
            READ_CONTEXT_TOOL_NAME => format!(
                "OUTPUT_LANGUAGE = {}\nChunks: {}\nRead them with read_transcript(chunk: 0..{}).\n\n\
SKILL:\n{}",
                self.output_language,
                self.chunks.len(),
                self.chunks.len().saturating_sub(1),
                self.skill
            ),
            READ_TRANSCRIPT_TOOL_NAME => {
                let Some(index) = args.get("chunk").and_then(Value::as_i64) else {
                    return "error: read_transcript needs an integer 'chunk'".to_string();
                };
                if index < 0 || index as usize >= self.chunks.len() {
                    return format!(
                        "error: chunk {index} does not exist; valid range is 0..{}",
                        self.chunks.len().saturating_sub(1)
                    );
                }
                let text = clip(&self.chunks[index as usize], MAX_TOOL_RESULT_CHARS);
                format!(
                    "chunk {index} of {}:\n{text}",
                    self.chunks.len().saturating_sub(1)
                )
            }
            other => format!("error: unknown tool '{other}'"),
        }
    }
}

fn clip(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        return text.to_string();
    }
    let head: String = text.chars().take(max).collect();
    format!("{head}\n[truncated]")
}

/// Removes a previously generated report so the model only sees the transcript.
pub fn clean_transcript(transcript: &str) -> String {
    strip_existing_report(transcript).trim().to_string()
}

/// Whether the cleaned transcript has enough spoken words to be analyzed.
pub fn meets_word_threshold(transcript: &str) -> bool {
    count_spoken_words(&clean_transcript(transcript)) >= MIN_SPOKEN_WORDS
}

/// Builds `MarkdownSections` from the arguments of a `save_meeting_report` call.
fn sections_from_args(args: &Value) -> MarkdownSections {
    let field = |name: &str| {
        args.get(name)
            .and_then(Value::as_str)
            .map(|value| clip(value.trim(), MAX_FIELD_CHARS))
            .unwrap_or_default()
    };
    MarkdownSections {
        executive_summary: field("executive_summary"),
        key_decisions: field("key_decisions"),
        topics_and_discussion: field("topics_and_discussion"),
        action_items: field("action_items"),
        key_questions: field("key_questions"),
        open_issues: field("open_issues"),
        important_facts: field("important_facts"),
    }
}

/// Arguments of a native tool call, accepting a JSON string or an object.
fn call_arguments(call: &Value) -> Value {
    match call.pointer("/function/arguments") {
        Some(Value::String(raw)) => serde_json::from_str(raw).unwrap_or_else(|_| json!({})),
        Some(value) => value.clone(),
        None => json!({}),
    }
}

fn call_name(call: &Value) -> &str {
    call.pointer("/function/name")
        .and_then(Value::as_str)
        .unwrap_or_default()
}

/// Drops the oldest exchange while the history exceeds the input budget. A `tool` result is
/// removed together with the message it belonged to: leaving one orphaned makes servers reject
/// the whole request.
fn trim_history(history: &mut Vec<Value>, budget: usize) {
    while history.len() > 3 {
        let used = estimate_messages_tokens(
            history
                .iter()
                .filter_map(|message| message.get("content").and_then(Value::as_str)),
        );
        if used <= budget {
            break;
        }
        history.remove(2);
        while role_of(history.get(2)) == Some("tool") {
            history.remove(2);
        }
    }
}

fn role_of(message: Option<&Value>) -> Option<&str> {
    message?.get("role")?.as_str()
}

fn map_system_prompt(output_language: &str) -> String {
    format!(
        "You extract factual meeting notes from one transcript chunk. \
Write concise structured notes in {output_language} covering only what this chunk states:\n\
- Participants, date/time if mentioned\n\
- Topics and step-by-step workflows/processes (trigger, steps, rules, conditions, exceptions, systems)\n\
- Explicit decisions/agreements (with reason if stated)\n\
- Action items (owner or Not specified, task, priority/status if stated, due date, timestamp [HH:MM:SS] if present)\n\
- Key questions (with answer if resolved here, or Open)\n\
- Open issues, bugs, blockers, or risks (with impact/next step if stated)\n\
- Important facts (dates, numbers, limits, endpoints, environments, laws, systems)\n\
Never invent information. Keep speaker labels and timestamps."
    )
}

fn reduce_system_prompt(output_language: &str) -> String {
    format!(
        "You are meeting-report-analyst. Synthesize the chunk notes into the final meeting report \
by calling save_meeting_report once.\n\
Rules: follow the SKILL instructions; write all report content in OUTPUT_LANGUAGE = {output_language}; \
merge repeated topics across chunks; prefer later clarifications, answers, and final agreed positions; \
preserve step-by-step workflows/processes and narrative executive summary; never invent facts.\n\
Deliver the report ONLY by calling save_meeting_report."
    )
}

/// Analyzes the transcript and returns the report sections.
///
/// Single-chunk transcripts use the proven single-pass tool loop; multi-chunk transcripts use
/// map-reduce (one tool-free note-taking request per chunk, then one reduce request calling
/// `save_meeting_report`) so no request re-sends an ever-growing transcript history.
pub async fn analyze_transcript(
    provider: &OpenAiCompatibleProvider,
    transcript: &str,
    output_language: &str,
    skill_prompt: &str,
) -> Result<MarkdownSections, AiError> {
    let cleaned = clean_transcript(transcript);
    if count_spoken_words(&cleaned) < MIN_SPOKEN_WORDS {
        return Ok(MarkdownSections::default());
    }
    let working = compact_transcript_for_analysis(&cleaned);
    let skill = if skill_prompt.trim().is_empty() {
        default_report_skill_prompt()
    } else {
        skill_prompt.trim().to_string()
    };
    let system = analysis_system_prompt(output_language);
    let tools = meeting_tools();
    let tool_choice = json!("auto");
    let tools_text: String = tools.iter().map(Value::to_string).collect();
    let tool_choice_text = tool_choice.to_string();
    let first_user = "Meeting transcript: call read_meeting_context first, then read_transcript \
for every chunk 0..last, then save_meeting_report."
        .to_string();

    let input_budget = provider.config().input_budget_tokens();
    let tools_overhead =
        estimate_messages_tokens([tools_text.as_str(), tool_choice_text.as_str()])
            + ROUNDING_SLACK_TOKENS;
    let fixed = estimate_messages_tokens([system.as_str(), first_user.as_str()]) + tools_overhead;
    let chunk_budget = input_budget
        .saturating_sub(fixed)
        .clamp(256, MAX_CHUNK_TOKENS);
    let context = ReportContext {
        skill,
        output_language: output_language.to_string(),
        chunks: chunk_transcript(&working, chunk_budget),
    };

    if context.chunks.len() > 1 {
        return analyze_multi_chunk_map_reduce(provider, &context, input_budget).await;
    }

    analyze_single_chunk(
        provider,
        &context,
        &system,
        &first_user,
        &tools,
        &tool_choice,
        input_budget,
        tools_overhead,
    )
    .await
}

async fn analyze_multi_chunk_map_reduce(
    provider: &OpenAiCompatibleProvider,
    context: &ReportContext,
    input_budget: usize,
) -> Result<MarkdownSections, AiError> {
    let map_system = map_system_prompt(&context.output_language);
    let total = context.chunks.len();
    let mut notes = Vec::with_capacity(total);
    for (idx, chunk) in context.chunks.iter().enumerate() {
        let user = format!("Chunk {} of {}:\n\n{}", idx + 1, total, chunk);
        let reply = provider
            .chat_complete(&[
                ChatMessage::system(map_system.clone()),
                ChatMessage::user(user),
            ])
            .await?;
        let trimmed = reply.trim();
        if !trimmed.is_empty() {
            notes.push(format!("### Chunk {}/{}\n{}", idx + 1, total, trimmed));
        }
    }

    let tools = vec![save_report_tool()];
    let tool_choice = json!("auto");
    let tools_text = serde_json::to_string(&tools).unwrap_or_default();
    let choice_text = tool_choice.to_string();
    let reduce_system = reduce_system_prompt(&context.output_language);
    let prefix = format!(
        "OUTPUT_LANGUAGE = {}\n\nSKILL:\n{}\n\nCHUNK NOTES:\n",
        context.output_language, context.skill
    );
    let fixed_tokens = estimate_messages_tokens([
        reduce_system.as_str(),
        prefix.as_str(),
        tools_text.as_str(),
        choice_text.as_str(),
    ]) + ROUNDING_SLACK_TOKENS;
    let notes_budget_bytes =
        max_bytes_for_tokens(input_budget.saturating_sub(fixed_tokens).max(256));

    let mut combined_notes = notes.join("\n\n");
    if combined_notes.len() > notes_budget_bytes && !notes.is_empty() {
        let per_note_chars = (notes_budget_bytes / notes.len()).saturating_sub(16).max(120);
        combined_notes = notes
            .iter()
            .map(|note| clip(note, per_note_chars))
            .collect::<Vec<_>>()
            .join("\n\n");
    }

    let mut history = vec![
        json!({ "role": "system", "content": reduce_system }),
        json!({ "role": "user", "content": format!("{prefix}{combined_notes}") }),
    ];
    let allowed = [SAVE_REPORT_TOOL_NAME];
    let mut nudged = false;
    let mut save_retried = false;
    let mut gathered: Option<MarkdownSections> = None;
    let mut last_error: Option<String> = None;

    for _round in 0..3 {
        let reply = provider
            .chat_with_tools(&history, &tools, tool_choice.clone())
            .await?;

        let native = reply.tool_calls();
        if let Some(call) = native.iter().find(|c| call_name(c) == SAVE_REPORT_TOOL_NAME) {
            let sections = sections_from_args(&call_arguments(call));
            if sections.is_empty() && !save_retried {
                save_retried = true;
                history.push(reply.message.clone());
                history.push(json!({
                    "role": "user",
                    "content": "(system: save_meeting_report arrived empty. Call it again with the seven sections filled from the chunk notes.)",
                }));
                continue;
            }
            if sections.is_empty() {
                last_error = Some(NO_REPORT_MESSAGE.to_string());
            } else {
                gathered = Some(sections);
            }
            break;
        }

        let parsed = parse_tool_calls(reply.content(), &allowed);
        if let Some(call) = parsed.calls.iter().find(|c| c.name == SAVE_REPORT_TOOL_NAME) {
            let sections = sections_from_args(&call.arguments);
            if sections.is_empty() && !save_retried {
                save_retried = true;
                history.push(json!({
                    "role": "user",
                    "content": "(system: save_meeting_report arrived empty. Call it again with the seven sections filled from the chunk notes.)",
                }));
                continue;
            }
            if sections.is_empty() {
                last_error = Some(NO_REPORT_MESSAGE.to_string());
            } else {
                gathered = Some(sections);
            }
            break;
        }

        if !nudged {
            nudged = true;
            history.push(json!({
                "role": "user",
                "content": "(system: call save_meeting_report now with the report.)",
            }));
            continue;
        }
        last_error = Some(NO_REPORT_MESSAGE.to_string());
        break;
    }

    match gathered {
        Some(sections) if !sections.is_empty() => Ok(sections),
        _ => Err(AiError::InvalidResponse(
            last_error.unwrap_or_else(|| NO_REPORT_MESSAGE.to_string()),
        )),
    }
}

#[allow(clippy::too_many_arguments)]
async fn analyze_single_chunk(
    provider: &OpenAiCompatibleProvider,
    context: &ReportContext,
    system: &str,
    first_user: &str,
    tools: &[Value],
    tool_choice: &Value,
    input_budget: usize,
    tools_overhead: usize,
) -> Result<MarkdownSections, AiError> {
    let history_budget = input_budget.saturating_sub(tools_overhead);
    let max_rounds = (context.chunks.len() + 8).clamp(MAX_ROUNDS, MAX_ROUND_CAP);

    let mut history = vec![
        json!({ "role": "system", "content": system }),
        json!({ "role": "user", "content": first_user }),
    ];
    let allowed = ReportContext::tool_names();
    let mut nudged = false;
    let mut save_retried = false;
    let mut gathered: Option<MarkdownSections> = None;
    let mut last_error: Option<String> = None;

    for _round in 0..max_rounds {
        trim_history(&mut history, history_budget);
        let reply = provider
            .chat_with_tools(&history, tools, tool_choice.clone())
            .await?;

        let native = reply.tool_calls();
        if !native.is_empty() {
            history.push(reply.message.clone());
            for call in native {
                let name = call_name(call).to_string();
                let args = call_arguments(call);
                if name == SAVE_REPORT_TOOL_NAME {
                    let sections = sections_from_args(&args);
                    if sections.is_empty() && !save_retried {
                        // Small models sometimes call the tool before reading anything.
                        save_retried = true;
                        history.push(json!({
                            "role": "user",
                            "content": "(system: save_meeting_report arrived empty. Call it again with the seven sections filled from the transcript.)",
                        }));
                    } else if sections.is_empty() {
                        last_error = Some(NO_REPORT_MESSAGE.to_string());
                    } else {
                        gathered = Some(sections);
                    }
                    break;
                }
                let id = call.get("id").cloned().unwrap_or(Value::Null);
                let result = context.call(&name, &args);
                history.push(json!({ "role": "tool", "tool_call_id": id, "content": result }));
            }
            if gathered.is_some() {
                break;
            }
            continue;
        }

        // The server may drop `tools` when the template declares no tool support: read the
        // call the model wrote as text instead.
        let parsed = parse_tool_calls(reply.content(), &allowed);
        if parsed
            .calls
            .iter()
            .any(|call| call.name == SAVE_REPORT_TOOL_NAME)
        {
            let call = parsed
                .calls
                .iter()
                .find(|call| call.name == SAVE_REPORT_TOOL_NAME)
                .expect("checked above");
            let sections = sections_from_args(&call.arguments);
            if sections.is_empty() && !save_retried {
                save_retried = true;
                history.push(json!({
                    "role": "user",
                    "content": "(system: save_meeting_report arrived empty. Call it again with the seven sections filled from the transcript.)",
                }));
                continue;
            }
            if sections.is_empty() {
                last_error = Some(NO_REPORT_MESSAGE.to_string());
                break;
            }
            gathered = Some(sections);
            break;
        }
        if parsed.calls.is_empty() {
            if !nudged {
                nudged = true;
                history.push(json!({
                    "role": "user",
                    "content": "(system: call save_meeting_report now with the report, or say plainly what is missing.)",
                }));
                continue;
            }
            last_error = Some(NO_REPORT_MESSAGE.to_string());
            break;
        }
        // Results go back as one synthetic user message: templates that ignored `tools` also
        // ignore role "tool" when the assistant message had no native calls.
        let results: Vec<String> = parsed
            .calls
            .iter()
            .map(|call| {
                let text = context.call(&call.name, &call.arguments);
                format!("- {} -> \"{}\"", call.name, clip(&text, MAX_TOOL_RESULT_CHARS))
            })
            .collect();
        history.push(json!({
            "role": "user",
            "content": format!("(system: tool results - {})", results.join("; ")),
        }));
    }

    match gathered {
        Some(sections) if !sections.is_empty() => Ok(sections),
        _ => Err(AiError::InvalidResponse(
            last_error.unwrap_or_else(|| NO_REPORT_MESSAGE.to_string()),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::ModelConfig;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

    fn provider(server: &MockServer) -> OpenAiCompatibleProvider {
        OpenAiCompatibleProvider::new(ModelConfig {
            endpoint: format!("{}/v1", server.uri()),
            model: "m".to_string(),
            context_limit_tokens: 16384,
            max_output_tokens: 2560,
            ..ModelConfig::default()
        })
        .expect("provider")
    }

    fn meeting() -> String {
        (0..40)
            .map(|index| format!("linea {index} de la reunion con suficiente contenido hablado"))
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn save_call(summary: &str) -> String {
        format!(
            "```tool_call\n{{\"name\": \"save_meeting_report\", \"arguments\": \
{{\"executive_summary\": \"{summary}\"}}}}\n```"
        )
    }

    /// Replies with the bodies in order, one per request.
    struct Sequence {
        bodies: Vec<Value>,
        seen: Arc<AtomicUsize>,
    }

    impl Respond for Sequence {
        fn respond(&self, _request: &Request) -> ResponseTemplate {
            let index = self.seen.fetch_add(1, Ordering::SeqCst);
            let body = self
                .bodies
                .get(index.min(self.bodies.len().saturating_sub(1)))
                .cloned()
                .unwrap_or_else(|| json!({}));
            ResponseTemplate::new(200).set_body_json(body)
        }
    }

    async fn serve(server: &MockServer, bodies: Vec<Value>) -> Arc<AtomicUsize> {
        let seen = Arc::new(AtomicUsize::new(0));
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(Sequence {
                bodies,
                seen: Arc::clone(&seen),
            })
            .mount(server)
            .await;
        seen
    }

    fn reply(content: &str, finish: &str) -> Value {
        json!({"choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": finish}]})
    }

    fn tool_reply(name: &str, arguments: Value) -> Value {
        json!({"choices": [{
            "message": {"role": "assistant", "content": "", "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": name, "arguments": arguments.to_string()}}
            ]},
            "finish_reason": "tool_calls"
        }]})
    }

    #[test]
    fn the_prompt_is_short_english_and_teaches_the_tools() {
        let prompt = analysis_system_prompt("Spanish");
        assert!(prompt.len() < 1_400, "{}", prompt.len());
        for tool in ReportContext::tool_names() {
            assert!(prompt.contains(tool), "{tool}");
        }
        assert!(prompt.contains("```tool_call"));
        assert!(prompt.contains("OUTPUT_LANGUAGE"));
        assert!(prompt.contains("Never say you will do something"));
        for spanish in ["Eres", "reunión", "informe", "herramienta"] {
            assert!(!prompt.contains(spanish), "{spanish}");
        }
    }

    #[test]
    fn tools_use_flat_section_strings() {
        let tools = meeting_tools();
        let save = tools
            .iter()
            .find(|tool| tool.pointer("/function/name").and_then(Value::as_str) == Some(SAVE_REPORT_TOOL_NAME))
            .expect("save tool");
        let properties = &save["function"]["parameters"]["properties"];
        for section in [
            "executive_summary",
            "key_decisions",
            "topics_and_discussion",
            "action_items",
            "key_questions",
            "open_issues",
            "important_facts",
        ] {
            assert_eq!(properties[section]["type"], "string", "{section}");
        }
    }

    #[tokio::test]
    async fn a_plain_text_tool_call_still_produces_the_report() {
        let server = MockServer::start().await;
        serve(&server, vec![reply(&save_call("- resumen"), "stop")]).await;

        let sections = analyze_transcript(&provider(&server), &meeting(), "Spanish", "")
            .await
            .unwrap();

        assert_eq!(sections.executive_summary, "- resumen");
    }

    #[tokio::test]
    async fn native_calls_read_the_context_and_then_save() {
        let server = MockServer::start().await;
        let seen = serve(
            &server,
            vec![
                tool_reply(READ_CONTEXT_TOOL_NAME, json!({})),
                tool_reply(READ_TRANSCRIPT_TOOL_NAME, json!({ "chunk": 0 })),
                tool_reply(SAVE_REPORT_TOOL_NAME, json!({ "executive_summary": "- nativo" })),
            ],
        )
        .await;

        let sections = analyze_transcript(&provider(&server), &meeting(), "Spanish", "")
            .await
            .unwrap();

        assert_eq!(sections.executive_summary, "- nativo");
        assert_eq!(seen.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn a_truncated_reply_is_retried_once() {
        let server = MockServer::start().await;
        let seen = serve(
            &server,
            vec![reply("", "length"), reply(&save_call("- reintento"), "stop")],
        )
        .await;

        let sections = analyze_transcript(&provider(&server), &meeting(), "Spanish", "")
            .await
            .unwrap();

        assert_eq!(sections.executive_summary, "- reintento");
        assert_eq!(seen.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn an_empty_reply_is_nudged_and_then_reported() {
        let server = MockServer::start().await;
        serve(
            &server,
            vec![
                tool_reply(READ_CONTEXT_TOOL_NAME, json!({})),
                reply("", "stop"),
                reply("", "stop"),
            ],
        )
        .await;

        let error = analyze_transcript(&provider(&server), &meeting(), "Spanish", "")
            .await
            .unwrap_err();

        assert!(error.to_string().contains("save_meeting_report"), "{error}");
    }

    #[test]
    fn trimming_never_leaves_a_tool_result_without_its_call() {
        let mut history = vec![
            json!({"role": "system", "content": "s"}),
            json!({"role": "user", "content": "u"}),
            json!({"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]}),
            json!({"role": "tool", "tool_call_id": "1", "content": "x".repeat(4000)}),
            json!({"role": "user", "content": "next"}),
        ];

        trim_history(&mut history, 10);

        for (index, message) in history.iter().enumerate() {
            if role_of(Some(message)) == Some("tool") {
                let previous = role_of(history.get(index.wrapping_sub(1)));
                assert_eq!(previous, Some("assistant"), "orphaned tool result at {index}");
            }
        }
        assert_eq!(role_of(history.first()), Some("system"));
        assert_eq!(role_of(history.last()), Some("user"));
    }

    #[tokio::test]
    async fn prose_without_a_call_is_nudged_once_before_failing() {
        let server = MockServer::start().await;
        let seen = serve(
            &server,
            vec![
                reply("I will now save the report.", "stop"),
                reply(&save_call("- ok"), "stop"),
            ],
        )
        .await;

        let sections = analyze_transcript(&provider(&server), &meeting(), "Spanish", "")
            .await
            .unwrap();

        assert_eq!(sections.executive_summary, "- ok");
        assert_eq!(seen.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn a_save_call_with_empty_arguments_is_retried() {
        let server = MockServer::start().await;
        serve(
            &server,
            vec![
                tool_reply(SAVE_REPORT_TOOL_NAME, json!({})),
                tool_reply(
                    SAVE_REPORT_TOOL_NAME,
                    json!({ "executive_summary": "- ok" }),
                ),
            ],
        )
        .await;

        let sections = analyze_transcript(&provider(&server), &meeting(), "Spanish", "")
            .await
            .unwrap();

        assert_eq!(sections.executive_summary, "- ok");
    }

    #[test]
    fn short_meetings_skip_the_model() {
        assert!(!meets_word_threshold("hola"));
        let sections = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(async {
                let server = MockServer::start().await;
                analyze_transcript(&provider(&server), "hola", "Spanish", "")
                    .await
                    .unwrap()
            });
        assert!(sections.is_empty());
    }

    #[tokio::test]
    async fn three_chunk_transcript_uses_three_map_requests_and_one_reduce_request_within_budget() {
        // ~18,000 chars -> 3 chunks at MAX_CHUNK_TOKENS = 2_500 (~7,500 bytes/chunk).
        let long_meeting = (0..220)
            .map(|i| {
                format!(
                    "[15:{:02}:00] Turno {i}: explicacion detallada del flujo de base completa y predictivo.",
                    i % 60
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        let expected_chunks = chunk_transcript(&compact_transcript_for_analysis(&long_meeting), MAX_CHUNK_TOKENS);
        assert_eq!(expected_chunks.len(), 3, "test fixture must produce 3 chunks");

        let server = MockServer::start().await;
        let seen_requests = Arc::new(std::sync::Mutex::new(Vec::<Value>::new()));
        let seen_clone = Arc::clone(&seen_requests);
        let budget = provider(&server).config().input_budget_tokens();

        struct MapReduceResponder {
            seen: Arc<std::sync::Mutex<Vec<Value>>>,
            budget: usize,
        }
        impl Respond for MapReduceResponder {
            fn respond(&self, request: &Request) -> ResponseTemplate {
                let body: Value = serde_json::from_slice(&request.body).unwrap();
                let mut guard = self.seen.lock().unwrap();
                let step = guard.len();
                guard.push(body.clone());
                drop(guard);

                // Verify no request exceeds the input budget.
                let messages = body["messages"].as_array().unwrap();
                let msg_tokens = estimate_messages_tokens(
                    messages.iter().filter_map(|m| m["content"].as_str()),
                );
                let tools_tokens = body
                    .get("tools")
                    .map(|t| estimate_messages_tokens([t.to_string().as_str()]))
                    .unwrap_or(0);
                assert!(
                    msg_tokens + tools_tokens <= self.budget,
                    "request {step} exceeded budget: {} > {}",
                    msg_tokens + tools_tokens,
                    self.budget
                );

                if step < 3 {
                    // Map requests must be tool-free.
                    assert!(
                        body.get("tools").is_none(),
                        "map request {step} should not send tools"
                    );
                    return ResponseTemplate::new(200).set_body_json(json!({
                        "choices": [{
                            "message": {
                                "role": "assistant",
                                "content": format!("Notas del bloque {}: flujo predictivo y base completa.", step + 1)
                            },
                            "finish_reason": "stop"
                        }]
                    }));
                }

                // Reduce request (step == 3) must carry tools and call save_meeting_report.
                assert!(body.get("tools").is_some(), "reduce request must send tools");
                let user_text = messages
                    .iter()
                    .filter_map(|m| m["content"].as_str())
                    .collect::<Vec<_>>()
                    .join("\n");
                for idx in 1..=3 {
                    assert!(
                        user_text.contains(&format!("Notas del bloque {idx}")),
                        "reduce request missing map notes {idx}"
                    );
                }
                let long_topics = "### Flujo de base completa\n\n".to_string()
                    + &"Detalle del flujo. ".repeat(300);
                ResponseTemplate::new(200).set_body_json(tool_reply(
                    SAVE_REPORT_TOOL_NAME,
                    json!({
                        "executive_summary": "Resumen narrativo completo.",
                        "topics_and_discussion": long_topics
                    }),
                ))
            }
        }

        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(MapReduceResponder {
                seen: seen_clone,
                budget,
            })
            .mount(&server)
            .await;

        let sections = analyze_transcript(&provider(&server), &long_meeting, "Spanish", "")
            .await
            .unwrap();

        assert_eq!(seen_requests.lock().unwrap().len(), 4, "expected 3 map + 1 reduce");
        assert_eq!(sections.executive_summary, "Resumen narrativo completo.");
        assert!(
            sections.topics_and_discussion.len() > 4_500,
            "topics_and_discussion should not be clipped at 4000 chars, got {}",
            sections.topics_and_discussion.len()
        );
    }

    #[tokio::test]
    async fn single_chunk_transcript_keeps_single_pass_tool_loop() {
        let server = MockServer::start().await;
        let seen = serve(
            &server,
            vec![
                tool_reply(READ_CONTEXT_TOOL_NAME, json!({})),
                tool_reply(READ_TRANSCRIPT_TOOL_NAME, json!({ "chunk": 0 })),
                tool_reply(SAVE_REPORT_TOOL_NAME, json!({ "executive_summary": "Resumen único." })),
            ],
        )
        .await;

        let sections = analyze_transcript(&provider(&server), &meeting(), "Spanish", "")
            .await
            .unwrap();

        assert_eq!(sections.executive_summary, "Resumen único.");
        assert_eq!(seen.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn plain_text_tool_fallback_does_not_truncate_skill_or_chunk() {
        let long_skill = "SKILL_START ".to_string() + &"regla ".repeat(450) + "SKILL_END";
        assert!(long_skill.len() > 2_500);

        let server = MockServer::start().await;
        let seen = serve(
            &server,
            vec![
                reply("```tool_call\n{\"name\": \"read_meeting_context\", \"arguments\": {}}\n```", "stop"),
                reply(&save_call("Resumen completo."), "stop"),
            ],
        )
        .await;

        let sections = analyze_transcript(&provider(&server), &meeting(), "Spanish", &long_skill)
            .await
            .unwrap();
        assert_eq!(sections.executive_summary, "Resumen completo.");
        assert_eq!(seen.load(Ordering::SeqCst), 2);

        let requests = server.received_requests().await.unwrap();
        let second_body: Value = serde_json::from_slice(&requests[1].body).unwrap();
        let last_user = second_body["messages"]
            .as_array()
            .unwrap()
            .last()
            .unwrap()["content"]
            .as_str()
            .unwrap();
        assert!(
            last_user.contains("SKILL_END"),
            "plain-text fallback truncated the skill prompt: {last_user}"
        );
    }
}
