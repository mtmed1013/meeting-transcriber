//! Meeting report rendering (`meeting-report-analyst` contract), transcript-preserving report
//! stripping, and atomic in-place file replacement.

use std::io::Write;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::CoreError;

const ENGLISH_TITLE: &str = "# Meeting Report";
const SPANISH_TITLE: &str = "# Reporte de la reunión";
const ENGLISH_TRANSCRIPT_HEADING: &str = "# Full Transcript";
const SPANISH_TRANSCRIPT_HEADING: &str = "# Transcripción completa";
const SEPARATOR: &str = "---";
/// The transcript heading is fixed by the report contract in both languages. Stripping also
/// accepts the Spanish variant written by older builds.
const TRANSCRIPT_HEADING: &str = ENGLISH_TRANSCRIPT_HEADING;

const LEGACY_DEFAULT_REPORT_SKILL_PROMPT_V1: &str = r#"You are meeting-report-analyst. Turn a meeting transcript into an accurate, structured report so that someone who did not attend understands its purpose, topics, decisions, action items, questions, blockers, and key facts.

LANGUAGE: OUTPUT_LANGUAGE is authoritative for all generated text. Do not detect or switch language from the transcript. Keep technical terms, code, API names, URLs, product names, and proper nouns unchanged.

ACCURACY:
- Read the whole transcript first. Use only what it states.
- Never invent decisions, tasks, owners, deadlines, names, dates, numbers, requirements, risks, or answers.
- Never invent sentiment, engagement, or quality metrics.
- Use null for a missing owner or due date. Never guess an owner. Keep vague deadlines verbatim, such as 'tomorrow'.
- Keep speaker labels as written.
- When statements change, prefer the final agreed position.
- Fix obvious speech-to-text errors from context. Never guess uncertain meaning.

SECTIONS (call save_meeting_report):
- executive_summary: 3 to 7 bullets, 150 words at most: purpose, main discussion, outcome, decisions, next steps, major blockers. Skip small talk.
- key_decisions: explicit decisions, commitments, or approvals only. A suggestion ('maybe', 'we could') is not a decision. Add reason only if explicitly discussed.
- topics_and_discussion: 3 to 8 chronological chapters, merging related turns. timestamp only if stated, else null. points: substantive summaries.
- action_items: executable commitments only, with owner or null, task, and due or null.
- key_questions: material questions about requirements, architecture, rules, risks, deadlines, or ownership. Give answer, or status 'Open' or 'Partially resolved' in OUTPUT_LANGUAGE.
- open_issues: unresolved problems, blockers, or risks explicitly discussed. Add impact and next_step only if stated.
- important_facts: exact dates, deadlines, limits, quantities, prices, percentages, URLs, systems, versions, constraints, dependencies.

DELIVERY: Deliver all sections only through save_meeting_report. Use empty arrays when nothing applies. Never reply with prose or Markdown."#;

/// Returns `true` when `prompt` matches an unmodified previous built-in default skill prompt,
/// allowing saved settings to upgrade automatically without overwriting user customizations.
pub fn is_legacy_default_report_skill_prompt(prompt: &str) -> bool {
    prompt.trim() == LEGACY_DEFAULT_REPORT_SKILL_PROMPT_V1.trim()
}

/// Default instructions for the `meeting-report-analyst` skill. The user can override them in
/// settings; a blank override falls back to this text.
pub fn default_report_skill_prompt() -> String {
    r#"You are meeting-report-analyst. Turn a meeting transcript into a thorough, accurate, structured Markdown report so someone who did not attend understands the purpose, workflows, topics, decisions, action items, questions, blockers, and key facts.

LANGUAGE: OUTPUT_LANGUAGE is authoritative for all generated text. Do not detect or switch language from the transcript. Keep technical terms, code, API names, URLs, product names, and proper nouns unchanged.

ACCURACY:
- Read ALL transcript chunks from start to finish before writing. Later turns often clarify, answer, or override earlier ones.
- Use only what the transcript states. Never invent decisions, tasks, owners, deadlines, names, dates, numbers, requirements, risks, or answers.
- Never invent sentiment, engagement, or quality metrics.
- Write 'Not specified' (translated to OUTPUT_LANGUAGE) for a missing owner or due date. Never guess an owner. Keep vague deadlines verbatim, such as 'tomorrow'.
- Keep speaker labels as written. When statements change, prefer the final agreed position.
- Fix obvious speech-to-text errors from context. Never guess uncertain meaning.

STYLE & DEPTH (avoid shallow one-line bullets):
- Write informative, concrete prose and structured details, not vague one-line dots.
- Explicitly document every workflow or process ('flujo') discussed: explain how it works end-to-end (trigger/entry, steps, rules, conditions, exceptions, systems involved, and differences between flows).

SECTIONS (call save_meeting_report with Markdown strings):
- executive_summary: 1 to 2 clear narrative paragraphs (not bare bullets) covering date/time if stated, objective, participants, main topics and workflows discussed, key decisions, and next steps. Skip small talk.
- key_decisions: explicit decisions, commitments, or approvals only (a suggestion like 'maybe' or 'we could' is not a decision), with reason/context when discussed.
- topics_and_discussion: chronological chapters (`### Title — HH:MM:SS` when timestamp exists, else `### Title`). Merge related turns and explain the substance, reasoning, technical details, and step-by-step workflows/processes discussed.
- action_items: Markdown checklist (`- [ ] ...`) for executable commitments only, with Owner, specific Task, Due date, plus Priority, Status, and Timestamp (`[HH:MM:SS]`) when present in the transcript.
- key_questions: material questions about requirements, architecture, rules, workflows, risks, deadlines, or ownership. Give the answer if resolved later, or status 'Open'/'Partially resolved' in OUTPUT_LANGUAGE, plus why it matters.
- open_issues: unresolved problems, bugs, blockers, or risks explicitly discussed, with impact and next step when stated.
- important_facts: exact dates, deadlines, limits, quantities, prices, percentages, URLs, endpoints, environments, laws/rules, systems, versions, constraints, and dependencies.

DELIVERY: Deliver all sections only through save_meeting_report. Use an empty string "" when nothing applies to a section. Never reply with prose outside the tool call."#
        .to_string()
}

/// Whether the requested output language is English. Anything else renders in Spanish, which is
/// the application's default output language.
fn is_english(output_language: &str) -> bool {
    let lang = output_language.trim().to_lowercase();
    lang == "english" || lang == "en" || lang.starts_with("en-") || lang.starts_with("en_")
}

/// Labels for one output language.
struct Labels {
    title: &'static str,
    none: &'static str,
    unspecified: &'static str,
    summary: &'static str,
    decisions: &'static str,
    topics: &'static str,
    actions: &'static str,
    questions: &'static str,
    issues: &'static str,
    facts: &'static str,
}

const ENGLISH: Labels = Labels {
    title: ENGLISH_TITLE,
    none: "None identified.",
    unspecified: "Not specified",
    summary: "## Executive Summary",
    decisions: "## Key Decisions and Agreements",
    topics: "## Topics and Discussion",
    actions: "## Action Items",
    questions: "## Key Questions",
    issues: "## Open Issues and Blockers",
    facts: "## Important Facts and Requirements",
};

const SPANISH: Labels = Labels {
    title: SPANISH_TITLE,
    none: "Ninguno identificado.",
    unspecified: "No especificado",
    summary: "## Resumen ejecutivo",
    decisions: "## Decisiones y acuerdos clave",
    topics: "## Temas y discusión",
    actions: "## Acciones pendientes",
    questions: "## Preguntas clave",
    issues: "## Temas abiertos y bloqueos",
    facts: "## Datos y requisitos importantes",
};

fn labels_for(output_language: &str) -> &'static Labels {
    if is_english(output_language) {
        &ENGLISH
    } else {
        &SPANISH
    }
}

/// Collapses text to one trimmed line; returns an empty string when nothing is left.
fn one_line(text: &str) -> String {
    text.lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

/// Joins the report body with the separator and the transcript, verbatim.
fn document(report: &str, transcript_heading: &str, original_transcript: &str) -> String {
    format!("{report}\n\n{SEPARATOR}\n\n{transcript_heading}\n\n{original_transcript}")
}

/// Renders a `## heading` block followed by its lines, or the "none" notice when there are none.
fn section_block(heading: &str, lines: &[String], none: &str) -> String {
    let body = if lines.is_empty() {
        none.to_string()
    } else {
        lines.join("\n")
    };
    format!("{heading}\n\n{body}")
}

/// Removes a previously generated report (and the optional `Full Transcript` heading) so the
/// original transcript can be recovered without duplicating reports.
///
/// Returns the input unchanged when it does not start with a report heading or has no `---`
/// separator line. Line endings are handled line by line, so CRLF files are supported.
pub fn strip_existing_report(markdown: &str) -> String {
    let lines: Vec<&str> = markdown.split_inclusive('\n').collect();
    let Some(first) = lines.first() else {
        return markdown.to_string();
    };
    let title = first.trim();
    if title != ENGLISH_TITLE && title != SPANISH_TITLE {
        return markdown.to_string();
    }
    let Some(separator) = lines.iter().position(|line| line.trim() == SEPARATOR) else {
        return markdown.to_string();
    };

    let mut index = separator + 1;
    // The separator is followed by one blank line, then an optional transcript heading and one more
    // blank line. Only that exact framing is removed, so transcript blank lines are preserved.
    if lines.get(index).is_some_and(|line| line.trim().is_empty()) {
        index += 1;
    }
    if lines
        .get(index)
        .map(|line| line.trim())
        .is_some_and(|line| line == ENGLISH_TRANSCRIPT_HEADING || line == SPANISH_TRANSCRIPT_HEADING)
    {
        index += 1;
        if lines.get(index).is_some_and(|line| line.trim().is_empty()) {
            index += 1;
        }
    }
    lines[index.min(lines.len())..].concat()
}

/// Report for a transcript below the word threshold: a short notice followed by the transcript.
pub fn short_meeting_markdown(output_language: &str, original_transcript: &str) -> String {
    let labels = labels_for(output_language);
    let notice = if is_english(output_language) {
        "Insufficient meeting content to generate a reliable report."
    } else {
        "Contenido insuficiente de la reunión para generar un reporte confiable."
    };
    document(
        &format!("{}\n\n{notice}", labels.title),
        TRANSCRIPT_HEADING,
        original_transcript,
    )
}

/// Renders a full report followed by the original transcript, verbatim, under `# Full Transcript`.
pub fn assemble_report_markdown(
    output_language: &str,
    sections: &ReportSections,
    original_transcript: &str,
) -> String {
    let l = labels_for(output_language);
    let en = is_english(output_language);
    let mut blocks = vec![l.title.to_string()];

    let summary: Vec<String> = sections
        .executive_summary
        .iter()
        .map(|item| one_line(item))
        .filter(|item| !item.is_empty())
        .map(|item| format!("- {item}"))
        .collect();
    blocks.push(section_block(l.summary, &summary, l.none));

    let (decision_label, reason_label) = if en {
        ("**Decision:**", "**Reason:**")
    } else {
        ("**Decisión:**", "**Motivo:**")
    };
    let mut decisions = Vec::new();
    for item in &sections.key_decisions {
        let decision = one_line(&item.decision);
        if decision.is_empty() {
            continue;
        }
        decisions.push(format!("- {decision_label} {decision}"));
        if let Some(reason) = item.reason.as_deref().map(one_line).filter(|r| !r.is_empty()) {
            decisions.push(format!("  - {reason_label} {reason}"));
        }
    }
    blocks.push(section_block(l.decisions, &decisions, l.none));

    let mut topics = Vec::new();
    for topic in &sections.topics_and_discussion {
        let title = one_line(&topic.title);
        if title.is_empty() {
            continue;
        }
        let timestamp = topic
            .timestamp
            .as_deref()
            .map(one_line)
            .filter(|t| !t.is_empty());
        let heading = match timestamp {
            Some(ts) => format!("### {title} — {ts}"),
            None => format!("### {title}"),
        };
        let points: Vec<String> = topic
            .points
            .iter()
            .map(|p| one_line(p))
            .filter(|p| !p.is_empty())
            .map(|p| format!("- {p}"))
            .collect();
        let mut block = heading;
        if !points.is_empty() {
            block.push_str("\n\n");
            block.push_str(&points.join("\n"));
        }
        topics.push(block);
    }
    blocks.push(match topics.is_empty() {
        true => section_block(l.topics, &[], l.none),
        false => format!("{}\n\n{}", l.topics, topics.join("\n\n")),
    });

    let (owner_label, task_label, due_label) = if en {
        ("**Owner:**", "**Task:**", "**Due:**")
    } else {
        ("**Responsable:**", "**Tarea:**", "**Fecha límite:**")
    };
    let mut actions = Vec::new();
    for item in &sections.action_items {
        let task = one_line(&item.task);
        if task.is_empty() {
            continue;
        }
        let owner = item
            .owner
            .as_deref()
            .map(one_line)
            .filter(|o| !o.is_empty())
            .unwrap_or_else(|| l.unspecified.to_string());
        let due = item
            .due
            .as_deref()
            .map(one_line)
            .filter(|d| !d.is_empty())
            .unwrap_or_else(|| l.unspecified.to_string());
        actions.push(format!(
            "- [ ] {owner_label} {owner} — {task_label} {task} — {due_label} {due}"
        ));
    }
    blocks.push(section_block(l.actions, &actions, l.none));

    let (question_label, answer_label, status_label) = if en {
        ("**Question:**", "**Answer:**", "**Status:**")
    } else {
        ("**Pregunta:**", "**Respuesta:**", "**Estado:**")
    };
    let mut questions = Vec::new();
    for item in &sections.key_questions {
        let question = one_line(&item.question);
        if question.is_empty() {
            continue;
        }
        questions.push(format!("- {question_label} {question}"));
        if let Some(answer) = item.answer.as_deref().map(one_line).filter(|a| !a.is_empty()) {
            questions.push(format!("  - {answer_label} {answer}"));
        }
        if let Some(status) = item.status.as_deref().map(one_line).filter(|s| !s.is_empty()) {
            questions.push(format!("  - {status_label} {status}"));
        }
    }
    blocks.push(section_block(l.questions, &questions, l.none));

    let (issue_label, impact_label, next_label) = if en {
        ("**Issue:**", "**Impact:**", "**Next step:**")
    } else {
        ("**Problema:**", "**Impacto:**", "**Siguiente paso:**")
    };
    let mut issues = Vec::new();
    for item in &sections.open_issues {
        let issue = one_line(&item.issue);
        if issue.is_empty() {
            continue;
        }
        issues.push(format!("- {issue_label} {issue}"));
        if let Some(impact) = item.impact.as_deref().map(one_line).filter(|i| !i.is_empty()) {
            issues.push(format!("  - {impact_label} {impact}"));
        }
        if let Some(next) = item.next_step.as_deref().map(one_line).filter(|n| !n.is_empty()) {
            issues.push(format!("  - {next_label} {next}"));
        }
    }
    blocks.push(section_block(l.issues, &issues, l.none));

    let facts: Vec<String> = sections
        .important_facts
        .iter()
        .map(|f| one_line(f))
        .filter(|f| !f.is_empty())
        .map(|f| format!("- {f}"))
        .collect();
    blocks.push(section_block(l.facts, &facts, l.none));

    document(&blocks.join("\n\n"), TRANSCRIPT_HEADING, original_transcript)
}

/// Normalises one Markdown section body written by the model: runs of three or more blank lines
/// shrink to two, a line starting with `- ` or `* ` gets a single `- ` marker, lines that would
/// read as the report separator or a reserved heading are backslash-escaped so the report
/// structure stays unambiguous, and the whole body is trimmed. Other Markdown is kept as written.
fn normalise_markdown_body(text: &str) -> String {
    let mut lines: Vec<String> = Vec::new();
    let mut blank_run = 0usize;
    for line in text.lines() {
        if line.trim().is_empty() {
            blank_run += 1;
            if blank_run <= 2 {
                lines.push(String::new());
            }
            continue;
        }
        blank_run = 0;
        lines.push(guard_reserved_line(&normalise_bullet(line)));
    }
    lines.join("\n").trim().to_string()
}

/// Rewrites a `- ` or `* ` bullet to a single `- ` marker, keeping the item text.
fn normalise_bullet(line: &str) -> String {
    match line.strip_prefix("- ").or_else(|| line.strip_prefix("* ")) {
        Some(rest) => {
            let rest = rest.trim_start();
            if rest.is_empty() {
                "-".to_string()
            } else {
                format!("- {rest}")
            }
        }
        None => line.to_string(),
    }
}

/// Escapes a line whose trimmed text equals the report separator or a reserved report or
/// transcript heading, so it cannot be mistaken for the structure that `strip_existing_report`
/// relies on. The escaped line still renders as the same text.
fn guard_reserved_line(line: &str) -> String {
    let content = line.trim();
    let reserved = [
        SEPARATOR,
        ENGLISH_TITLE,
        SPANISH_TITLE,
        ENGLISH_TRANSCRIPT_HEADING,
        SPANISH_TRANSCRIPT_HEADING,
    ];
    if reserved.contains(&content) {
        let indent = &line[..line.len() - line.trim_start().len()];
        format!("{indent}\\{}", line.trim_start())
    } else {
        line.to_string()
    }
}

/// Renders a `## heading` block whose body is normalised Markdown, or the "none" notice when the
/// body is blank.
fn markdown_section_block(heading: &str, text: &str, none: &str) -> String {
    let body = normalise_markdown_body(text);
    let lines = if body.is_empty() { Vec::new() } else { vec![body] };
    section_block(heading, &lines, none)
}

/// Renders a full report from Markdown section text, followed by the original transcript,
/// verbatim, under `# Full Transcript`. Each section body is normalised and an empty body shows
/// the localized "none" notice.
pub fn assemble_markdown_report(
    output_language: &str,
    sections: &MarkdownSections,
    original_transcript: &str,
) -> String {
    let l = labels_for(output_language);
    let blocks = [
        l.title.to_string(),
        markdown_section_block(l.summary, &sections.executive_summary, l.none),
        markdown_section_block(l.decisions, &sections.key_decisions, l.none),
        markdown_section_block(l.topics, &sections.topics_and_discussion, l.none),
        markdown_section_block(l.actions, &sections.action_items, l.none),
        markdown_section_block(l.questions, &sections.key_questions, l.none),
        markdown_section_block(l.issues, &sections.open_issues, l.none),
        markdown_section_block(l.facts, &sections.important_facts, l.none),
    ];
    document(&blocks.join("\n\n"), TRANSCRIPT_HEADING, original_transcript)
}

/// Replaces `path` with `content` via a temporary file in the same directory, synced before the
/// rename. Keeps the permissions of an existing file.
pub fn write_atomic_in_place(path: &Path, content: &str) -> Result<(), CoreError> {
    let parent = match path.parent() {
        Some(p) if !p.as_os_str().is_empty() => p,
        _ => Path::new("."),
    };
    let mut temp = tempfile::NamedTempFile::new_in(parent)?;
    temp.write_all(content.as_bytes())?;
    temp.flush()?;
    temp.as_file().sync_all()?;
    if let Ok(metadata) = std::fs::metadata(path) {
        std::fs::set_permissions(temp.path(), metadata.permissions())?;
    }
    temp.persist(path).map_err(|err| CoreError::Io(err.error))?;
    Ok(())
}

/// One key decision, with its reason when the meeting stated one.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DecisionItem {
    pub decision: String,
    pub reason: Option<String>,
}

/// One discussed topic. `timestamp` is the approximate start, when the transcript has one.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TopicItem {
    pub title: String,
    pub timestamp: Option<String>,
    pub points: Vec<String>,
}

/// One explicit follow-up task.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActionItem {
    pub owner: Option<String>,
    pub task: String,
    pub due: Option<String>,
}

/// One question that materially affected the meeting.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuestionItem {
    pub question: String,
    pub answer: Option<String>,
    pub status: Option<String>,
}

/// One unresolved issue or blocker.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct IssueItem {
    pub issue: String,
    pub impact: Option<String>,
    pub next_step: Option<String>,
}

/// Structured content of a meeting report. Empty vectors render as "none identified".
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReportSections {
    pub executive_summary: Vec<String>,
    pub key_decisions: Vec<DecisionItem>,
    pub topics_and_discussion: Vec<TopicItem>,
    pub action_items: Vec<ActionItem>,
    pub key_questions: Vec<QuestionItem>,
    pub open_issues: Vec<IssueItem>,
    pub important_facts: Vec<String>,
}

/// Report content written as finished Markdown per section, for models that answer with section
/// text instead of nested JSON. Empty or blank fields render as "none identified".
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct MarkdownSections {
    pub executive_summary: String,
    pub key_decisions: String,
    pub topics_and_discussion: String,
    pub action_items: String,
    pub key_questions: String,
    pub open_issues: String,
    pub important_facts: String,
}

impl MarkdownSections {
    /// Whether every section is blank after trimming whitespace.
    pub fn is_empty(&self) -> bool {
        [
            &self.executive_summary,
            &self.key_decisions,
            &self.topics_and_discussion,
            &self.action_items,
            &self.key_questions,
            &self.open_issues,
            &self.important_facts,
        ]
        .iter()
        .all(|section| section.trim().is_empty())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ES_TITLE: &str = "# Reporte de la reunión";
    const EN_TITLE: &str = "# Meeting Report";
    const SHORT_ES: &str =
        "# Reporte de la reunión\n\nContenido insuficiente de la reunión para generar un reporte confiable.";
    const SHORT_EN: &str =
        "# Meeting Report\n\nInsufficient meeting content to generate a reliable report.";

    fn full_sections() -> ReportSections {
        ReportSections {
            executive_summary: vec!["Se revisó el plan.".to_string()],
            key_decisions: vec![
                DecisionItem {
                    decision: "Usar MySQL.".to_string(),
                    reason: Some("Ya existe en producción.".to_string()),
                },
                DecisionItem {
                    decision: "Publicar el lunes.".to_string(),
                    reason: None,
                },
            ],
            topics_and_discussion: vec![TopicItem {
                title: "Estrategia de API".to_string(),
                timestamp: Some("00:18:32".to_string()),
                points: vec!["Mover Excel a un worker.".to_string(), "Sin bloqueos HTTP.".to_string()],
            }],
            action_items: vec![
                ActionItem {
                    owner: Some("Mateo".to_string()),
                    task: "Validar SSO".to_string(),
                    due: Some("viernes".to_string()),
                },
                ActionItem {
                    owner: None,
                    task: "Confirmar URL de producción".to_string(),
                    due: None,
                },
            ],
            key_questions: vec![
                QuestionItem {
                    question: "¿Cuál es el límite?".to_string(),
                    answer: Some("60 solicitudes por minuto.".to_string()),
                    status: None,
                },
                QuestionItem {
                    question: "¿Quién aprueba?".to_string(),
                    answer: None,
                    status: Some("Abierto".to_string()),
                },
            ],
            open_issues: vec![IssueItem {
                issue: "Falta el certificado.".to_string(),
                impact: Some("Bloquea el despliegue.".to_string()),
                next_step: None,
            }],
            important_facts: vec!["Límite: **2,000 filas**".to_string()],
        }
    }

    fn sections_english() -> ReportSections {
        ReportSections {
            key_decisions: vec![DecisionItem {
                decision: "Ship on Monday.".to_string(),
                reason: None,
            }],
            action_items: vec![ActionItem {
                owner: None,
                task: "Confirm the date".to_string(),
                due: None,
            }],
            key_questions: vec![QuestionItem {
                question: "Who approves?".to_string(),
                answer: None,
                status: Some("Open".to_string()),
            }],
            open_issues: vec![IssueItem {
                issue: "Certificate missing.".to_string(),
                impact: Some("Blocks deploy.".to_string()),
                next_step: Some("Request it.".to_string()),
            }],
            ..ReportSections::default()
        }
    }

    // ----- strip_existing_report -----

    #[test]
    fn strips_english_report_and_full_transcript_heading() {
        let md = format!(
            "{EN_TITLE}\n\n## Executive Summary\n\n- Plan\n\n---\n\n# Full Transcript\n\n# 🧠 Reunión\n\nhola\n"
        );
        assert_eq!(strip_existing_report(&md), "# 🧠 Reunión\n\nhola\n");
    }

    #[test]
    fn strips_spanish_report_with_spanish_transcript_heading() {
        let md = format!(
            "{ES_TITLE}\n\n## Resumen ejecutivo\n\n- Plan\n\n---\n\n# Transcripción completa\n\n**[10:00:01]**\nhola\n"
        );
        assert_eq!(strip_existing_report(&md), "**[10:00:01]**\nhola\n");
    }

    #[test]
    fn strips_report_when_transcript_heading_is_absent() {
        let md = format!("{ES_TITLE}\n\nResumen.\n\n---\n\n# 🧠 Reunión\n\ncuerpo");
        assert_eq!(strip_existing_report(&md), "# 🧠 Reunión\n\ncuerpo");
    }

    #[test]
    fn leaves_documents_without_a_leading_report_heading_unchanged() {
        let md = "# 🧠 Reunión\n\n---\n\nTexto\n";
        assert_eq!(strip_existing_report(md), md);
    }

    #[test]
    fn leaves_report_heading_without_separator_unchanged() {
        let md = format!("{EN_TITLE}\n\nSolo un resumen sin separador.\n");
        assert_eq!(strip_existing_report(&md), md);
    }

    #[test]
    fn heading_must_match_exactly_not_just_by_prefix() {
        let md = "# Meeting Reports\n\nx\n\n---\n\ncuerpo\n";
        assert_eq!(strip_existing_report(md), md);
    }

    #[test]
    fn separator_must_be_a_full_line() {
        let md = format!("{EN_TITLE}\n\nx ---\n\ncuerpo\n");
        assert_eq!(strip_existing_report(&md), md);
    }

    #[test]
    fn strips_crlf_reports_written_on_windows() {
        let md = "# Meeting Report\r\n\r\n## Executive Summary\r\n\r\n- x\r\n\r\n---\r\n\r\n# Full Transcript\r\n\r\n# 🧠 Reunión\r\n\r\nhola\r\n";
        assert_eq!(strip_existing_report(md), "# 🧠 Reunión\r\n\r\nhola\r\n");
    }

    #[test]
    fn keeps_internal_separator_lines_of_the_transcript() {
        let md = format!("{EN_TITLE}\n\n---\n\n# Full Transcript\n\nA\n---\nB\n");
        assert_eq!(strip_existing_report(&md), "A\n---\nB\n");
    }

    #[test]
    fn stripping_is_idempotent_on_the_recovered_transcript() {
        let transcript = "# 🧠 Reunión\n\n**[10:00:01]**\nHola.\n";
        let stripped = strip_existing_report(&assemble_report_markdown("Spanish", &full_sections(), transcript));
        assert_eq!(stripped, transcript);
        assert_eq!(strip_existing_report(&stripped), stripped);
    }

    #[test]
    fn strip_recovers_a_short_meeting_report_exactly() {
        let transcript = "hola\n";
        let md = short_meeting_markdown("English", transcript);
        assert_eq!(strip_existing_report(&md), transcript);
    }

    #[test]
    fn strip_recovers_a_transcript_that_begins_with_a_blank_line() {
        let transcript = "\nhola\n";
        let md = assemble_report_markdown("English", &ReportSections::default(), transcript);
        assert_eq!(strip_existing_report(&md), transcript);
    }

    // ----- short_meeting_markdown -----

    #[test]
    fn short_meeting_spanish_notice_then_transcript() {
        let out = short_meeting_markdown("Spanish", "hola");
        assert_eq!(out, format!("{SHORT_ES}\n\n---\n\n# Full Transcript\n\nhola"));
    }

    #[test]
    fn short_meeting_english_notice_then_transcript() {
        let out = short_meeting_markdown("English", "hi there");
        assert_eq!(out, format!("{SHORT_EN}\n\n---\n\n# Full Transcript\n\nhi there"));
    }

    #[test]
    fn short_meeting_preserves_transcript_verbatim() {
        let transcript = "  **[00:00:01]**  \n\t# not a heading\n---\n";
        let out = short_meeting_markdown("Spanish", transcript);
        assert!(out.ends_with(&format!("# Full Transcript\n\n{transcript}")), "{out}");
    }

    #[test]
    fn language_matching_is_case_insensitive_and_accepts_codes() {
        assert_eq!(short_meeting_markdown("english", "x"), short_meeting_markdown("English", "x"));
        assert_eq!(short_meeting_markdown("en", "x"), short_meeting_markdown("English", "x"));
        assert_eq!(short_meeting_markdown("es", "x"), short_meeting_markdown("Spanish", "x"));
        assert_eq!(short_meeting_markdown("Español", "x"), short_meeting_markdown("Spanish", "x"));
    }

    #[test]
    fn unknown_output_language_falls_back_to_spanish() {
        assert_eq!(short_meeting_markdown("Klingon", "x"), short_meeting_markdown("Spanish", "x"));
    }

    // ----- assemble_report_markdown -----

    #[test]
    fn assemble_spanish_full_report_matches_contract() {
        let transcript = "# 🧠 Reunión\n\nTexto original.\n";
        let out = assemble_report_markdown("Spanish", &full_sections(), transcript);
        let expected = "\
# Reporte de la reunión

## Resumen ejecutivo

- Se revisó el plan.

## Decisiones y acuerdos clave

- **Decisión:** Usar MySQL.
  - **Motivo:** Ya existe en producción.
- **Decisión:** Publicar el lunes.

## Temas y discusión

### Estrategia de API — 00:18:32

- Mover Excel a un worker.
- Sin bloqueos HTTP.

## Acciones pendientes

- [ ] **Responsable:** Mateo — **Tarea:** Validar SSO — **Fecha límite:** viernes
- [ ] **Responsable:** No especificado — **Tarea:** Confirmar URL de producción — **Fecha límite:** No especificado

## Preguntas clave

- **Pregunta:** ¿Cuál es el límite?
  - **Respuesta:** 60 solicitudes por minuto.
- **Pregunta:** ¿Quién aprueba?
  - **Estado:** Abierto

## Temas abiertos y bloqueos

- **Problema:** Falta el certificado.
  - **Impacto:** Bloquea el despliegue.

## Datos y requisitos importantes

- Límite: **2,000 filas**

---

# Full Transcript

# 🧠 Reunión

Texto original.
";
        assert_eq!(out, expected);
    }

    #[test]
    fn assemble_english_labels_and_headings() {
        let out = assemble_report_markdown("English", &sections_english(), "t");
        assert!(out.starts_with("# Meeting Report\n\n## Executive Summary\n\nNone identified.\n\n"), "{out}");
        assert!(out.contains("## Key Decisions and Agreements\n\n- **Decision:** Ship on Monday.\n\n"), "{out}");
        assert!(out.contains("## Topics and Discussion\n\nNone identified.\n\n"), "{out}");
        assert!(
            out.contains("- [ ] **Owner:** Not specified — **Task:** Confirm the date — **Due:** Not specified\n"),
            "{out}"
        );
        assert!(out.contains("## Key Questions\n\n- **Question:** Who approves?\n  - **Status:** Open\n"), "{out}");
        assert!(
            out.contains(
                "## Open Issues and Blockers\n\n- **Issue:** Certificate missing.\n  - **Impact:** Blocks deploy.\n  - **Next step:** Request it.\n"
            ),
            "{out}"
        );
        assert!(out.contains("## Important Facts and Requirements\n\nNone identified.\n\n---\n\n# Full Transcript\n\nt"), "{out}");
    }

    #[test]
    fn assemble_empty_sections_in_spanish_use_ninguno_identificado() {
        let out = assemble_report_markdown("Spanish", &ReportSections::default(), "");
        assert_eq!(out.matches("Ninguno identificado.").count(), 7);
        assert!(!out.contains("None identified."));
    }

    #[test]
    fn assemble_produces_exactly_one_report_and_one_transcript_heading() {
        let out = assemble_report_markdown("English", &full_sections(), "# 🧠 Reunión\n");
        assert_eq!(out.matches("# Meeting Report").count(), 1);
        assert_eq!(out.matches("# Full Transcript").count(), 1);
    }

    #[test]
    fn assemble_keeps_transcript_bytes_including_crlf_and_trailing_spaces() {
        let transcript = "línea uno  \r\n---\r\n  final";
        let out = assemble_report_markdown("Spanish", &ReportSections::default(), transcript);
        assert!(out.ends_with(&format!("# Full Transcript\n\n{transcript}")), "{out}");
    }

    #[test]
    fn assemble_skips_blank_optional_fields_and_items_without_text() {
        let sections = ReportSections {
            key_decisions: vec![DecisionItem {
                decision: "Sí".to_string(),
                reason: Some("   ".to_string()),
            }],
            action_items: vec![ActionItem {
                owner: Some("  ".to_string()),
                task: "  ".to_string(),
                due: Some("".to_string()),
            }],
            ..ReportSections::default()
        };
        let out = assemble_report_markdown("English", &sections, "t");
        assert!(out.contains("- **Decision:** Sí\n\n"), "{out}");
        assert!(!out.contains("Reason:"), "{out}");
        assert!(out.contains("## Action Items\n\nNone identified.\n"), "{out}");
    }

    #[test]
    fn assemble_collapses_multiline_item_text_into_one_bullet_line() {
        let sections = ReportSections {
            executive_summary: vec!["línea uno\nlínea dos".to_string()],
            ..ReportSections::default()
        };
        let out = assemble_report_markdown("English", &sections, "t");
        assert!(out.contains("- línea uno línea dos\n"), "{out}");
    }

    #[test]
    fn assemble_then_strip_recovers_transcript_for_each_language() {
        for lang in ["Spanish", "English"] {
            let transcript = "# 🧠 Reunión\n\n**[10:00:01]**\nHola.\n";
            let md = assemble_report_markdown(lang, &full_sections(), transcript);
            assert_eq!(strip_existing_report(&md), transcript, "{lang}");
        }
    }

    // ----- default_report_skill_prompt -----

    #[test]
    fn default_report_skill_prompt_states_the_contract_rules() {
        let prompt = default_report_skill_prompt();
        for expected in [
            "save_meeting_report",
            "OUTPUT_LANGUAGE",
            "Not specified",
            "Never invent",
            "suggestion",
            "final agreed position",
            "narrative",
            "workflow",
        ] {
            assert!(prompt.contains(expected), "missing {expected:?} in:\n{prompt}");
        }
        assert!(!prompt.contains("else null"), "stale JSON wording in:\n{prompt}");
    }

    #[test]
    fn default_report_skill_prompt_is_compact_and_covers_the_contract() {
        let prompt = default_report_skill_prompt();
        assert!(prompt.len() <= 3200, "{} bytes:\n{prompt}", prompt.len());
        for section in [
            "executive_summary",
            "key_decisions",
            "topics_and_discussion",
            "action_items",
            "key_questions",
            "open_issues",
            "important_facts",
        ] {
            assert!(prompt.contains(section), "missing section {section:?}");
        }
        for expected in [
            "OUTPUT_LANGUAGE",
            "speech-to-text",
            "final agreed position",
            "metrics",
            "save_meeting_report",
            "suggestion",
            "Never invent",
            "Not specified",
            "workflow",
            "narrative",
        ] {
            assert!(prompt.contains(expected), "missing {expected:?} in:\n{prompt}");
        }
    }

    // ----- MarkdownSections / assemble_markdown_report -----

    fn markdown_sections() -> MarkdownSections {
        MarkdownSections {
            executive_summary: "- Se revisó el plan.\n- Sin bloqueos.".to_string(),
            key_decisions: "- **Decisión:** Usar MySQL.".to_string(),
            topics_and_discussion: "### Estrategia de API\n\n- Mover Excel a un worker.".to_string(),
            action_items: "- [ ] **Responsable:** Mateo — **Tarea:** Validar SSO".to_string(),
            key_questions: "- **Pregunta:** ¿Quién aprueba?".to_string(),
            open_issues: "- **Problema:** Falta el certificado.".to_string(),
            important_facts: "Límite: **2,000 filas**".to_string(),
        }
    }

    #[test]
    fn markdown_report_spanish_headings_and_body_match_contract() {
        let out = assemble_markdown_report("Spanish", &markdown_sections(), "t");
        let expected = "\
# Reporte de la reunión

## Resumen ejecutivo

- Se revisó el plan.
- Sin bloqueos.

## Decisiones y acuerdos clave

- **Decisión:** Usar MySQL.

## Temas y discusión

### Estrategia de API

- Mover Excel a un worker.

## Acciones pendientes

- [ ] **Responsable:** Mateo — **Tarea:** Validar SSO

## Preguntas clave

- **Pregunta:** ¿Quién aprueba?

## Temas abiertos y bloqueos

- **Problema:** Falta el certificado.

## Datos y requisitos importantes

Límite: **2,000 filas**

---

# Full Transcript

t";
        assert_eq!(out, expected);
    }

    #[test]
    fn markdown_report_english_headings_and_none_label_for_empty_sections() {
        let sections = MarkdownSections {
            key_decisions: "- Ship on Monday.".to_string(),
            ..MarkdownSections::default()
        };
        let out = assemble_markdown_report("en", &sections, "t");
        assert!(out.starts_with("# Meeting Report\n\n## Executive Summary\n\nNone identified.\n\n"), "{out}");
        assert!(out.contains("## Key Decisions and Agreements\n\n- Ship on Monday.\n\n"), "{out}");
        assert!(
            out.contains("## Important Facts and Requirements\n\nNone identified.\n\n---\n\n# Full Transcript\n\nt"),
            "{out}"
        );
        assert_eq!(out.matches("None identified.").count(), 6, "{out}");
        assert!(!out.contains("Ninguno identificado."), "{out}");
    }

    #[test]
    fn markdown_report_whitespace_only_sections_are_treated_as_empty() {
        let sections = MarkdownSections {
            open_issues: " \n\t\n ".to_string(),
            ..MarkdownSections::default()
        };
        let out = assemble_markdown_report("Spanish", &sections, "t");
        assert!(out.contains("## Temas abiertos y bloqueos\n\nNinguno identificado.\n\n"), "{out}");
        assert_eq!(out.matches("Ninguno identificado.").count(), 7, "{out}");
    }

    #[test]
    fn markdown_report_normalises_bullet_markers_and_keeps_indentation() {
        let sections = MarkdownSections {
            executive_summary: "* a\n-  b\n*   c\n  * nested\nplain text".to_string(),
            ..MarkdownSections::default()
        };
        let out = assemble_markdown_report("English", &sections, "t");
        assert!(
            out.contains("## Executive Summary\n\n- a\n- b\n- c\n  * nested\nplain text\n\n"),
            "{out}"
        );
    }

    #[test]
    fn markdown_report_collapses_runs_of_three_or_more_blank_lines_and_trims_the_body() {
        let sections = MarkdownSections {
            executive_summary: "\n\na\n\n\n\nb\n   \n\n\n".to_string(),
            ..MarkdownSections::default()
        };
        let out = assemble_markdown_report("Spanish", &sections, "t");
        assert!(out.contains("## Resumen ejecutivo\n\na\n\n\nb\n\n## Decisiones y acuerdos clave"), "{out}");
        assert!(!out.contains("\n\n\n\n"), "{out}");
    }

    #[test]
    fn markdown_report_keeps_other_markdown_as_written() {
        let body = "**Límite:** 2,000\n| a | b |\n|---|---|\n> nota\n1. paso\n---x";
        let sections = MarkdownSections {
            important_facts: body.to_string(),
            ..MarkdownSections::default()
        };
        let out = assemble_markdown_report("English", &sections, "t");
        assert!(out.contains(&format!("## Important Facts and Requirements\n\n{body}\n\n---\n\n# Full Transcript\n\nt")), "{out}");
    }

    #[test]
    fn markdown_report_keeps_transcript_bytes_with_crlf_and_trailing_spaces() {
        let transcript = "línea uno  \r\n---\r\n  final  \r\n\r\n";
        let out = assemble_markdown_report("Spanish", &MarkdownSections::default(), transcript);
        assert!(out.ends_with(&format!("# Full Transcript\n\n{transcript}")), "{out}");
    }

    #[test]
    fn markdown_report_normalises_crlf_section_bodies() {
        let sections = MarkdownSections {
            executive_summary: "- a\r\n\r\n\r\n- b  \r\n".to_string(),
            ..MarkdownSections::default()
        };
        let out = assemble_markdown_report("English", &sections, "t");
        assert!(out.contains("## Executive Summary\n\n- a\n\n\n- b\n\n## Key Decisions"), "{out}");
    }

    #[test]
    fn markdown_report_escapes_section_lines_that_would_duplicate_report_structure() {
        let sections = MarkdownSections {
            important_facts: "antes\n---\n# Full Transcript\n# Meeting Report".to_string(),
            ..MarkdownSections::default()
        };
        let transcript = "# 🧠 Reunión\n\ncuerpo\n";
        let md = assemble_markdown_report("English", &sections, transcript);
        assert_eq!(md.lines().filter(|l| l.trim() == "---").count(), 1, "{md}");
        assert_eq!(md.lines().filter(|l| l.trim() == "# Full Transcript").count(), 1, "{md}");
        assert_eq!(md.lines().filter(|l| l.trim() == "# Meeting Report").count(), 1, "{md}");
        assert_eq!(strip_existing_report(&md), transcript);
    }

    #[test]
    fn markdown_report_has_one_report_and_transcript_heading_after_repeated_assembly() {
        let transcript = "# 🧠 Reunión\n\nHola.\n";
        let mut md = assemble_markdown_report("English", &markdown_sections(), transcript);
        for _ in 0..2 {
            let recovered = strip_existing_report(&md);
            md = assemble_markdown_report("English", &markdown_sections(), &recovered);
        }
        assert_eq!(md.lines().filter(|l| l.trim() == "# Meeting Report").count(), 1, "{md}");
        assert_eq!(md.lines().filter(|l| l.trim() == "# Full Transcript").count(), 1, "{md}");
        assert_eq!(strip_existing_report(&md), transcript);
    }

    #[test]
    fn markdown_sections_is_empty_only_when_every_field_is_blank() {
        assert!(MarkdownSections::default().is_empty());
        let blank = MarkdownSections {
            key_questions: " \n\t".to_string(),
            ..MarkdownSections::default()
        };
        assert!(blank.is_empty());
        let filled = MarkdownSections {
            important_facts: "x".to_string(),
            ..MarkdownSections::default()
        };
        assert!(!filled.is_empty());
    }

    #[test]
    fn markdown_sections_deserialize_missing_fields_as_empty() {
        let parsed: MarkdownSections = serde_json::from_str(r#"{"open_issues":"- Riesgo"}"#).unwrap();
        assert_eq!(parsed.open_issues, "- Riesgo");
        assert!(parsed.executive_summary.is_empty());
        assert!(!parsed.is_empty());
    }

    // ----- write_atomic_in_place -----

    #[test]
    fn atomic_write_creates_a_new_file_with_exact_content() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("Reunión - nueva.md");
        write_atomic_in_place(&path, "# Hola\n\nñandú ✓\r\n").unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "# Hola\n\nñandú ✓\r\n");
    }

    #[test]
    fn atomic_write_replaces_existing_content_completely() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nota.md");
        std::fs::write(&path, "contenido anterior muy largo ".repeat(50)).unwrap();
        write_atomic_in_place(&path, "nuevo").unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "nuevo");
    }

    #[test]
    fn atomic_write_leaves_no_temporary_files_behind() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nota.md");
        std::fs::write(&path, "a").unwrap();
        write_atomic_in_place(&path, "b").unwrap();
        let names: Vec<String> = std::fs::read_dir(dir.path())
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(names, vec!["nota.md".to_string()]);
    }

    #[test]
    fn atomic_write_accepts_empty_content() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("vacia.md");
        std::fs::write(&path, "algo").unwrap();
        write_atomic_in_place(&path, "").unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "");
    }

    #[test]
    fn atomic_write_fails_when_parent_directory_is_missing_and_creates_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("no-such-dir").join("nota.md");
        let err = write_atomic_in_place(&path, "x").unwrap_err();
        assert!(matches!(err, CoreError::Io(_)), "{err:?}");
        assert!(!path.exists());
    }

    #[test]
    fn atomic_write_fails_when_target_is_a_directory_and_keeps_the_directory() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("carpeta");
        std::fs::create_dir(&target).unwrap();
        let err = write_atomic_in_place(&target, "x").unwrap_err();
        assert!(matches!(err, CoreError::Io(_)), "{err:?}");
        assert!(target.is_dir());
    }

    #[cfg(unix)]
    #[test]
    fn atomic_write_keeps_existing_file_permissions() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nota.md");
        std::fs::write(&path, "a").unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).unwrap();
        write_atomic_in_place(&path, "b").unwrap();
        let mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o644);
    }
}
