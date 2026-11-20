//! Transcript correction rules and spoken-word counting.

use crate::config::CorrectionRule;

/// Line the recorder writes when the speaker changes; it is not spoken content.
pub const SPEAKER_CHANGE_MARKER: &str = "— 🔄 Cambio de hablante estimado —";

/// Applies correction rules in order. Each rule sees the output of the rules before it.
///
/// Matches are non-overlapping and scanned left to right. A rule with an empty `find` is skipped.
/// `replace` is inserted literally.
pub fn apply_corrections(text: &str, rules: &[CorrectionRule]) -> String {
    rules.iter().fold(text.to_string(), |current, rule| apply_rule(&current, rule))
}

fn apply_rule(text: &str, rule: &CorrectionRule) -> String {
    if rule.find.is_empty() {
        return text.to_string();
    }
    let pattern: Vec<char> = rule.find.chars().collect();
    let hay: Vec<(usize, char)> = text.char_indices().collect();
    let n = pattern.len();

    let mut out = String::with_capacity(text.len());
    let mut copied_to = 0; // byte offset in `text` already copied to `out`
    let mut i = 0; // index into `hay`
    while i + n <= hay.len() {
        let window = &hay[i..i + n];
        let matched = window
            .iter()
            .zip(&pattern)
            .all(|((_, ch), pat)| chars_equal(*ch, *pat, rule.case_sensitive));
        if matched && (!rule.whole_word || is_word_boundary(&hay, i, i + n)) {
            let start = hay[i].0;
            let end = hay.get(i + n).map_or(text.len(), |(offset, _)| *offset);
            out.push_str(&text[copied_to..start]);
            out.push_str(&rule.replace);
            copied_to = end;
            i += n;
        } else {
            i += 1;
        }
    }
    out.push_str(&text[copied_to..]);
    out
}

fn chars_equal(a: char, b: char, case_sensitive: bool) -> bool {
    if case_sensitive {
        a == b
    } else {
        a.to_lowercase().eq(b.to_lowercase())
    }
}

/// A match is a whole word when neither neighbouring character is a Unicode alphanumeric.
fn is_word_boundary(hay: &[(usize, char)], start: usize, end: usize) -> bool {
    let before_ok = start == 0 || !hay[start - 1].1.is_alphanumeric();
    let after_ok = end >= hay.len() || !hay[end].1.is_alphanumeric();
    before_ok && after_ok
}

/// Deterministic pre-compaction of a transcript for model analysis.
///
/// Works on a copy, so the stored original transcript is never altered. Per line, runs of spaces
/// and tabs collapse to one space and the line is trimmed; `**[hh:mm:ss]**` becomes `[hh:mm:ss]`;
/// `---` rules and the speaker-change marker lines are dropped; blank lines collapse to at most one,
/// and leading or trailing blank lines are removed. Every spoken word is kept.
pub fn compact_transcript_for_analysis(text: &str) -> String {
    let mut out: Vec<String> = Vec::new();
    for raw in text.lines() {
        let collapsed = raw
            .split([' ', '\t', '\r'])
            .filter(|token| !token.is_empty())
            .collect::<Vec<_>>()
            .join(" ");
        if collapsed == "---" || collapsed == SPEAKER_CHANGE_MARKER {
            continue;
        }
        let line = match split_timestamp_prefix(&collapsed) {
            Some((stamp, rest)) => {
                let rest = rest.trim_start();
                if rest.is_empty() {
                    format!("[{stamp}]")
                } else {
                    format!("[{stamp}] {rest}")
                }
            }
            None => collapsed,
        };
        if line.is_empty() && out.last().is_none_or(String::is_empty) {
            continue;
        }
        out.push(line);
    }
    while out.last().is_some_and(String::is_empty) {
        out.pop();
    }
    out.join("\n")
}

/// Counts spoken words, ignoring timestamps, markdown headers, the `🕒` start line, `---` rules,
/// and the speaker-change marker. Only tokens containing a Unicode alphanumeric character count.
pub fn count_spoken_words(transcript: &str) -> usize {
    transcript.lines().map(count_line_words).sum()
}

fn count_line_words(line: &str) -> usize {
    let mut text = line.trim();
    if let Some(rest) = strip_timestamp_prefix(text) {
        text = rest.trim();
    }
    if text.is_empty()
        || text.starts_with('🕒')
        || is_markdown_header(text)
        || is_horizontal_rule(text)
    {
        return 0;
    }
    let without_marker = text.replace(SPEAKER_CHANGE_MARKER, " ");
    without_marker
        .split_whitespace()
        .filter(|token| token.chars().any(char::is_alphanumeric))
        .count()
}

/// Returns the text after a leading `**[hh:mm:ss]**` timestamp, if there is one.
fn strip_timestamp_prefix(text: &str) -> Option<&str> {
    split_timestamp_prefix(text).map(|(_, rest)| rest)
}

/// Splits a leading timestamp into its stamp and the text after it. Accepts the transcript form
/// `**[hh:mm:ss]**` and the compacted form `[hh:mm:ss]`, so compaction never changes the word count.
fn split_timestamp_prefix(text: &str) -> Option<(&str, &str)> {
    let (rest, closer) = match text.strip_prefix("**[") {
        Some(rest) => (rest, "]**"),
        None => (text.strip_prefix('[')?, "]"),
    };
    let end = rest.find(closer)?;
    let stamp = &rest[..end];
    if stamp.is_empty() || !stamp.chars().all(|c| c.is_ascii_digit() || c == ':') {
        return None;
    }
    Some((stamp, &rest[end + closer.len()..]))
}

/// A markdown heading: one to six `#` followed by whitespace or the end of the line.
fn is_markdown_header(text: &str) -> bool {
    let hashes = text.chars().take_while(|c| *c == '#').count();
    (1..=6).contains(&hashes)
        && text[hashes..]
            .chars()
            .next()
            .is_none_or(char::is_whitespace)
}

fn is_horizontal_rule(text: &str) -> bool {
    text.len() >= 3 && text.chars().all(|c| c == '-')
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rule(find: &str, replace: &str, case_sensitive: bool, whole_word: bool) -> CorrectionRule {
        CorrectionRule {
            find: find.to_string(),
            replace: replace.to_string(),
            case_sensitive,
            whole_word,
        }
    }

    #[test]
    fn case_sensitive_exact_replacement_only_touches_identical_case() {
        let rules = [rule("Teh", "The", true, false)];
        assert_eq!(apply_corrections("Teh cat teh", &rules), "The cat teh");
    }

    #[test]
    fn default_matching_ignores_case() {
        let rules = [rule("api", "API", false, false)];
        assert_eq!(
            apply_corrections("Api and API and apiary", &rules),
            "API and API and APIary"
        );
    }

    #[test]
    fn whole_word_skips_matches_inside_longer_words() {
        let rules = [rule("api", "API", false, true)];
        assert_eq!(
            apply_corrections("api apiary (api) xapi", &rules),
            "API apiary (API) xapi"
        );
    }

    #[test]
    fn whole_word_boundaries_are_unicode_aware() {
        let rules = [rule("año", "ANIO", true, true)];
        assert_eq!(apply_corrections("el año, añoso año", &rules), "el ANIO, añoso ANIO");
        assert_eq!(apply_corrections("éaño año", &rules), "éaño ANIO");
    }

    #[test]
    fn case_insensitive_matching_handles_non_ascii_letters() {
        let rules = [rule("AÑO", "ANIO", false, false)];
        assert_eq!(apply_corrections("un año", &rules), "un ANIO");
    }

    #[test]
    fn digits_are_word_characters_but_underscores_are_boundaries() {
        let digits = [rule("v", "V", true, true)];
        assert_eq!(apply_corrections("v2 v", &digits), "v2 V");

        let underscore = [rule("api", "API", true, true)];
        assert_eq!(apply_corrections("api_key", &underscore), "API_key");
    }

    #[test]
    fn punctuation_adjacent_occurrences_count_as_whole_words() {
        let rules = [rule("api", "API", false, true)];
        assert_eq!(apply_corrections("Use the api.", &rules), "Use the API.");
    }

    #[test]
    fn matches_do_not_overlap_and_replacements_are_not_rescanned() {
        assert_eq!(apply_corrections("aaaa", &[rule("aa", "b", true, false)]), "bb");
        assert_eq!(
            apply_corrections("cat cat", &[rule("cat", "cats", true, false)]),
            "cats cats"
        );
    }

    #[test]
    fn rules_apply_in_order_and_later_rules_see_earlier_output() {
        let rules = [rule("a", "b", true, false), rule("b", "c", true, false)];
        assert_eq!(apply_corrections("a", &rules), "c");
    }

    #[test]
    fn empty_find_is_ignored_and_no_rules_leave_text_unchanged() {
        assert_eq!(apply_corrections("texto", &[rule("", "X", true, false)]), "texto");
        assert_eq!(apply_corrections("texto", &[]), "texto");
    }

    #[test]
    fn replacement_text_is_literal() {
        let rules = [rule("dinero", "$1 [x]", true, false)];
        assert_eq!(apply_corrections("dinero", &rules), "$1 [x]");
    }

    #[test]
    fn counts_words_in_a_recorded_transcript_and_skips_structure() {
        let transcript = "\
# 🧠 Reunión

🕒 Inicio: 10:00

---

**[10:00:01]**
Hola equipo, revisemos el plan hoy.

**[10:00:31]**
— 🔄 Cambio de hablante estimado —
Sí, de acuerdo.

";
        assert_eq!(count_spoken_words(transcript), 9);
    }

    #[test]
    fn timestamps_alone_or_inline_are_not_words() {
        assert_eq!(count_spoken_words("**[12:34:56]**\n"), 0);
        assert_eq!(count_spoken_words("**[12:34:56]** hola mundo"), 2);
    }

    #[test]
    fn compacted_bracket_timestamps_are_not_words_but_plain_brackets_are() {
        assert_eq!(count_spoken_words("[10:00:01]\nhola"), 1);
        assert_eq!(count_spoken_words("[00:05] hola mundo"), 2);
        assert_eq!(count_spoken_words("[nota] hola"), 2);
    }

    #[test]
    fn markdown_headers_are_skipped_but_hashtags_are_words() {
        assert_eq!(count_spoken_words("## Resumen\n# Título\n"), 0);
        assert_eq!(count_spoken_words("#tema importante"), 2);
    }

    #[test]
    fn start_line_and_rules_are_skipped_only_at_line_start() {
        assert_eq!(count_spoken_words("🕒 Inicio: 10:00"), 0);
        assert_eq!(count_spoken_words("---"), 0);
        assert_eq!(count_spoken_words("dijo --- algo"), 2);
        assert_eq!(count_spoken_words("🕒 Inicio: 10:00 dijo"), 0);
    }

    #[test]
    fn speaker_change_marker_text_is_not_counted() {
        assert_eq!(count_spoken_words(SPEAKER_CHANGE_MARKER), 0);
        assert_eq!(count_spoken_words(&format!("{SPEAKER_CHANGE_MARKER} hola")), 1);
        assert_eq!(count_spoken_words(&format!("hola\n{SPEAKER_CHANGE_MARKER}\nadiós")), 2);
    }

    #[test]
    fn punctuation_only_tokens_are_not_words() {
        assert_eq!(count_spoken_words("— … ¿ ? !"), 0);
        assert_eq!(count_spoken_words("¿Qué tal?"), 2);
        assert_eq!(count_spoken_words("2,000 registros"), 2);
    }

    #[test]
    fn empty_and_whitespace_transcripts_have_no_words() {
        assert_eq!(count_spoken_words(""), 0);
        assert_eq!(count_spoken_words("  \n\t \n"), 0);
    }

    #[test]
    fn compact_collapses_space_and_tab_runs_and_trims_each_line() {
        let text = "  hola   mundo\t\tque  tal \t\n\tsegunda\t línea  ";
        assert_eq!(compact_transcript_for_analysis(text), "hola mundo que tal\nsegunda línea");
    }

    #[test]
    fn compact_turns_bold_timestamps_into_bare_brackets() {
        let text = "**[10:00:01]**\nHola.\n**[00:05]**   Sí, de acuerdo.";
        assert_eq!(
            compact_transcript_for_analysis(text),
            "[10:00:01]\nHola.\n[00:05] Sí, de acuerdo."
        );
    }

    #[test]
    fn compact_drops_rules_and_speaker_markers_but_keeps_the_start_line() {
        let text = format!(
            "🕒 Inicio: 10:00\n---\n**[10:00:01]**\nHola.\n{SPEAKER_CHANGE_MARKER}\nAdiós.\n---"
        );
        assert_eq!(
            compact_transcript_for_analysis(&text),
            "🕒 Inicio: 10:00\n[10:00:01]\nHola.\nAdiós."
        );
    }

    #[test]
    fn compact_collapses_blank_line_runs_to_one_blank_line() {
        assert_eq!(compact_transcript_for_analysis("a\n\n\n\n  \nb"), "a\n\nb");
        assert_eq!(compact_transcript_for_analysis("a\n\n---\n\nb"), "a\n\nb");
    }

    #[test]
    fn compact_trims_leading_and_trailing_blank_lines() {
        assert_eq!(compact_transcript_for_analysis("\n\n  hola\n\n"), "hola");
        assert_eq!(compact_transcript_for_analysis("  \n\t\n"), "");
        assert_eq!(compact_transcript_for_analysis(""), "");
    }

    #[test]
    fn compact_is_idempotent() {
        let once = compact_transcript_for_analysis("**[00:00:01]**  a\n\n\n---\n\nb ");
        assert_eq!(compact_transcript_for_analysis(&once), once);
    }

    #[test]
    fn compact_keeps_every_spoken_word_and_the_word_count() {
        let transcript = "# 🧠 Reunión\n\n🕒 Inicio: 10:00\n\n---\n\n**[10:00:01]**   Hola equipo,\trevisemos el plan.\n\n**[10:00:31]**\n— 🔄 Cambio de hablante estimado —\nSí,  de acuerdo.\n**[00:01:05]** Mañana lo cerramos.\n---\n";
        let compacted = compact_transcript_for_analysis(transcript);
        assert_eq!(count_spoken_words(transcript), 11);
        assert_eq!(count_spoken_words(&compacted), 11);
        assert!(compacted.contains("Hola equipo, revisemos el plan."), "{compacted}");
        assert!(compacted.contains("Mañana lo cerramos."), "{compacted}");
    }
}
