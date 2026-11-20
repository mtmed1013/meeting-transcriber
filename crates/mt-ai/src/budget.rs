//! Conservative token estimation and transcript chunking for bounded prompts.

/// Conservative token estimate: one token per three UTF-8 bytes, rounded up, plus one.
///
/// Bytes rather than chars are counted, so multi-byte text (accents, emoji) is never
/// under-estimated.
pub fn estimate_tokens(text: &str) -> usize {
    text.len() / 3 + 1
}

/// Sum of [`estimate_tokens`] over every message content.
pub fn estimate_messages_tokens<'a>(contents: impl IntoIterator<Item = &'a str>) -> usize {
    contents.into_iter().map(estimate_tokens).sum()
}

/// Largest byte length whose [`estimate_tokens`] is at most `tokens`.
pub(crate) fn max_bytes_for_tokens(tokens: usize) -> usize {
    // len / 3 + 1 <= tokens  <=>  len <= 3 * (tokens - 1) + 2
    match tokens.checked_sub(1) {
        None => 0,
        Some(full) => full.saturating_mul(3).saturating_add(2),
    }
}

/// Byte offset at or before `max` that lands on a char boundary. Returns the first character's
/// length when even that character is longer than `max`, so splitting always makes progress.
fn char_split_point(text: &str, max: usize) -> usize {
    let mut end = max.min(text.len());
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    if end == 0 {
        text.chars().next().map_or(0, char::len_utf8)
    } else {
        end
    }
}

/// Splits `transcript` into chunks whose [`estimate_tokens`] is at most `max_chunk_tokens`.
///
/// Lines (keeping their line endings) are packed greedily. When a chunk must end, it ends after
/// the last blank line (paragraph boundary) if there is one. A line that alone exceeds the limit
/// is split at char boundaries. Concatenating the chunks reproduces `transcript` exactly.
/// A single character larger than the limit is emitted whole, because it cannot be split.
pub fn chunk_transcript(transcript: &str, max_chunk_tokens: usize) -> Vec<String> {
    let limit = max_bytes_for_tokens(max_chunk_tokens).max(1);
    let mut chunks = Vec::new();
    let mut current = String::new();
    // Byte length of `current` right after its last blank line.
    let mut paragraph_break: Option<usize> = None;

    for line in transcript.split_inclusive('\n') {
        if line.len() > limit {
            if !current.is_empty() {
                chunks.push(std::mem::take(&mut current));
            }
            paragraph_break = None;
            let mut rest = line;
            while rest.len() > limit {
                let cut = char_split_point(rest, limit);
                chunks.push(rest[..cut].to_string());
                rest = &rest[cut..];
            }
            current.push_str(rest);
        } else {
            while current.len() + line.len() > limit && !current.is_empty() {
                match paragraph_break.take() {
                    Some(at) => {
                        let tail = current.split_off(at);
                        chunks.push(std::mem::replace(&mut current, tail));
                    }
                    None => chunks.push(std::mem::take(&mut current)),
                }
            }
            current.push_str(line);
        }
        if line.trim().is_empty() {
            paragraph_break = Some(current.len());
        }
    }

    if !current.is_empty() {
        chunks.push(current);
    }
    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimate_is_conservative_and_byte_based() {
        assert_eq!(estimate_tokens(""), 1);
        assert_eq!(estimate_tokens("abc"), 2);
        assert_eq!(estimate_tokens("abcdefghi"), 4);
        // "ñ" is one char but two bytes: the estimate must not undercount it.
        assert_eq!(estimate_tokens("ññ"), 2);
    }

    #[test]
    fn message_estimate_sums_each_content() {
        let total = estimate_messages_tokens(["abc", "abcdefghi"]);
        assert_eq!(total, estimate_tokens("abc") + estimate_tokens("abcdefghi"));
        assert_eq!(estimate_messages_tokens(std::iter::empty()), 0);
    }

    #[test]
    fn short_transcript_is_one_chunk_unchanged() {
        assert_eq!(chunk_transcript("hola\nadiós\n", 100), vec!["hola\nadiós\n".to_string()]);
    }

    #[test]
    fn empty_transcript_has_no_chunks() {
        assert!(chunk_transcript("", 10).is_empty());
    }

    #[test]
    fn chunks_fit_the_limit_split_on_lines_and_reassemble_exactly() {
        let transcript: String = (0..200)
            .map(|i| format!("Línea {i}: hablamos del punto {i} con detalle.\n"))
            .collect();
        let chunks = chunk_transcript(&transcript, 100);
        assert!(chunks.len() > 1);
        for chunk in &chunks {
            assert!(estimate_tokens(chunk) <= 100, "{chunk:?}");
        }
        for chunk in &chunks {
            assert!(chunk.ends_with('\n'), "{chunk:?}");
        }
        assert_eq!(chunks.concat(), transcript);
    }

    #[test]
    fn prefers_paragraph_boundaries_when_a_chunk_must_end() {
        // Limit for 31 tokens is 92 bytes; three 41-byte paragraphs with blank lines overflow it.
        let p1 = format!("{}\n", "a".repeat(40));
        let p2 = format!("{}\n", "b".repeat(40));
        let p3 = format!("{}\n", "c".repeat(40));
        let transcript = format!("{p1}\n{p2}\n{p3}");
        let chunks = chunk_transcript(&transcript, 31);
        assert_eq!(chunks, vec![format!("{p1}\n{p2}\n"), p3]);
    }

    #[test]
    fn oversized_line_falls_back_to_char_boundary_split() {
        let line = format!("{}\n", "ñ".repeat(100));
        let chunks = chunk_transcript(&line, 5);
        assert!(chunks.len() > 1);
        for chunk in &chunks {
            assert!(estimate_tokens(chunk) <= 5, "{chunk:?}");
        }
        assert_eq!(chunks.concat(), line);
    }

    #[test]
    fn oversized_emoji_is_never_split_inside_a_char() {
        let text = "🧠".repeat(10);
        let chunks = chunk_transcript(&text, 1);
        assert_eq!(chunks.len(), 10);
        assert_eq!(chunks.concat(), text);
    }

    #[test]
    fn zero_limit_still_makes_progress() {
        let chunks = chunk_transcript("abc", 0);
        assert_eq!(chunks.concat(), "abc");
    }
}
