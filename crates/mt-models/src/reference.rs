//! Parsing of user-supplied Hugging Face model references.

use crate::error::ModelHubError;

/// Revision used when the reference does not name one.
pub const DEFAULT_REVISION: &str = "main";

/// A pinned Hugging Face repository reference.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RepoRef {
    /// `org/model`.
    pub repo_id: String,
    /// Branch, tag or commit. May contain `/` (for example `refs/pr/1`).
    pub revision: String,
}

impl RepoRef {
    /// Filesystem-safe folder stem for this repository, for example `org--model`.
    pub fn folder_stem(&self) -> String {
        self.repo_id.replace('/', "--")
    }
}

/// Parses `org/model`, `https://huggingface.co/org/model` or
/// `https://huggingface.co/org/model/tree/<revision>`.
pub fn parse_reference(input: &str) -> Result<RepoRef, ModelHubError> {
    const URL_PREFIX: &str = "https://huggingface.co/";

    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Err(invalid("la referencia está vacía"));
    }
    if trimmed.contains(['\\', '?', '#']) || trimmed.contains("..") {
        return Err(invalid("contiene caracteres o secuencias no permitidas"));
    }

    let (path, from_url) = match trimmed.strip_prefix(URL_PREFIX) {
        Some(rest) => (rest, true),
        None if trimmed.contains(':') => {
            return Err(invalid("solo se admiten URLs de huggingface.co"));
        }
        None => (trimmed, false),
    };

    let segments: Vec<&str> = path.trim_end_matches('/').split('/').collect();
    let (org, model, revision) = match segments.as_slice() {
        [org, model] => (*org, *model, DEFAULT_REVISION.to_string()),
        [org, model, kind, rest @ ..]
            if from_url && (*kind == "tree" || *kind == "resolve") && !rest.is_empty() =>
        {
            (*org, *model, rest.join("/"))
        }
        _ => return Err(invalid("se esperaba org/model o una URL de huggingface.co/org/model")),
    };

    if !is_name_segment(org) || !is_name_segment(model) {
        return Err(invalid("organización o modelo no válidos"));
    }
    if !revision.split('/').all(is_name_segment) {
        return Err(invalid("revisión no válida"));
    }

    Ok(RepoRef { repo_id: format!("{org}/{model}"), revision })
}

fn invalid(reason: &str) -> ModelHubError {
    ModelHubError::InvalidReference(reason.to_string())
}

/// Non-empty path segment made of ASCII letters, digits and `-_.+`, never `.`.
fn is_name_segment(segment: &str) -> bool {
    !segment.is_empty()
        && segment != "."
        && segment
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'+'))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_invalid(input: &str) {
        let err = parse_reference(input).expect_err(input);
        assert!(matches!(err, ModelHubError::InvalidReference(_)), "{input:?} -> {err}");
    }

    #[test]
    fn parses_shorthand_with_default_revision() {
        let reference = parse_reference("  org/model ").unwrap();
        assert_eq!(
            reference,
            RepoRef { repo_id: "org/model".into(), revision: DEFAULT_REVISION.into() }
        );
        assert_eq!(reference.folder_stem(), "org--model");
    }

    #[test]
    fn parses_hub_urls_with_optional_tree_or_resolve_revision() {
        let root = parse_reference("https://huggingface.co/org/model").unwrap();
        assert_eq!(root.repo_id, "org/model");
        assert_eq!(root.revision, "main");

        let tree = parse_reference("https://huggingface.co/org/model/tree/v1.2").unwrap();
        assert_eq!(tree.revision, "v1.2");

        let resolve = parse_reference("https://huggingface.co/org/model/resolve/refs/pr/1").unwrap();
        assert_eq!(
            resolve,
            RepoRef { repo_id: "org/model".into(), revision: "refs/pr/1".into() }
        );
    }

    #[test]
    fn rejects_empty_and_whitespace_only_input() {
        assert_invalid("");
        assert_invalid("   \t ");
    }

    #[test]
    fn rejects_traversal_and_unsafe_characters() {
        for input in [
            "org/../model",
            "../model",
            "org\\model",
            "org/model?x=1",
            "org/model#frag",
            "https://huggingface.co/org/model/tree/../../x",
            "org/mo del",
        ] {
            assert_invalid(input);
        }
    }

    #[test]
    fn rejects_missing_parts_and_foreign_hosts() {
        for input in [
            "gpt2",
            "org/",
            "/model",
            "org/model/extra",
            "https://example.com/org/model",
            "https://huggingface.co/org",
            "https://huggingface.co/org/model/blob/main/config.json",
            "https://huggingface.co/org/model/tree/",
        ] {
            assert_invalid(input);
        }
    }
}
