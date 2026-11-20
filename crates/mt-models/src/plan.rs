//! Format inspection and download planning for MLX and GGUF repositories.

use std::collections::BTreeMap;

use crate::error::ModelHubError;
use crate::hub::{RepoInfo, Sibling};
use crate::reference::RepoRef;

/// Model layout families this app installs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub enum ModelFormat {
    /// `.safetensors` weights with `config.json` (Apple Silicon MLX runtime).
    Mlx,
    /// Single-file or sharded `.gguf` weights (llama.cpp style runtimes).
    Gguf,
}

/// One selectable GGUF quantization.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GgufOption {
    /// Name to pass back as the selection; first shard for sharded models.
    pub filename: String,
    /// Quantization tag such as `Q4_K_M`, or `desconocida`.
    pub quant: String,
    /// Total bytes across all shards.
    pub size_bytes: u64,
    /// Published SHA-256 when the model is a single file.
    pub sha256: Option<String>,
}

/// One file to fetch.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PlannedFile {
    /// Flat, safe filename; the same name is used locally.
    pub name: String,
    /// Expected size in bytes; `0` when unknown.
    pub size_bytes: u64,
    /// Expected SHA-256 when published.
    pub sha256: Option<String>,
}

/// Ordered download plan for one model.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelPlan {
    pub repo: RepoRef,
    pub format: ModelFormat,
    /// Install folder name under the models directory.
    pub folder: String,
    /// Selected quantization for GGUF plans.
    pub quant: Option<String>,
    /// Files in download order. For MLX, `config.json` is always last.
    pub files: Vec<PlannedFile>,
    pub total_bytes: u64,
}

impl ModelPlan {
    /// Builds a plan for `reference`. `gguf_selection` picks a GGUF file and is only valid for GGUF repositories.
    pub fn build(
        info: &RepoInfo,
        reference: &RepoRef,
        gguf_selection: Option<&str>,
    ) -> Result<ModelPlan, ModelHubError> {
        match inspect_repo_format(info)? {
            ModelFormat::Mlx if gguf_selection.is_some() => Err(unsupported(
                "la selección de GGUF solo aplica a repositorios GGUF",
            )),
            ModelFormat::Mlx => Self::for_mlx(info, reference),
            ModelFormat::Gguf => Self::for_gguf(info, reference, gguf_selection),
        }
    }

    /// MLX plan: top-level `.safetensors`, tokenizer files, `config.json` last.
    pub fn for_mlx(info: &RepoInfo, reference: &RepoRef) -> Result<ModelPlan, ModelHubError> {
        let mut weights = Vec::new();
        let mut support = Vec::new();
        let mut config = None;
        for sibling in top_level_files(info) {
            let name = sibling.rfilename.as_str();
            match name {
                "config.json" => config = Some(sibling),
                _ if !is_mlx_file(name) => {}
                _ if name.ends_with(".safetensors") => weights.push(sibling),
                _ => support.push(sibling),
            }
        }
        let config = config.ok_or_else(|| unsupported("falta config.json en la raíz del repositorio"))?;
        if weights.is_empty() {
            return Err(unsupported("no hay pesos .safetensors en la raíz del repositorio"));
        }
        weights.sort_by(|a, b| a.rfilename.cmp(&b.rfilename));
        support.sort_by(|a, b| a.rfilename.cmp(&b.rfilename));
        let files = weights
            .into_iter()
            .chain(support)
            .chain(std::iter::once(config))
            .map(planned_file)
            .collect();
        Ok(plan_from_files(reference, ModelFormat::Mlx, None, files))
    }

    /// GGUF plan. With `None`, prefers Q4_K_M, Q4_K_S, Q5_K_M, Q8_0, or the only option.
    pub fn for_gguf(
        info: &RepoInfo,
        reference: &RepoRef,
        selected: Option<&str>,
    ) -> Result<ModelPlan, ModelHubError> {
        let mut groups = gguf_groups(info);
        let index = match selected {
            Some(name) => groups
                .iter()
                .position(|group| group.option.filename == name)
                .ok_or_else(|| unsupported(&format!("no hay un GGUF completo llamado {name}")))?,
            None => PREFERRED_QUANTS
                .iter()
                .find_map(|quant| groups.iter().position(|group| group.option.quant == *quant))
                .unwrap_or(0),
        };
        if index >= groups.len() {
            return Err(unsupported("no hay archivos GGUF completos en la raíz del repositorio"));
        }
        let group = groups.swap_remove(index);
        Ok(plan_from_files(reference, ModelFormat::Gguf, Some(group.option.quant), group.files))
    }
}

/// Decides the model format from the top-level files of a repository.
pub fn inspect_repo_format(info: &RepoInfo) -> Result<ModelFormat, ModelHubError> {
    let mut has_weights = false;
    let mut has_config = false;
    let mut has_gguf = false;
    for sibling in top_level_files(info) {
        let name = sibling.rfilename.as_str();
        has_weights |= name.ends_with(".safetensors");
        has_config |= name == "config.json";
        has_gguf |= name.ends_with(".gguf");
    }
    if has_weights && has_config {
        Ok(ModelFormat::Mlx)
    } else if has_gguf {
        Ok(ModelFormat::Gguf)
    } else {
        Err(unsupported(
            "no hay pesos .safetensors con config.json ni archivos .gguf en la raíz del repositorio",
        ))
    }
}

/// Lists the selectable GGUF quantizations; incomplete shard sets are skipped.
pub fn list_gguf_options(info: &RepoInfo) -> Vec<GgufOption> {
    gguf_groups(info).into_iter().map(|group| group.option).collect()
}

/// Preferred quantizations when the caller does not choose one, best first.
const PREFERRED_QUANTS: [&str; 4] = ["Q4_K_M", "Q4_K_S", "Q5_K_M", "Q8_0"];
/// Quantization tag used when the filename carries none.
const UNKNOWN_QUANT: &str = "desconocida";

/// One selectable GGUF option together with the files that install it.
struct GgufGroup {
    option: GgufOption,
    files: Vec<PlannedFile>,
}

fn unsupported(reason: &str) -> ModelHubError {
    ModelHubError::UnsupportedFormat(reason.to_string())
}

/// Top-level files with a flat, safe name. Nested and hidden files never take part in planning.
fn top_level_files(info: &RepoInfo) -> impl Iterator<Item = &Sibling> {
    info.siblings.iter().filter(|sibling| is_plain_name(&sibling.rfilename))
}

fn is_plain_name(name: &str) -> bool {
    !name.is_empty()
        && !name.starts_with('.')
        && !name.contains(['/', '\\'])
        && !name.contains("..")
}

/// True for files an MLX install needs: allowed extension, flat, visible, and not the README.
fn is_mlx_file(name: &str) -> bool {
    const EXTENSIONS: [&str; 6] = ["safetensors", "json", "jinja", "txt", "model", "tiktoken"];
    name != "README.md"
        && is_plain_name(name)
        && name
            .rsplit_once('.')
            .is_some_and(|(_, ext)| EXTENSIONS.contains(&ext))
}

fn planned_file(sibling: &Sibling) -> PlannedFile {
    PlannedFile {
        name: sibling.rfilename.clone(),
        size_bytes: sibling.size_bytes(),
        sha256: sibling.sha256().map(str::to_string),
    }
}

fn plan_from_files(
    reference: &RepoRef,
    format: ModelFormat,
    quant: Option<String>,
    files: Vec<PlannedFile>,
) -> ModelPlan {
    let total_bytes = files.iter().map(|file| file.size_bytes).sum::<u64>();
    ModelPlan {
        repo: reference.clone(),
        format,
        folder: reference.folder_stem(),
        quant,
        files,
        total_bytes,
    }
}

/// Every GGUF option in the repository, sorted by filename. Incomplete shard sets are dropped.
fn gguf_groups(info: &RepoInfo) -> Vec<GgufGroup> {
    let mut groups = Vec::new();
    let mut shards: BTreeMap<&str, Vec<(u32, u32, &Sibling)>> = BTreeMap::new();
    for sibling in top_level_files(info) {
        let name = sibling.rfilename.as_str();
        if !name.ends_with(".gguf") {
            continue;
        }
        match parse_shard(name) {
            Some((base, index, total)) => {
                shards.entry(base).or_default().push((index, total, sibling));
            }
            None => groups.push(GgufGroup {
                option: GgufOption {
                    filename: name.to_string(),
                    quant: quant_tag(name.strip_suffix(".gguf").unwrap_or(name)),
                    size_bytes: sibling.size_bytes(),
                    sha256: sibling.sha256().map(str::to_string),
                },
                files: vec![planned_file(sibling)],
            }),
        }
    }
    for (base, mut parts) in shards {
        parts.sort_by_key(|(index, _, _)| *index);
        let total = parts[0].1;
        let complete = parts.len() == total as usize
            && parts.iter().enumerate().all(|(position, (index, part_total, _))| {
                *index as usize == position + 1 && *part_total == total
            });
        if !complete {
            continue;
        }
        let files: Vec<PlannedFile> = parts.iter().map(|(_, _, s)| planned_file(s)).collect();
        groups.push(GgufGroup {
            option: GgufOption {
                filename: files[0].name.clone(),
                quant: quant_tag(base),
                size_bytes: files.iter().map(|file| file.size_bytes).sum::<u64>(),
                sha256: None,
            },
            files,
        });
    }
    groups.sort_by(|a, b| a.option.filename.cmp(&b.option.filename));
    groups
}

/// Splits `<base>-NNNNN-of-NNNNN.gguf` into `(base, index, total)`.
fn parse_shard(name: &str) -> Option<(&str, u32, u32)> {
    let stem = name.strip_suffix(".gguf")?;
    let (head, total) = stem.rsplit_once("-of-")?;
    let (base, index) = head.rsplit_once('-')?;
    let (index, total) = (parse_number(index)?, parse_number(total)?);
    (!base.is_empty() && (1..=total).contains(&index)).then_some((base, index, total))
}

fn parse_number(text: &str) -> Option<u32> {
    if text.is_empty() || !text.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    text.parse().ok()
}

/// Quantization tag from the last filename token that looks like one (`Q4_K_M`, `IQ4_XS`, `F16`).
fn quant_tag(stem: &str) -> String {
    stem.split(['-', '.'])
        .rev()
        .find(|token| is_quant_token(token))
        .map_or_else(|| UNKNOWN_QUANT.to_string(), str::to_ascii_uppercase)
}

fn is_quant_token(token: &str) -> bool {
    let upper = token.to_ascii_uppercase();
    if matches!(upper.as_str(), "F16" | "F32" | "BF16") {
        return true;
    }
    let body = upper.strip_prefix("IQ").or_else(|| upper.strip_prefix('Q'));
    body.is_some_and(|body| {
        body.starts_with(|c: char| c.is_ascii_digit())
            && body.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_')
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn info(files: &[(&str, u64)]) -> RepoInfo {
        RepoInfo {
            repo_id: "org/model".into(),
            sha: None,
            gated: serde_json::Value::Bool(false),
            tags: Vec::new(),
            siblings: files
                .iter()
                .map(|(name, size)| Sibling { rfilename: (*name).into(), size: Some(*size), lfs: None })
                .collect(),
        }
    }

    fn reference() -> RepoRef {
        RepoRef { repo_id: "org/model".into(), revision: "main".into() }
    }

    fn names(plan: &ModelPlan) -> Vec<&str> {
        plan.files.iter().map(|file| file.name.as_str()).collect()
    }

    #[test]
    fn inspect_detects_mlx_and_gguf_and_rejects_nested_or_partial_layouts() {
        let mlx = info(&[("model.safetensors", 1), ("config.json", 1)]);
        assert_eq!(inspect_repo_format(&mlx).unwrap(), ModelFormat::Mlx);
        let gguf = info(&[("model-Q4_K_M.gguf", 1)]);
        assert_eq!(inspect_repo_format(&gguf).unwrap(), ModelFormat::Gguf);
        for files in [
            &[("model.safetensors", 1)][..],
            &[("sub/model.safetensors", 1), ("sub/config.json", 1)][..],
            &[("README.md", 1)][..],
        ] {
            let err = inspect_repo_format(&info(files)).unwrap_err();
            assert!(matches!(err, ModelHubError::UnsupportedFormat(_)), "{err}");
        }
    }

    #[test]
    fn gguf_options_group_complete_shards_and_skip_incomplete_sets() {
        let repo = info(&[
            ("Llama-Q4_K_M.gguf", 10),
            ("big-Q8_0-00001-of-00002.gguf", 5),
            ("big-Q8_0-00002-of-00002.gguf", 7),
            ("broken-Q5_K_M-00001-of-00003.gguf", 1),
            ("broken-Q5_K_M-00003-of-00003.gguf", 1),
            ("mystery.gguf", 2),
            ("nested/Q2_K.gguf", 9),
        ]);
        let summary: Vec<(String, String, u64)> = list_gguf_options(&repo)
            .into_iter()
            .map(|option| (option.filename, option.quant, option.size_bytes))
            .collect();
        assert_eq!(
            summary,
            vec![
                ("Llama-Q4_K_M.gguf".into(), "Q4_K_M".into(), 10),
                ("big-Q8_0-00001-of-00002.gguf".into(), "Q8_0".into(), 12),
                ("mystery.gguf".into(), "desconocida".into(), 2),
            ]
        );
    }

    #[test]
    fn mlx_plan_keeps_flat_allowed_files_with_weights_first_and_config_last() {
        let repo = info(&[
            ("config.json", 10),
            ("tokenizer.json", 3),
            ("model.safetensors", 100),
            ("README.md", 1),
            ("image.png", 5),
            ("sub/extra.json", 2),
            (".hidden.json", 1),
        ]);
        let plan = ModelPlan::build(&repo, &reference(), None).unwrap();
        assert_eq!(plan.format, ModelFormat::Mlx);
        assert_eq!(plan.folder, "org--model");
        assert_eq!(names(&plan), ["model.safetensors", "tokenizer.json", "config.json"]);
        assert_eq!(plan.total_bytes, 113);
    }

    #[test]
    fn gguf_plan_prefers_known_quants_and_honours_explicit_selection() {
        let repo = info(&[
            ("m-Q8_0.gguf", 8),
            ("m-Q2_K.gguf", 2),
            ("m-Q5_K_M-00001-of-00002.gguf", 3),
            ("m-Q5_K_M-00002-of-00002.gguf", 4),
        ]);
        let auto = ModelPlan::for_gguf(&repo, &reference(), None).unwrap();
        assert_eq!(auto.quant.as_deref(), Some("Q5_K_M"));
        assert_eq!(
            names(&auto),
            ["m-Q5_K_M-00001-of-00002.gguf", "m-Q5_K_M-00002-of-00002.gguf"]
        );
        assert_eq!(auto.total_bytes, 7);

        let explicit = ModelPlan::build(&repo, &reference(), Some("m-Q8_0.gguf")).unwrap();
        assert_eq!(explicit.quant.as_deref(), Some("Q8_0"));
        assert_eq!(names(&explicit), ["m-Q8_0.gguf"]);

        let err = ModelPlan::for_gguf(&repo, &reference(), Some("m-Q5_K_M-00002-of-00002.gguf"))
            .unwrap_err();
        assert!(matches!(err, ModelHubError::UnsupportedFormat(_)), "{err}");
    }
}
