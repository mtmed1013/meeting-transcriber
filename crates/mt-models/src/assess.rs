//! Fit assessment of a model against the current machine, in Spanish for the UI.

use crate::hardware::MachineInfo;
use crate::hub::RepoInfo;
use crate::plan::{ModelFormat, ModelPlan, inspect_repo_format, list_gguf_options};
use crate::profile::{SuggestedModelConfig, extract_suggested_config};
use crate::reference::RepoRef;

/// Context length assumed when the caller does not choose one.
pub const DEFAULT_CONTEXT_TOKENS: u32 = 4096;

const BYTES_PER_GIB: u64 = 1024 * 1024 * 1024;
/// Memory kept free for the runtime itself, on top of weights and KV cache.
const RUNTIME_RESERVE_BYTES: u64 = BYTES_PER_GIB * 3 / 2;
/// Keys + values for one head-dimension element: two tensors of 2 bytes each (fp16).
const KV_BYTES_PER_HEAD_DIM: u64 = 4;
/// Share of RAM (percent) below which a model fits with a comfortable margin.
const FITS_PERCENT: u64 = 65;
/// Share of RAM (percent) below which a model fits, but tightly.
const TIGHT_PERCENT: u64 = 80;
/// Free disk must cover the download times this factor (percent).
const DISK_MARGIN_PERCENT: u64 = 105;
const NO_FORMAT_NOTE: &str =
    "No se puede instalar: el repositorio no tiene pesos MLX (.safetensors con config.json) ni archivos GGUF.";
const NO_TOOL_NOTE: &str =
    "No se puede instalar: este modelo no declara soporte para herramientas (tool calling) en su plantilla de chat o metadatos de Hugging Face.";
/// Hub tags that declare tool calling (compared case-insensitively).
const TOOL_TAGS: [&str; 3] = ["tool-use", "function-calling", "tools"];
/// Chat-template fragments that show a template renders tool definitions or tool calls.
const TOOL_MARKERS: [&str; 6] = [
    "tools",
    "tool_call",
    "<tool_call>",
    "[AVAILABLE_TOOLS]",
    "tool_calls",
    "function_call",
];
/// GGUF metadata key holding the embedded chat template.
const GGUF_CHAT_TEMPLATE_KEY: &[u8] = b"tokenizer.chat_template";

/// How well the model fits the machine's memory budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub enum Fit {
    /// Comfortable margin.
    Fits,
    /// Fits, but within the last margin of the budget.
    Tight,
    /// Does not fit the memory budget.
    TooBig,
}

/// Reasons a model cannot be installed on this machine at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub enum Blocker {
    /// The model family does not run on this OS.
    UnsupportedPlatform,
    /// The OS is supported, but this CPU architecture is not (for example Intel Mac for MLX).
    UnsupportedArchitecture,
    /// The CPU architecture is not one we can reason about.
    UnknownArchitecture,
    /// The repository has no files this app can install.
    NoCompatibleFiles,
    /// The repository does not declare tool calling in its metadata or chat template.
    NoToolSupport,
}

/// Input for [`assess`].
#[derive(Debug, Clone, Copy)]
pub struct AssessRequest<'a> {
    pub info: &'a RepoInfo,
    pub reference: &'a RepoRef,
    /// Optional `config.json` used to size the KV cache exactly.
    pub config: Option<&'a serde_json::Value>,
    /// Optional `generation_config.json`, source of suggested sampling settings.
    pub generation_config: Option<&'a serde_json::Value>,
    /// Optional `tokenizer_config.json`, scanned for a tool-calling chat template.
    pub tokenizer_config: Option<&'a serde_json::Value>,
    /// Leading bytes of a GGUF file, scanned for an embedded `tokenizer.chat_template`.
    pub gguf_header: Option<&'a [u8]>,
    pub machine: &'a MachineInfo,
    /// Explicit GGUF file; `None` lets the planner choose.
    pub gguf_filename: Option<&'a str>,
    pub context_tokens: u32,
}

/// Result of assessing one model on one machine.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Assessment {
    /// `None` only when the repository format could not be inspected.
    pub format: Option<ModelFormat>,
    pub fit: Fit,
    pub blocker: Option<Blocker>,
    pub weights_bytes: u64,
    /// Estimated KV cache for `context_tokens`.
    pub cache_bytes: u64,
    pub context_tokens: u32,
    /// Weights + cache + runtime overhead.
    pub needed_bytes: u64,
    pub ram_bytes: u64,
    /// True when free disk covers the download plus a safety margin.
    pub disk_ok: bool,
    pub disk_free_bytes: u64,
    /// True when several GGUF quantizations exist and none was chosen.
    pub needs_selection: bool,
    /// True when the repository declares tool calling (see [`detect_tool_support`]).
    pub supports_tools: bool,
    /// Suggested runtime settings (see [`extract_suggested_config`]).
    pub suggested_config: SuggestedModelConfig,
    /// Human-readable explanation in Spanish.
    pub verdict: String,
}

impl Assessment {
    /// True when the model may be installed right now.
    pub fn can_install(&self) -> bool {
        self.blocker.is_none()
            && !self.needs_selection
            && self.disk_ok
            && self.fit != Fit::TooBig
    }
}

/// Assesses one model against one machine.
pub fn assess(req: &AssessRequest<'_>) -> Assessment {
    let supports_tools = detect_tool_support(req.info, req.tokenizer_config, req.gguf_header);
    let format = match inspect_repo_format(req.info) {
        Ok(format) => format,
        Err(_) => {
            return build_assessment(
                req,
                AssessmentDraft {
                    format: None,
                    blocker: Some(Blocker::NoCompatibleFiles),
                    weights_bytes: 0,
                    evaluable: false,
                    needs_selection: false,
                    supports_tools,
                    notes: vec![NO_FORMAT_NOTE.to_string()],
                },
            );
        }
    };

    let mut notes = Vec::new();
    let mut blocker = None;
    if let Some((reason, note)) = platform_check(format, req.machine) {
        blocker = Some(reason);
        notes.push(note);
    }

    let plan = match format {
        ModelFormat::Mlx => ModelPlan::for_mlx(req.info, req.reference),
        ModelFormat::Gguf => ModelPlan::build(req.info, req.reference, req.gguf_filename),
    };
    let (weights_bytes, evaluable) = match plan {
        Ok(plan) => (plan.total_bytes, true),
        Err(err) => {
            blocker = blocker.or(Some(Blocker::NoCompatibleFiles));
            notes.push(format!("No se puede instalar: {err}."));
            (0, false)
        }
    };
    // Platform and file-layout blockers keep priority; the tool check only applies to installable models.
    if evaluable && blocker.is_none() && !supports_tools {
        blocker = Some(Blocker::NoToolSupport);
        notes.push(NO_TOOL_NOTE.to_string());
    }

    let needs_selection = format == ModelFormat::Gguf
        && req.gguf_filename.is_none()
        && list_gguf_options(req.info).len() > 1;
    build_assessment(
        req,
        AssessmentDraft {
            format: Some(format),
            blocker,
            weights_bytes,
            evaluable,
            needs_selection,
            supports_tools,
            notes,
        },
    )
}

/// True when the repository declares tool calling: a matching hub tag, a tool-calling chat template in
/// `tokenizer_config.json`, or one embedded in the GGUF header sample.
pub fn detect_tool_support(
    info: &RepoInfo,
    tokenizer_config: Option<&serde_json::Value>,
    gguf_header: Option<&[u8]>,
) -> bool {
    let tagged = info
        .tags
        .iter()
        .any(|tag| TOOL_TAGS.iter().any(|known| tag.eq_ignore_ascii_case(known)));
    tagged
        || tokenizer_config.is_some_and(|config| {
            chat_templates(config)
                .into_iter()
                .any(template_declares_tools)
        })
        || gguf_header
            .and_then(gguf_chat_template)
            .is_some_and(|bytes| template_declares_tools(&String::from_utf8_lossy(bytes)))
}

/// Chat templates of a `tokenizer_config.json`: a single string, or a list of `{ "template": ... }`.
pub(crate) fn chat_templates(config: &serde_json::Value) -> Vec<&str> {
    match config.get("chat_template") {
        Some(serde_json::Value::String(template)) => vec![template.as_str()],
        Some(serde_json::Value::Array(entries)) => entries
            .iter()
            .filter_map(|entry| entry.get("template")?.as_str())
            .collect(),
        _ => Vec::new(),
    }
}

fn template_declares_tools(template: &str) -> bool {
    TOOL_MARKERS.iter().any(|marker| template.contains(marker))
}

/// The value of `tokenizer.chat_template` in a GGUF header sample, when it is a string value.
/// The value follows the key as a little-endian `u32` type (8 = string) and a `u64` byte length.
pub(crate) fn gguf_chat_template(header: &[u8]) -> Option<&[u8]> {
    const GGUF_TYPE_STRING: u32 = 8;
    // A key is stored as a u64 length followed by its bytes, so the length prefix pins the exact key:
    // `tokenizer.chat_template.tool_use` contains the same bytes but has a different length.
    let key_len = GGUF_CHAT_TEMPLATE_KEY.len();
    let key_at = header.windows(key_len).enumerate().find_map(|(at, window)| {
        let length_prefix = at.checked_sub(8).and_then(|start| header.get(start..at));
        (window == GGUF_CHAT_TEMPLATE_KEY
            && length_prefix == Some(&(key_len as u64).to_le_bytes()[..]))
        .then_some(at)
    })?;
    let rest = &header[key_at + key_len..];
    let (kind, rest) = rest.split_first_chunk::<4>()?;
    if u32::from_le_bytes(*kind) != GGUF_TYPE_STRING {
        return None;
    }
    let (len, body) = rest.split_first_chunk::<8>()?;
    let len = usize::try_from(u64::from_le_bytes(*len)).unwrap_or(usize::MAX);
    // The sample may end inside a long template; scan what was received.
    Some(&body[..len.min(body.len())])
}

/// Platform and CPU architecture check, with the Spanish explanation when blocked.
fn platform_check(format: ModelFormat, machine: &MachineInfo) -> Option<(Blocker, String)> {
    let os = machine.os.as_str();
    let arch = machine.arch.as_str();
    match format {
        ModelFormat::Mlx if os != "macos" => Some((
            Blocker::UnsupportedPlatform,
            format!("MLX solo funciona en macOS con Apple Silicon; este sistema es {os}."),
        )),
        ModelFormat::Mlx if arch != "aarch64" => Some((
            Blocker::UnsupportedArchitecture,
            format!(
                "MLX requiere Apple Silicon (aarch64); arquitectura detectada: {}.",
                arch_label(arch)
            ),
        )),
        ModelFormat::Mlx => None,
        ModelFormat::Gguf if !matches!(os, "macos" | "windows") => Some((
            Blocker::UnsupportedPlatform,
            format!("Los modelos GGUF solo se soportan en macOS y Windows; este sistema es {os}."),
        )),
        ModelFormat::Gguf if !matches!(arch, "x86_64" | "aarch64") => Some((
            Blocker::UnknownArchitecture,
            format!(
                "No se pudo determinar la arquitectura del procesador ({}); no se puede evaluar el modelo.",
                arch_label(arch)
            ),
        )),
        ModelFormat::Gguf => None,
    }
}

fn arch_label(arch: &str) -> &str {
    if arch.is_empty() { "desconocida" } else { arch }
}

struct AssessmentDraft {
    format: Option<ModelFormat>,
    blocker: Option<Blocker>,
    weights_bytes: u64,
    evaluable: bool,
    needs_selection: bool,
    supports_tools: bool,
    notes: Vec<String>,
}

/// Computes memory, disk and verdict from the plan's download size and the machine facts.
fn build_assessment(req: &AssessRequest<'_>, draft: AssessmentDraft) -> Assessment {
    let AssessmentDraft {
        format,
        blocker,
        weights_bytes,
        evaluable,
        needs_selection,
        supports_tools,
        notes,
    } = draft;
    let machine = req.machine;
    let cache_bytes = cache_bytes(req.config, weights_bytes, req.context_tokens);
    let needed_bytes = weights_bytes
        .saturating_add(cache_bytes)
        .saturating_add(RUNTIME_RESERVE_BYTES);
    let fit = classify(needed_bytes, machine.ram_bytes);
    let disk_ok = u128::from(weights_bytes) * u128::from(DISK_MARGIN_PERCENT)
        <= u128::from(machine.disk_free_bytes) * 100;

    let mut parts = notes;
    // Without a usable plan there is no download size to explain, so only the notes are shown.
    if evaluable {
        parts.push(fit_message(fit, needed_bytes, machine));
        if !disk_ok {
            parts.push(format!(
                "Espacio en disco insuficiente: la descarga necesita {} libres y hay {}.",
                gib(weights_bytes.saturating_mul(DISK_MARGIN_PERCENT) / 100),
                gib(machine.disk_free_bytes)
            ));
        }
        if needs_selection {
            parts.push("Hay varias cuantizaciones GGUF: elige una antes de instalar.".to_string());
        }
    }

    Assessment {
        format,
        fit,
        blocker,
        weights_bytes,
        cache_bytes,
        context_tokens: req.context_tokens,
        needed_bytes,
        ram_bytes: machine.ram_bytes,
        disk_ok,
        disk_free_bytes: machine.disk_free_bytes,
        needs_selection,
        supports_tools,
        suggested_config: extract_suggested_config(
            req.config,
            req.generation_config,
            req.tokenizer_config,
            req.gguf_header,
        ),
        verdict: parts.join(" "),
    }
}

fn classify(needed_bytes: u64, ram_bytes: u64) -> Fit {
    let needed = u128::from(needed_bytes) * 100;
    let ram = u128::from(ram_bytes);
    if needed <= ram * u128::from(FITS_PERCENT) {
        Fit::Fits
    } else if needed <= ram * u128::from(TIGHT_PERCENT) {
        Fit::Tight
    } else {
        Fit::TooBig
    }
}

fn fit_message(fit: Fit, needed_bytes: u64, machine: &MachineInfo) -> String {
    if machine.ram_bytes == 0 {
        return "No se pudo detectar la RAM del equipo, así que no se puede estimar el ajuste."
            .to_string();
    }
    let needed = gib(needed_bytes);
    let ram = gib(machine.ram_bytes);
    match fit {
        Fit::Fits => format!("Cabe cómodamente: necesita {needed} de los {ram} de RAM."),
        Fit::Tight => format!("Cabe, pero con poco margen: necesita {needed} de los {ram} de RAM."),
        Fit::TooBig => format!(
            "No cabe en la memoria: necesita {needed} y el equipo tiene {ram} de RAM (el límite recomendado es el 80%)."
        ),
    }
}

fn gib(bytes: u64) -> String {
    format!("{:.1} GiB", bytes as f64 / BYTES_PER_GIB as f64)
}

/// Estimated KV cache for `context_tokens`: exact from `config.json` when it has the fields, otherwise
/// a conservative guess of one eighth of the weights per 4096 tokens of context.
fn cache_bytes(config: Option<&serde_json::Value>, weights_bytes: u64, context_tokens: u32) -> u64 {
    match config.and_then(kv_bytes_per_token) {
        Some(per_token) => per_token.saturating_mul(u64::from(context_tokens)),
        None => {
            let scaled = u128::from(weights_bytes) * u128::from(context_tokens)
                / (8 * u128::from(DEFAULT_CONTEXT_TOKENS));
            u64::try_from(scaled).unwrap_or(u64::MAX)
        }
    }
}

fn kv_bytes_per_token(config: &serde_json::Value) -> Option<u64> {
    let layers = config_count(config, "num_hidden_layers")?;
    let attention_heads = config_count(config, "num_attention_heads");
    let kv_heads = config_count(config, "num_key_value_heads").or(attention_heads)?;
    let head_dim = match config_count(config, "head_dim") {
        Some(dim) => dim,
        None => config_count(config, "hidden_size")? / attention_heads?,
    };
    Some(layers.saturating_mul(kv_heads).saturating_mul(head_dim).saturating_mul(KV_BYTES_PER_HEAD_DIM))
}

fn config_count(config: &serde_json::Value, key: &str) -> Option<u64> {
    config.get(key)?.as_u64().filter(|value| *value > 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hub::Sibling;
    use serde_json::json;

    const GIB: u64 = 1024 * 1024 * 1024;
    const MIB: u64 = 1024 * 1024;

    fn repo(files: &[(&str, u64)]) -> RepoInfo {
        RepoInfo {
            repo_id: "org/model".into(),
            sha: None,
            gated: json!(false),
            siblings: files
                .iter()
                .map(|(name, size)| Sibling { rfilename: (*name).into(), size: Some(*size), lfs: None })
                .collect(),
            tags: Vec::new(),
        }
    }

    fn with_tags(mut info: RepoInfo, tags: &[&str]) -> RepoInfo {
        info.tags = tags.iter().map(|tag| (*tag).to_string()).collect();
        info
    }

    const TOOL_TEMPLATE: &str =
        "{% if tools %}{% for tool in tools %}{{ tool }}{% endfor %}{% endif %}";

    /// One GGUF string metadata entry: length-prefixed key, type 8, length-prefixed value.
    fn gguf_string_entry(key: &str, value: &str) -> Vec<u8> {
        let mut entry = (key.len() as u64).to_le_bytes().to_vec();
        entry.extend_from_slice(key.as_bytes());
        entry.extend_from_slice(&8u32.to_le_bytes());
        entry.extend_from_slice(&(value.len() as u64).to_le_bytes());
        entry.extend_from_slice(value.as_bytes());
        entry
    }

    /// A GGUF header sample holding `tokenizer.chat_template` as a length-prefixed string value.
    fn gguf_header_with_template(template: &str) -> Vec<u8> {
        let mut header = b"GGUF\x03\x00\x00\x00junk".to_vec();
        header.extend(gguf_string_entry("tokenizer.chat_template", template));
        header
    }

    fn machine(os: &str, arch: &str, ram_gib: u64, disk_gib: u64) -> MachineInfo {
        MachineInfo {
            os: os.into(),
            arch: arch.into(),
            ram_bytes: ram_gib * GIB,
            disk_free_bytes: disk_gib * GIB,
            unified_memory: os == "macos" && arch == "aarch64",
            gpu_vram_bytes: None,
        }
    }

    fn assess_with(
        info: &RepoInfo,
        machine: &MachineInfo,
        config: Option<&serde_json::Value>,
    ) -> Assessment {
        assess_request(info, machine, config, None, None)
    }

    fn assess_request(
        info: &RepoInfo,
        machine: &MachineInfo,
        config: Option<&serde_json::Value>,
        tokenizer_config: Option<&serde_json::Value>,
        gguf_header: Option<&[u8]>,
    ) -> Assessment {
        assess_full(info, machine, config, None, tokenizer_config, gguf_header)
    }

    fn assess_full(
        info: &RepoInfo,
        machine: &MachineInfo,
        config: Option<&serde_json::Value>,
        generation_config: Option<&serde_json::Value>,
        tokenizer_config: Option<&serde_json::Value>,
        gguf_header: Option<&[u8]>,
    ) -> Assessment {
        let reference = RepoRef { repo_id: "org/model".into(), revision: "main".into() };
        assess(&AssessRequest {
            info,
            reference: &reference,
            config,
            generation_config,
            tokenizer_config,
            gguf_header,
            machine,
            gguf_filename: None,
            context_tokens: DEFAULT_CONTEXT_TOKENS,
        })
    }

    #[test]
    fn fits_when_weights_and_exact_cache_leave_a_comfortable_margin() {
        let info = with_tags(repo(&[("model-Q4_K_M.gguf", 2 * GIB)]), &["tool-use"]);
        let config = json!({ "num_hidden_layers": 2, "num_key_value_heads": 2, "head_dim": 64 });
        let result = assess_with(&info, &machine("macos", "aarch64", 16, 100), Some(&config));
        assert_eq!(result.format, Some(ModelFormat::Gguf));
        assert_eq!(result.fit, Fit::Fits, "{}", result.verdict);
        assert_eq!(result.cache_bytes, 4 * MIB);
        assert_eq!(result.needed_bytes, 2 * GIB + 4 * MIB + GIB + GIB / 2);
        assert_eq!(result.blocker, None);
        assert!(result.disk_ok);
        assert!(result.can_install(), "{}", result.verdict);
    }

    #[test]
    fn tight_when_needed_is_between_65_and_80_percent_of_ram() {
        // 4 GiB weights + 0.5 GiB fallback cache + 1.5 GiB reserve = 6 GiB on 8 GiB RAM (75%).
        let info = with_tags(repo(&[("model-Q4_K_M.gguf", 4 * GIB)]), &["tool-use"]);
        let result = assess_with(&info, &machine("windows", "x86_64", 8, 100), None);
        assert_eq!(result.cache_bytes, GIB / 2);
        assert_eq!(result.fit, Fit::Tight, "{}", result.verdict);
        assert!(result.can_install(), "{}", result.verdict);
    }

    #[test]
    fn too_big_when_needed_exceeds_80_percent_of_ram() {
        let info = with_tags(repo(&[("model-Q4_K_M.gguf", 8 * GIB)]), &["tool-use"]);
        let result = assess_with(&info, &machine("macos", "aarch64", 8, 100), None);
        assert_eq!(result.fit, Fit::TooBig);
        assert!(result.verdict.contains("No cabe"), "{}", result.verdict);
        assert!(!result.can_install());
    }

    #[test]
    fn disk_full_blocks_install_when_download_plus_margin_exceeds_free_space() {
        // 1 GiB download needs 1.05 GiB free; only 1 GiB is available.
        let info = with_tags(repo(&[("model-Q4_K_M.gguf", GIB)]), &["tool-use"]);
        let result = assess_with(&info, &machine("macos", "aarch64", 16, 1), None);
        assert_eq!(result.fit, Fit::Fits);
        assert!(!result.disk_ok);
        assert!(result.verdict.contains("disco"), "{}", result.verdict);
        assert!(!result.can_install());
    }

    #[test]
    fn mlx_is_blocked_on_windows_and_intel_macs_but_not_on_apple_silicon() {
        let info = with_tags(
            repo(&[("model.safetensors", GIB), ("config.json", 1)]),
            &["tool-use"],
        );

        let windows = assess_with(&info, &machine("windows", "x86_64", 16, 100), None);
        assert_eq!(windows.format, Some(ModelFormat::Mlx));
        assert_eq!(windows.blocker, Some(Blocker::UnsupportedPlatform));
        assert!(!windows.can_install());

        let intel = assess_with(&info, &machine("macos", "x86_64", 16, 100), None);
        assert_eq!(intel.blocker, Some(Blocker::UnsupportedArchitecture));
        assert!(!intel.can_install());

        let silicon = assess_with(&info, &machine("macos", "aarch64", 16, 100), None);
        assert_eq!(silicon.blocker, None, "{}", silicon.verdict);
    }

    #[test]
    fn chat_template_that_renders_tools_passes_the_tool_check() {
        let info = repo(&[("model-Q4_K_M.gguf", GIB)]);
        let tokenizer = json!({ "chat_template": TOOL_TEMPLATE });
        let result = assess_request(&info, &machine("macos", "aarch64", 16, 100), None, Some(&tokenizer), None);
        assert!(result.supports_tools);
        assert_eq!(result.blocker, None, "{}", result.verdict);
        assert!(result.can_install(), "{}", result.verdict);
    }

    #[test]
    fn chat_template_list_form_is_scanned_for_tool_markers() {
        let tokenizer = json!({
            "chat_template": [
                { "name": "default", "template": "plain chat" },
                { "name": "tool_use", "template": "<tool_call>{{ call }}</tool_call>" }
            ]
        });
        assert!(detect_tool_support(&repo(&[]), Some(&tokenizer), None));
    }

    #[test]
    fn tool_use_tag_passes_the_tool_check_whatever_the_casing() {
        let info = with_tags(repo(&[("model-Q4_K_M.gguf", GIB)]), &["Text-Generation", "Tool-Use"]);
        let result = assess_with(&info, &machine("macos", "aarch64", 16, 100), None);
        assert!(result.supports_tools);
        assert!(result.can_install(), "{}", result.verdict);
    }

    #[test]
    fn gguf_header_chat_template_with_tools_passes_the_tool_check() {
        let info = repo(&[("model-Q4_K_M.gguf", GIB)]);
        let header = gguf_header_with_template(TOOL_TEMPLATE);
        let result = assess_request(&info, &machine("windows", "x86_64", 16, 100), None, None, Some(&header));
        assert!(result.supports_tools);
        assert!(result.can_install(), "{}", result.verdict);
    }

    #[test]
    fn tool_words_outside_the_chat_template_do_not_count() {
        // GGUF vocabularies can hold "<tool_call>" as a token; only tokenizer.chat_template counts.
        let info = repo(&[("model-Q4_K_M.gguf", GIB)]);
        let mut vocabulary = b"tokenizer.ggml.tokens".to_vec();
        vocabulary.extend_from_slice(b"<tool_call> tools tool_calls function_call");
        assert!(!detect_tool_support(&info, None, Some(&vocabulary)));
        let plain = gguf_header_with_template("{{ messages }}");
        assert!(!detect_tool_support(&info, None, Some(&plain)));
    }

    #[test]
    fn gguf_chat_template_is_matched_by_exact_key_not_by_prefix() {
        // A longer sibling key that starts with the same text must not be mistaken for the chat template.
        let info = repo(&[("model-Q4_K_M.gguf", GIB)]);
        let mut header = b"GGUF\x03\x00\x00\x00".to_vec();
        header.extend(gguf_string_entry("tokenizer.chat_template.tool_use", "plain"));
        header.extend(gguf_string_entry("tokenizer.chat_template", TOOL_TEMPLATE));
        assert!(detect_tool_support(&info, None, Some(&header)));

        let mut plain = b"GGUF\x03\x00\x00\x00".to_vec();
        plain.extend(gguf_string_entry("tokenizer.chat_template.tool_use", TOOL_TEMPLATE));
        plain.extend(gguf_string_entry("tokenizer.chat_template", "{{ messages }}"));
        assert!(!detect_tool_support(&info, None, Some(&plain)));
    }

    #[test]
    fn model_without_tool_support_is_blocked_and_cannot_install() {
        let info = repo(&[("model-Q4_K_M.gguf", GIB)]);
        let tokenizer = json!({ "chat_template": "{{ messages }}" });
        let result = assess_request(&info, &machine("macos", "aarch64", 16, 100), None, Some(&tokenizer), None);
        assert!(!result.supports_tools);
        assert_eq!(result.blocker, Some(Blocker::NoToolSupport));
        assert!(!result.can_install());
        assert!(
            result.verdict.contains(
                "No se puede instalar: este modelo no declara soporte para herramientas (tool calling) en su plantilla de chat o metadatos de Hugging Face."
            ),
            "{}",
            result.verdict
        );
    }

    #[test]
    fn assessment_exposes_suggested_settings_from_generation_config() {
        let info = with_tags(repo(&[("model-Q4_K_M.gguf", GIB)]), &["tool-use"]);
        let generation = json!({ "temperature": 0.6, "top_p": 0.95 });
        let result = assess_full(
            &info,
            &machine("macos", "aarch64", 16, 100),
            None,
            Some(&generation),
            None,
            None,
        );
        assert_eq!(result.suggested_config.temperature, Some(0.6));
        assert_eq!(result.suggested_config.top_p, Some(0.95));
        assert_eq!(result.suggested_config.source_summary, "generation_config.json");
        assert!(result.can_install(), "{}", result.verdict);
    }

    #[test]
    fn platform_blocker_keeps_priority_over_the_tool_check() {
        let info = repo(&[("model.safetensors", GIB), ("config.json", 1)]);
        let result = assess_with(&info, &machine("windows", "x86_64", 16, 100), None);
        assert_eq!(result.blocker, Some(Blocker::UnsupportedPlatform));
    }
}
