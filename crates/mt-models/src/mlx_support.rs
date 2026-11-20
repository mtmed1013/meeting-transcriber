//! Which model architectures the installed MLX runtime (`mlx_lm`) can load, read from its
//! `site-packages` folder without importing Python.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use serde_json::Value;

/// Package folder of the MLX text runtime inside `site-packages`.
const MLX_LM: &str = "mlx_lm";
/// Opening line of the alias table in `mlx_lm/utils.py`.
const MODEL_REMAPPING_START: &str = "MODEL_REMAPPING = {";

/// Locates the `site-packages` folder of a virtual environment: the first existing
/// `lib/python3.*/site-packages` directory, in name order.
pub fn find_site_packages(venv: &Path) -> Option<PathBuf> {
    let mut libs: Vec<PathBuf> = fs::read_dir(venv.join("lib"))
        .ok()?
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("python3."))
        })
        .collect();
    libs.sort();
    libs.into_iter()
        .map(|lib| lib.join("site-packages"))
        .find(|site| site.is_dir())
}

/// Model types the installed `mlx_lm` supports: one per `mlx_lm/models/*.py` file (names starting
/// with `_` skipped), plus each `MODEL_REMAPPING` alias whose target is already supported.
/// Empty when the venv or its `mlx_lm` package is missing.
pub fn supported_model_types(venv: &Path) -> BTreeSet<String> {
    let Some(site) = find_site_packages(venv) else {
        return BTreeSet::new();
    };
    let package = site.join(MLX_LM);
    let mut types = model_names(&package.join("models"));
    if let Ok(utils) = fs::read_to_string(package.join("utils.py")) {
        for (alias, target) in remapping(&utils) {
            if types.contains(&target) {
                types.insert(alias);
            }
        }
    }
    types
}

/// The `model_type` named in a model folder's `config.json`, if readable.
pub fn model_type_from_dir(model_dir: &Path) -> Option<String> {
    let config: Value =
        serde_json::from_slice(&fs::read(model_dir.join("config.json")).ok()?).ok()?;
    config.get("model_type")?.as_str().map(str::to_string)
}

/// Why `model_type` cannot run on the installed runtime, or `None` when it is supported.
/// An empty `mlx_lm_version` means the runtime is not installed.
pub fn unsupported_reason(
    model_type: &str,
    supported: &BTreeSet<String>,
    mlx_lm_version: &str,
) -> Option<String> {
    if supported.contains(model_type) {
        return None;
    }
    if mlx_lm_version.is_empty() {
        return Some(format!(
            "No se puede usar el modelo «{model_type}»: el runtime MLX no está instalado."
        ));
    }
    Some(format!(
        "mlx-lm {mlx_lm_version} no soporta el modelo «{model_type}». Actualiza el runtime MLX o elige otro modelo."
    ))
}

/// `.py` file stems in a folder, skipping private names that start with `_`.
fn model_names(models_dir: &Path) -> BTreeSet<String> {
    fs::read_dir(models_dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|entry| {
            let name = entry.file_name().to_string_lossy().into_owned();
            let stem = name.strip_suffix(".py")?;
            (!stem.starts_with('_')).then(|| stem.to_string())
        })
        .collect()
}

/// `"a": "b"` pairs inside the `MODEL_REMAPPING = {…}` block of mlx_lm's `utils.py`.
fn remapping(utils: &str) -> Vec<(String, String)> {
    let Some(start) = utils.find(MODEL_REMAPPING_START) else {
        return Vec::new();
    };
    let body = &utils[start..];
    let end = body.find("\n}").unwrap_or(body.len());
    body[..end]
        .lines()
        .skip(1)
        .filter_map(|line| {
            let line = line.split('#').next()?;
            let (key, value) = line.split_once(':')?;
            let clean = |s: &str| {
                s.trim()
                    .trim_end_matches(',')
                    .trim()
                    .trim_matches('"')
                    .to_string()
            };
            Some((clean(key), clean(value)))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    /// `<root>/lib/python3.12/site-packages`, created on demand.
    fn site_packages(root: &Path) -> PathBuf {
        let dir = root.join("lib/python3.12/site-packages");
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn set(items: &[&str]) -> BTreeSet<String> {
        items.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn site_packages_is_found_under_the_python_lib_folder() {
        let dir = tempfile::tempdir().unwrap();
        let expected = site_packages(dir.path());
        assert_eq!(find_site_packages(dir.path()), Some(expected));
    }

    #[test]
    fn site_packages_is_none_when_the_venv_has_no_python_lib() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(find_site_packages(dir.path()), None);
        fs::create_dir_all(dir.path().join("lib/python3.12")).unwrap();
        assert_eq!(find_site_packages(dir.path()), None);
    }

    #[test]
    fn stems_are_collected_and_underscore_names_skipped() {
        let dir = tempfile::tempdir().unwrap();
        let models = site_packages(dir.path()).join("mlx_lm/models");
        fs::create_dir_all(&models).unwrap();
        for name in ["llama.py", "qwen3.py", "__init__.py", "_private.py", "notes.txt"] {
            fs::write(models.join(name), "").unwrap();
        }
        assert_eq!(
            supported_model_types(dir.path()),
            set(&["llama", "qwen3"])
        );
    }

    #[test]
    fn remapping_aliases_are_added_only_when_their_target_is_supported() {
        let dir = tempfile::tempdir().unwrap();
        let package = site_packages(dir.path()).join("mlx_lm");
        fs::create_dir_all(package.join("models")).unwrap();
        fs::write(package.join("models/llama.py"), "").unwrap();
        fs::write(
            package.join("utils.py"),
            "MODEL_REMAPPING = {\n    \"mistral\": \"llama\",\n    \"ghost\": \"nothere\",  # c\n}\n",
        )
        .unwrap();
        assert_eq!(
            supported_model_types(dir.path()),
            set(&["llama", "mistral"])
        );
    }

    #[test]
    fn missing_venv_yields_an_empty_set() {
        let dir = tempfile::tempdir().unwrap();
        assert!(supported_model_types(&dir.path().join("absent")).is_empty());
    }

    #[test]
    fn model_type_is_read_from_config_json() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("config.json"), r#"{"model_type": "qwen3"}"#).unwrap();
        assert_eq!(model_type_from_dir(dir.path()), Some("qwen3".to_string()));
    }

    #[test]
    fn model_type_is_none_without_a_usable_config_json() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(model_type_from_dir(dir.path()), None);
        fs::write(dir.path().join("config.json"), r#"{"hidden_size": 4}"#).unwrap();
        assert_eq!(model_type_from_dir(dir.path()), None);
        fs::write(dir.path().join("config.json"), "not json").unwrap();
        assert_eq!(model_type_from_dir(dir.path()), None);
    }

    #[test]
    fn supported_model_has_no_reason() {
        assert_eq!(unsupported_reason("llama", &set(&["llama"]), "0.32.2"), None);
    }

    #[test]
    fn unsupported_model_names_the_runtime_version() {
        assert_eq!(
            unsupported_reason("brandnew", &set(&["llama"]), "0.32.2"),
            Some(
                "mlx-lm 0.32.2 no soporta el modelo «brandnew». Actualiza el runtime MLX o elige otro modelo."
                    .to_string()
            )
        );
    }

    #[test]
    fn empty_version_says_the_runtime_is_not_installed() {
        assert_eq!(
            unsupported_reason("llama", &BTreeSet::new(), ""),
            Some("No se puede usar el modelo «llama»: el runtime MLX no está instalado.".to_string())
        );
    }
}
