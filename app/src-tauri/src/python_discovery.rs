//! Locates a Python >= 3.10 interpreter with the host's native architecture for Meeting
//! Transcriber's local speech-to-text and AI runtimes.

use crate::stt_provisioning::hidden_command;
use serde::Deserialize;
use std::collections::HashSet;
use std::env;
use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};

/// Oldest accepted Python 3 minor version.
pub const MIN_PYTHON_MINOR: u32 = 10;
/// Oldest Python 3 minor version `mlx-lm` supports; the MLX runtime venv requires it.
pub const MLX_PYTHON_MINOR: u32 = 11;
/// Minor versions preferred over any other accepted version.
const PREFERRED_MINOR_VERSIONS: std::ops::RangeInclusive<u32> = 11..=14;

const PYTHON_PROBE: &str = "import json, platform, sys; print(json.dumps({'version': list(sys.version_info[:3]), 'machine': platform.machine(), 'executable': sys.executable}))";
const VERSIONED_EXECUTABLES: &[&str] = &[
    "python3.14",
    "python3.13",
    "python3.12",
    "python3.11",
    "python3.10",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct PythonVersion {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
}

impl std::fmt::Display for PythonVersion {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PythonInstallation {
    pub executable: PathBuf,
    pub version: PythonVersion,
    pub architecture: String,
}

/// A program to probe, plus launcher arguments such as `-3` for the Windows `py` launcher.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Candidate {
    program: PathBuf,
    launcher_args: Vec<String>,
}

impl Candidate {
    fn direct(program: PathBuf) -> Self {
        Self {
            program,
            launcher_args: Vec::new(),
        }
    }

    fn label(&self) -> String {
        let program = self.program.display();
        if self.launcher_args.is_empty() {
            program.to_string()
        } else {
            format!("{program} {}", self.launcher_args.join(" "))
        }
    }
}

#[derive(Debug, Deserialize)]
struct ProbeResult {
    version: [u32; 3],
    machine: String,
    #[serde(default)]
    executable: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ArchFamily {
    Arm,
    X64,
}

/// Finds and probes Python candidates and returns the best compatible interpreter.
pub fn discover_python() -> Result<PythonInstallation, String> {
    discover_python_at_least(MIN_PYTHON_MINOR)
}

/// Like [`discover_python`], but accepts only Python 3 interpreters whose minor version is at least
/// `minimum_minor`. Candidate collection, probing, and selection are shared with `discover_python`.
pub fn discover_python_at_least(minimum_minor: u32) -> Result<PythonInstallation, String> {
    select_python_at_least(
        platform_candidates(),
        probe_candidate,
        env::consts::ARCH,
        minimum_minor,
    )
}

/// Returns `true` only if the interpreter is Python >= 3.10 with the host's native architecture.
pub fn is_python_compatible(executable: &Path) -> bool {
    probe_candidate(&Candidate::direct(executable.to_path_buf()))
        .map(|probe| is_compatible_probe(&probe, env::consts::ARCH))
        .unwrap_or(false)
}

#[cfg(not(windows))]
fn platform_candidates() -> Vec<Candidate> {
    let path = env::var_os("PATH").unwrap_or_default();
    let home = env::var_os("HOME").map(PathBuf::from);
    collect_unix_candidates(
        &path,
        home.as_deref(),
        &[
            PathBuf::from("/opt/homebrew/bin"),
            PathBuf::from("/usr/local/bin"),
            PathBuf::from("/opt/local/bin"),
        ],
        &[PathBuf::from(
            "/Library/Frameworks/Python.framework/Versions",
        )],
        &[
            PathBuf::from("/opt/anaconda3"),
            PathBuf::from("/opt/miniconda3"),
            PathBuf::from("/opt/homebrew/Caskroom/miniforge/base"),
            PathBuf::from("/opt/homebrew/Caskroom/miniconda/base"),
        ],
    )
}

#[cfg(windows)]
fn platform_candidates() -> Vec<Candidate> {
    let mut candidates = Vec::new();
    let path = env::var_os("PATH").unwrap_or_default();
    for directory in env::split_paths(&path) {
        add_executables_in(&mut candidates, &directory);
    }

    let local_programs = env::var_os("LOCALAPPDATA")
        .map(|local| PathBuf::from(local).join("Programs").join("Python"));
    let program_files = env::var_os("ProgramFiles").map(PathBuf::from);
    for root in local_programs.into_iter().chain(program_files) {
        add_python3_installations(&mut candidates, &root);
    }

    candidates.push(Candidate {
        program: PathBuf::from("py"),
        launcher_args: vec!["-3".into()],
    });
    deduplicate_candidates(candidates)
}

#[cfg(not(windows))]
fn collect_unix_candidates(
    path: &OsStr,
    home: Option<&Path>,
    bin_dirs: &[PathBuf],
    framework_roots: &[PathBuf],
    prefix_roots: &[PathBuf],
) -> Vec<Candidate> {
    let mut candidates = Vec::new();
    for directory in env::split_paths(path) {
        add_executables_in(&mut candidates, &directory);
    }

    for directory in bin_dirs {
        add_executables_in(&mut candidates, directory);
    }

    for root in framework_roots {
        add_versioned_installations(&mut candidates, root);
    }

    if let Some(home) = home {
        add_versioned_installations(&mut candidates, &home.join(".pyenv/versions"));
        add_versioned_installations(&mut candidates, &home.join(".local/share/uv/python"));

        for prefix in ["miniforge3", "mambaforge", "anaconda3", "miniconda3"] {
            add_executables_in(&mut candidates, &home.join(prefix).join("bin"));
        }
    }

    for prefix in prefix_roots {
        add_executables_in(&mut candidates, &prefix.join("bin"));
    }

    deduplicate_candidates(candidates)
}

/// Adds the interpreters in each `<root>/<version>/bin` directory, e.g. pyenv and uv layouts.
fn add_versioned_installations(candidates: &mut Vec<Candidate>, root: &Path) {
    let Ok(entries) = fs::read_dir(root) else {
        return;
    };

    let mut directories = entries
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.is_dir())
        .collect::<Vec<_>>();
    directories.sort();

    for directory in directories {
        add_executables_in(candidates, &directory.join("bin"));
    }
}

/// Adds `<root>\Python3*\python.exe` for each Python 3 installation directory under `root`.
#[cfg(windows)]
fn add_python3_installations(candidates: &mut Vec<Candidate>, root: &Path) {
    let Ok(entries) = fs::read_dir(root) else {
        return;
    };

    let mut directories = entries
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_dir()
                && path
                    .file_name()
                    .and_then(OsStr::to_str)
                    .is_some_and(|name| name.to_ascii_lowercase().starts_with("python3"))
        })
        .collect::<Vec<_>>();
    directories.sort();

    for directory in directories {
        let executable = directory.join(executable_file_name("python"));
        if executable.is_file() {
            candidates.push(Candidate::direct(executable));
        }
    }
}

fn add_executables_in(candidates: &mut Vec<Candidate>, directory: &Path) {
    for name in VERSIONED_EXECUTABLES
        .iter()
        .copied()
        .chain(["python3", "python"])
    {
        let executable = directory.join(executable_file_name(name));
        if executable.is_file() {
            candidates.push(Candidate::direct(executable));
        }
    }
}

fn executable_file_name(name: &str) -> String {
    if cfg!(windows) {
        format!("{name}.exe")
    } else {
        name.to_string()
    }
}

/// Removes duplicate interpreters, resolving symlinks for absolute paths so aliases collapse.
fn deduplicate_candidates(candidates: Vec<Candidate>) -> Vec<Candidate> {
    let mut seen = HashSet::new();
    candidates
        .into_iter()
        .filter_map(|candidate| {
            let program = if candidate.program.is_absolute() {
                candidate
                    .program
                    .canonicalize()
                    .unwrap_or_else(|_| candidate.program.clone())
            } else {
                candidate.program.clone()
            };
            let resolved = Candidate {
                program,
                launcher_args: candidate.launcher_args,
            };
            seen.insert((resolved.program.clone(), resolved.launcher_args.clone()))
                .then_some(resolved)
        })
        .collect()
}

fn probe_candidate(candidate: &Candidate) -> Result<ProbeResult, String> {
    let label = candidate.label();
    let output = hidden_command(&candidate.program)
        .args(&candidate.launcher_args)
        .args(["-c", PYTHON_PROBE])
        .output()
        .map_err(|error| format!("{label}: no se pudo ejecutar ({error})"))?;

    if !output.status.success() {
        let detail = String::from_utf8_lossy(&output.stderr).trim().to_string();
        return Err(if detail.is_empty() {
            format!("{label}: terminó con código {}", output.status)
        } else {
            format!("{label}: {detail}")
        });
    }

    serde_json::from_slice(&output.stdout)
        .map_err(|error| format!("{label}: respuesta inválida ({error})"))
}

fn arch_family(architecture: &str) -> Option<ArchFamily> {
    match architecture.to_ascii_lowercase().as_str() {
        "arm64" | "aarch64" => Some(ArchFamily::Arm),
        "x86_64" | "amd64" => Some(ArchFamily::X64),
        _ => None,
    }
}

/// Architecture name shown to the user for the host, e.g. `arm64` on Apple Silicon.
fn required_arch_label(host_arch: &str) -> &str {
    match arch_family(host_arch) {
        Some(ArchFamily::Arm) => "arm64",
        Some(ArchFamily::X64) => "x86_64",
        None => host_arch,
    }
}

fn arch_matches(host_arch: &str, machine: &str) -> bool {
    matches!(
        (arch_family(host_arch), arch_family(machine)),
        (Some(host), Some(reported)) if host == reported
    )
}

fn is_compatible_probe(probe: &ProbeResult, host_arch: &str) -> bool {
    probe.version[0] == 3
        && probe.version[1] >= MIN_PYTHON_MINOR
        && arch_matches(host_arch, &probe.machine)
}

/// Preferred minors first, then the newest version; equal ranks keep discovery order.
fn preference_rank(version: PythonVersion) -> (bool, PythonVersion) {
    (PREFERRED_MINOR_VERSIONS.contains(&version.minor), version)
}

fn select_python_at_least<F>(
    candidates: Vec<Candidate>,
    mut probe: F,
    host_arch: &str,
    minimum_minor: u32,
) -> Result<PythonInstallation, String>
where
    F: FnMut(&Candidate) -> Result<ProbeResult, String>,
{
    let mut compatible = Vec::new();
    let mut rejected = Vec::new();

    for (order, candidate) in deduplicate_candidates(candidates).into_iter().enumerate() {
        let label = candidate.label();
        let result = match probe(&candidate) {
            Ok(result) => result,
            Err(error) => {
                rejected.push(error);
                continue;
            }
        };
        let version = PythonVersion {
            major: result.version[0],
            minor: result.version[1],
            patch: result.version[2],
        };

        if version.major != 3 {
            rejected.push(format!("{label}: Python {version} no es Python 3"));
            continue;
        }
        if version.minor < minimum_minor {
            rejected.push(format!(
                "{label}: Python {version} es anterior a 3.{minimum_minor}"
            ));
            continue;
        }
        if !arch_matches(host_arch, &result.machine) {
            rejected.push(format!(
                "{label}: Python {version} usa arquitectura {}, se requiere {}",
                result.machine,
                required_arch_label(host_arch)
            ));
            continue;
        }

        let executable = match result.executable.filter(|reported| !reported.is_empty()) {
            Some(reported) => PathBuf::from(reported),
            None if candidate.launcher_args.is_empty() => candidate.program.clone(),
            None => {
                rejected.push(format!("{label}: no se pudo resolver sys.executable"));
                continue;
            }
        };

        compatible.push((
            order,
            PythonInstallation {
                executable,
                version,
                architecture: result.machine,
            },
        ));
    }

    compatible.sort_by(|(left_order, left), (right_order, right)| {
        preference_rank(right.version)
            .cmp(&preference_rank(left.version))
            .then_with(|| left_order.cmp(right_order))
    });

    if let Some((_, selected)) = compatible.into_iter().next() {
        return Ok(selected);
    }

    Err(no_compatible_python_error(
        host_arch,
        minimum_minor,
        &rejected,
    ))
}

fn no_compatible_python_error(host_arch: &str, minimum_minor: u32, rejected: &[String]) -> String {
    let hint = if cfg!(target_os = "macos") {
        "Instala Python desde Homebrew (brew install python@3.12).".to_string()
    } else if cfg!(windows) {
        "Instálalo desde python.org o con winget (winget install Python.Python.3.12).".to_string()
    } else {
        format!("Instala Python 3.{minimum_minor} o superior.")
    };
    let detail = if rejected.is_empty() {
        " No se encontraron ejecutables de Python.".to_string()
    } else {
        format!(" Candidatos descartados: {}.", rejected.join("; "))
    };
    format!(
        "No se encontró un Python compatible: se requiere Python 3.{minimum_minor} o superior con arquitectura {}. {hint}{detail}",
        required_arch_label(host_arch)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The STT minimum (3.10); the shared selector is exercised through this default.
    fn select_python<F>(
        candidates: Vec<Candidate>,
        probe: F,
        host_arch: &str,
    ) -> Result<PythonInstallation, String>
    where
        F: FnMut(&Candidate) -> Result<ProbeResult, String>,
    {
        select_python_at_least(candidates, probe, host_arch, MIN_PYTHON_MINOR)
    }

    fn probe(version: [u32; 3], machine: &str) -> Result<ProbeResult, String> {
        Ok(ProbeResult {
            version,
            machine: machine.into(),
            executable: None,
        })
    }

    fn direct(path: &str) -> Candidate {
        Candidate::direct(PathBuf::from(path))
    }

    #[test]
    fn rejects_apple_python_39_even_when_architecture_matches() {
        let error = select_python(
            vec![direct("/usr/bin/python3")],
            |_| probe([3, 9, 6], "arm64"),
            "aarch64",
        )
        .unwrap_err();

        assert!(error.contains("/usr/bin/python3: Python 3.9.6 es anterior a 3.10"));
        assert!(error.starts_with("No se encontró un Python compatible"));
    }

    #[test]
    fn rejects_x86_64_python_on_apple_silicon() {
        let error = select_python(
            vec![direct("/usr/local/bin/python3.13")],
            |_| probe([3, 13, 2], "x86_64"),
            "aarch64",
        )
        .unwrap_err();

        assert!(error.contains(
            "/usr/local/bin/python3.13: Python 3.13.2 usa arquitectura x86_64, se requiere arm64"
        ));
    }

    #[test]
    fn rejects_arm64_python_on_intel_host() {
        let error = select_python(
            vec![direct("/opt/homebrew/bin/python3.13")],
            |_| probe([3, 13, 2], "arm64"),
            "x86_64",
        )
        .unwrap_err();

        assert!(error.contains("usa arquitectura arm64, se requiere x86_64"));
    }

    #[test]
    fn accepts_platform_architecture_aliases() {
        let arm = select_python(
            vec![direct("/fake/arm")],
            |_| probe([3, 12, 4], "aarch64"),
            "aarch64",
        )
        .unwrap();
        assert_eq!(arm.architecture, "aarch64");

        let windows_x64 = select_python(
            vec![direct("/fake/windows")],
            |_| probe([3, 12, 4], "AMD64"),
            "x86_64",
        )
        .unwrap();
        assert_eq!(windows_x64.architecture, "AMD64");
    }

    #[test]
    fn prefers_newest_python_in_preferred_range() {
        let candidates = vec![
            direct("/fake/3.10"),
            direct("/fake/3.12"),
            direct("/fake/3.14"),
            direct("/fake/3.13"),
        ];
        let selected = select_python(
            candidates,
            |candidate| {
                let version = match candidate.program.to_string_lossy().as_ref() {
                    "/fake/3.10" => [3, 10, 12],
                    "/fake/3.12" => [3, 12, 13],
                    "/fake/3.14" => [3, 14, 6],
                    "/fake/3.13" => [3, 13, 13],
                    _ => unreachable!(),
                };
                probe(version, "arm64")
            },
            "aarch64",
        )
        .unwrap();

        assert_eq!(selected.executable, PathBuf::from("/fake/3.14"));
        assert_eq!(
            selected.version,
            PythonVersion {
                major: 3,
                minor: 14,
                patch: 6
            }
        );
    }

    #[test]
    fn prefers_preferred_range_over_newer_version_and_accepts_3_10_alone() {
        let selected = select_python(
            vec![direct("/fake/3.15"), direct("/fake/3.12")],
            |candidate| {
                if candidate.program.ends_with("3.15") {
                    probe([3, 15, 0], "arm64")
                } else {
                    probe([3, 12, 1], "arm64")
                }
            },
            "aarch64",
        )
        .unwrap();
        assert_eq!(selected.version.minor, 12);

        let only_310 = select_python(
            vec![direct("/fake/3.10")],
            |_| probe([3, 10, 12], "arm64"),
            "aarch64",
        )
        .unwrap();
        assert_eq!(only_310.version.minor, 10);
    }

    #[test]
    fn selected_executable_is_sys_executable_when_probe_reports_it() {
        let selected = select_python(
            vec![direct("/fake/launcher")],
            |_| {
                Ok(ProbeResult {
                    version: [3, 14, 6],
                    machine: "arm64".into(),
                    executable: Some("/opt/homebrew/opt/python@3.14/bin/python3.14".into()),
                })
            },
            "aarch64",
        )
        .unwrap();

        assert_eq!(
            selected.executable,
            PathBuf::from("/opt/homebrew/opt/python@3.14/bin/python3.14")
        );
    }

    #[test]
    fn launcher_without_reported_executable_is_rejected() {
        let error = select_python(
            vec![Candidate {
                program: PathBuf::from("py"),
                launcher_args: vec!["-3".into()],
            }],
            |_| probe([3, 12, 4], "AMD64"),
            "x86_64",
        )
        .unwrap_err();

        assert!(error.contains("py -3: no se pudo resolver sys.executable"));
    }

    #[cfg(not(windows))]
    #[test]
    fn restricted_finder_path_still_finds_homebrew_framework_uv_and_pyenv() {
        let scratch = tempfile::tempdir().unwrap();
        let root = scratch.path();
        let finder_bin = root.join("finder-bin");
        let brew_bin = root.join("homebrew/bin");
        let framework_root = root.join("framework/Versions");
        let home = root.join("home");
        let conda = root.join("conda");
        let fixtures = [
            finder_bin.join("python3"),
            brew_bin.join("python3.14"),
            framework_root.join("3.13/bin/python3.13"),
            home.join(".pyenv/versions/3.12/bin/python3.12"),
            home.join(".local/share/uv/python/cpython-3.11/bin/python3.11"),
            home.join("miniforge3/bin/python3"),
            conda.join("bin/python3.14"),
        ];
        for path in &fixtures {
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(path, "fake interpreter").unwrap();
        }

        let path = env::join_paths([finder_bin.clone()]).unwrap();
        let candidates = collect_unix_candidates(
            &path,
            Some(&home),
            std::slice::from_ref(&brew_bin),
            std::slice::from_ref(&framework_root),
            std::slice::from_ref(&conda),
        );

        for expected in fixtures {
            let expected = expected.canonicalize().unwrap();
            assert!(
                candidates
                    .iter()
                    .any(|candidate| candidate.program == expected),
                "missing {}",
                expected.display()
            );
        }
    }

    #[test]
    fn deduplicates_repeated_candidates_but_keeps_distinct_launchers() {
        let repeated = deduplicate_candidates(vec![
            direct("/fake/python3.14"),
            direct("/fake/python3.14"),
        ]);
        assert_eq!(repeated, vec![direct("/fake/python3.14")]);

        let launchers = deduplicate_candidates(vec![
            Candidate {
                program: PathBuf::from("py"),
                launcher_args: vec!["-3".into()],
            },
            direct("py"),
        ]);
        assert_eq!(launchers.len(), 2);
    }

    #[cfg(unix)]
    #[test]
    fn deduplicates_symlink_aliases_to_the_same_interpreter() {
        let scratch = tempfile::tempdir().unwrap();
        let real = scratch.path().join("python3.14");
        fs::write(&real, "fake interpreter").unwrap();
        let alias = scratch.path().join("python3");
        std::os::unix::fs::symlink(&real, &alias).unwrap();

        let deduplicated = deduplicate_candidates(vec![
            Candidate::direct(real.clone()),
            Candidate::direct(alias),
        ]);

        assert_eq!(
            deduplicated,
            vec![Candidate::direct(real.canonicalize().unwrap())]
        );
    }

    #[test]
    fn probes_each_unique_candidate_once() {
        let mut probes = 0;
        let selected = select_python(
            vec![direct("/fake/python3.12"), direct("/fake/python3.12")],
            |_| {
                probes += 1;
                probe([3, 12, 4], "arm64")
            },
            "aarch64",
        )
        .unwrap();

        assert_eq!(probes, 1);
        assert_eq!(selected.version.minor, 12);
    }

    #[test]
    fn venv_compatibility_requires_supported_version_and_native_architecture() {
        let native = ProbeResult {
            version: [3, 12, 4],
            machine: "arm64".into(),
            executable: None,
        };
        assert!(is_compatible_probe(&native, "aarch64"));

        let old = ProbeResult {
            version: [3, 9, 6],
            machine: "arm64".into(),
            executable: None,
        };
        assert!(!is_compatible_probe(&old, "aarch64"));

        let foreign = ProbeResult {
            version: [3, 12, 4],
            machine: "x86_64".into(),
            executable: None,
        };
        assert!(!is_compatible_probe(&foreign, "aarch64"));
    }

    #[test]
    fn missing_venv_interpreter_is_not_compatible() {
        assert!(!is_python_compatible(Path::new(
            "/fake/does-not-exist/python3"
        )));
    }

    #[test]
    fn probe_json_accepts_executable_or_null() {
        let parsed: ProbeResult = serde_json::from_str(
            r#"{"version":[3,14,6],"machine":"arm64","executable":"/opt/homebrew/bin/python3.14"}"#,
        )
        .unwrap();
        assert_eq!(
            parsed.executable.as_deref(),
            Some("/opt/homebrew/bin/python3.14")
        );

        let null: ProbeResult =
            serde_json::from_str(r#"{"version":[3,14,6],"machine":"arm64","executable":null}"#)
                .unwrap();
        assert_eq!(null.executable, None);
    }

    #[test]
    fn no_python_error_is_spanish_and_names_the_platform_source() {
        let error = select_python(Vec::new(), |_| unreachable!(), "aarch64").unwrap_err();

        assert!(error.contains("3.10 o superior"));
        if cfg!(target_os = "macos") {
            assert!(error.contains("Homebrew"));
        }
        if cfg!(windows) {
            assert!(error.contains("python.org") && error.contains("winget"));
        }
    }

    #[test]
    fn minimum_minor_is_a_parameter_so_mlx_can_require_3_11() {
        let error = select_python_at_least(
            vec![direct("/fake/3.10")],
            |_| probe([3, 10, 12], "arm64"),
            "aarch64",
            MLX_PYTHON_MINOR,
        )
        .unwrap_err();
        assert!(
            error.contains("/fake/3.10: Python 3.10.12 es anterior a 3.11"),
            "{error}"
        );
        assert!(
            error.contains("se requiere Python 3.11 o superior"),
            "{error}"
        );

        let selected = select_python_at_least(
            vec![direct("/fake/3.10"), direct("/fake/3.11")],
            |candidate| {
                if candidate.program.ends_with("3.10") {
                    probe([3, 10, 12], "arm64")
                } else {
                    probe([3, 11, 9], "arm64")
                }
            },
            "aarch64",
            MLX_PYTHON_MINOR,
        )
        .unwrap();
        assert_eq!(selected.version.minor, 11);
        assert_eq!(selected.executable, PathBuf::from("/fake/3.11"));
    }

    #[test]
    fn version_displays_as_dotted_triplet() {
        assert_eq!(
            PythonVersion {
                major: 3,
                minor: 14,
                patch: 6
            }
            .to_string(),
            "3.14.6"
        );
    }
}
