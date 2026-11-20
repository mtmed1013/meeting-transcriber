//! Cross-platform detection of the machine facts used for fit assessment.

use std::path::{Path, PathBuf};
use std::process::Command;

/// Hardware and platform facts for the machine running the app.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MachineInfo {
    /// `std::env::consts::OS` value, for example `macos` or `windows`.
    pub os: String,
    /// `std::env::consts::ARCH` value, for example `aarch64` or `x86_64`.
    pub arch: String,
    /// Installed RAM in bytes; `0` when detection failed.
    pub ram_bytes: u64,
    /// Free bytes on the volume that holds the models directory.
    pub disk_free_bytes: u64,
    /// True on Apple Silicon, where GPU and CPU share RAM.
    pub unified_memory: bool,
    /// Dedicated GPU memory in bytes when reported.
    pub gpu_vram_bytes: Option<u64>,
}

/// Detects the current machine. `target_dir` decides which volume's free space is reported.
pub fn detect_machine(target_dir: &Path) -> MachineInfo {
    let arch = std::env::consts::ARCH.to_string();
    let probe = probe(target_dir);
    MachineInfo {
        os: std::env::consts::OS.to_string(),
        unified_memory: cfg!(target_os = "macos") && arch == "aarch64",
        arch,
        ram_bytes: probe.ram_bytes,
        disk_free_bytes: probe.disk_free_bytes,
        gpu_vram_bytes: probe.gpu_vram_bytes,
    }
}

/// Platform-specific readings; `0` means the value could not be detected.
#[derive(Default)]
struct Probe {
    ram_bytes: u64,
    disk_free_bytes: u64,
    gpu_vram_bytes: Option<u64>,
}

#[cfg(target_os = "macos")]
fn probe(target_dir: &Path) -> Probe {
    Probe {
        ram_bytes: run_stdout("sysctl", ["-n", "hw.memsize"])
            .and_then(|out| out.trim().parse().ok())
            .unwrap_or(0),
        disk_free_bytes: df_available_bytes(target_dir),
        gpu_vram_bytes: None,
    }
}

#[cfg(all(unix, not(target_os = "macos")))]
fn probe(target_dir: &Path) -> Probe {
    Probe {
        ram_bytes: meminfo_total_bytes(&std::fs::read_to_string("/proc/meminfo").unwrap_or_default()),
        disk_free_bytes: df_available_bytes(target_dir),
        gpu_vram_bytes: None,
    }
}

#[cfg(target_os = "windows")]
fn probe(target_dir: &Path) -> Probe {
    use std::os::windows::process::CommandExt;

    // CREATE_NO_WINDOW: do not flash a console for the PowerShell child.
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;

    let disk = match nearest_existing_ancestor(target_dir).and_then(|p| drive_letter(&p)) {
        Some(drive) => format!(
            "(Get-CimInstance -ClassName Win32_LogicalDisk -Filter \"DeviceID='{drive}'\").FreeSpace"
        ),
        None => "0".to_string(),
    };
    let script = format!(
        "$cs = Get-CimInstance -ClassName Win32_ComputerSystem; \
         $gpu = Get-CimInstance -ClassName Win32_VideoController | Sort-Object -Property AdapterRAM -Descending | Select-Object -First 1; \
         [pscustomobject]@{{ ram = $cs.TotalPhysicalMemory; vram = $gpu.AdapterRAM; free = {disk} }} | ConvertTo-Json -Compress"
    );
    let json: serde_json::Value = Command::new("powershell")
        .args(["-NoProfile", "-NonInteractive", "-Command", script.as_str()])
        .creation_flags(CREATE_NO_WINDOW)
        .output()
        .ok()
        .and_then(|out| serde_json::from_slice(&out.stdout).ok())
        .unwrap_or(serde_json::Value::Null);
    let number = |key: &str| json.get(key).and_then(serde_json::Value::as_u64).unwrap_or(0);
    Probe {
        ram_bytes: number("ram"),
        disk_free_bytes: number("free"),
        gpu_vram_bytes: Some(number("vram")).filter(|bytes| *bytes > 0),
    }
}

/// Parses the `MemTotal:` line of `/proc/meminfo` (reported in kiB).
#[cfg(all(unix, not(target_os = "macos")))]
fn meminfo_total_bytes(meminfo: &str) -> u64 {
    meminfo
        .lines()
        .find_map(|line| line.strip_prefix("MemTotal:"))
        .and_then(|rest| rest.split_whitespace().next())
        .and_then(|kib| kib.parse::<u64>().ok())
        .map_or(0, |kib| kib * 1024)
}

/// Free bytes on the volume holding `target_dir`, read from `df -Pk` on the nearest existing ancestor.
#[cfg(unix)]
fn df_available_bytes(target_dir: &Path) -> u64 {
    let Some(ancestor) = nearest_existing_ancestor(target_dir) else {
        return 0;
    };
    run_stdout("df", [std::ffi::OsStr::new("-Pk"), ancestor.as_os_str()])
        .and_then(|out| {
            out.lines()
                .nth(1)
                .and_then(|line| line.split_whitespace().nth(3))
                .and_then(|kib| kib.parse::<u64>().ok())
        })
        .map_or(0, |kib| kib * 1024)
}

/// Runs a program and returns its stdout when it exits successfully.
#[cfg(unix)]
fn run_stdout<I, S>(program: &str, args: I) -> Option<String>
where
    I: IntoIterator<Item = S>,
    S: AsRef<std::ffi::OsStr>,
{
    let output = Command::new(program).args(args).output().ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).into_owned())
}

/// Absolute form of `path`, or of its closest ancestor that exists on disk.
fn nearest_existing_ancestor(path: &Path) -> Option<PathBuf> {
    let absolute = std::path::absolute(path).ok()?;
    absolute
        .ancestors()
        .find(|candidate| candidate.exists())
        .map(Path::to_path_buf)
}

/// Drive designator such as `C:` for a Windows path, `None` for UNC or relative paths.
#[cfg(target_os = "windows")]
fn drive_letter(path: &Path) -> Option<String> {
    match path.components().next()? {
        std::path::Component::Prefix(prefix) => {
            let raw = prefix.as_os_str().to_str()?;
            let text = raw.strip_prefix(r"\\?\").unwrap_or(raw);
            let bytes = text.as_bytes();
            (bytes.len() == 2 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':')
                .then(|| text.to_string())
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_machine_reports_platform_and_disk_for_temp_dir() {
        let dir = tempfile::tempdir().unwrap();
        let info = detect_machine(dir.path());
        assert_eq!(info.os, std::env::consts::OS);
        assert_eq!(info.arch, std::env::consts::ARCH);
        if cfg!(any(target_os = "macos", target_os = "linux")) {
            assert!(info.ram_bytes > 0, "ram: {info:?}");
            assert!(info.disk_free_bytes > 0, "disk: {info:?}");
        }
    }

    #[test]
    fn disk_is_read_from_nearest_existing_ancestor_of_missing_target() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("models").join("not-yet");
        assert_eq!(nearest_existing_ancestor(&missing), Some(dir.path().to_path_buf()));
        if cfg!(unix) {
            let info = detect_machine(&missing);
            assert!(info.disk_free_bytes > 0, "disk: {info:?}");
        }
    }
}
