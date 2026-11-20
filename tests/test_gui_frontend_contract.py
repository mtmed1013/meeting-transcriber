import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "app" / "src"
RUST = ROOT / "app" / "src-tauri" / "src"


def rust_commands():
    commands = {}
    for path in RUST.glob("*.rs"):
        text = path.read_text(encoding="utf-8")
        for match in re.finditer(
            r"#\[tauri::command\]\s*pub\s+(?:async\s+)?fn\s+(\w+)\s*\((.*?)\)\s*(?:->|\{)",
            text,
            re.S,
        ):
            params = [
                p.strip()
                for p in re.split(r",\s*(?=\w+\s*:)", match.group(2))
                if p.strip()
            ]
            required = []
            for param in params:
                name, ty = [part.strip() for part in param.split(":", 1)]
                if ty.startswith(("AppHandle", "State<", "tauri::")) or ty.startswith("Option<"):
                    continue
                required.append(name)
            commands[match.group(1)] = required
    return commands


class FrontendContractTest(unittest.TestCase):
    def test_unified_index_has_session_and_settings_tabs(self):
        index = SRC / "index.html"
        self.assertTrue(index.is_file(), "app/src/index.html must exist")
        text = index.read_text(encoding="utf-8")
        self.assertIn('id="tab-session"', text)
        self.assertIn('id="tab-corrections"', text)

    def test_no_inline_style_or_innerhtml(self):
        for name in ("index.html", "assets/app.js"):
            text = (SRC / name).read_text(encoding="utf-8")
            code = (
                re.sub(r"/\*.*?\*/|//[^\n]*", "", text, flags=re.S)
                if name.endswith(".js")
                else text
            )
            self.assertIsNone(re.search(r"\sstyle\s*=", code), name)
            self.assertNotIn("innerHTML", code, name)

    def test_required_command_args_are_sent(self):
        commands = rust_commands()
        self.assertIn("start_session", commands)
        self.assertEqual(commands["start_session"], [])
        self.assertEqual(commands["stop_session"], [])
        for page in ("index.html", "assets/app.js"):
            text = (SRC / page).read_text(encoding="utf-8")
            for match in re.finditer(r'invoke\("(\w+)"(?:,\s*\{(.*?)\})?\)', text, re.S):
                cmd, args = match.group(1), match.group(2) or ""
                if cmd not in commands:
                    continue
                for required in commands[cmd]:
                    camel = re.sub(r"_(\w)", lambda m: m.group(1).upper(), required)
                    self.assertRegex(
                        args,
                        rf"\b{camel}\b",
                        f"{page}: {cmd} needs '{camel}'",
                    )


if __name__ == "__main__":
    unittest.main()
