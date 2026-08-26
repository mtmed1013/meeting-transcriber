# Configurable Whisper settings design

## Goal

Make the initial meeting transcriber configurable without editing Python code.
The Obsidian note directory and the Faster-Whisper model name must come from
environment variables, while preserving the current macOS behavior when no
configuration is supplied.

## Configuration

The application will load a project-local `.env` file using
`python-dotenv`. Variables already present in the process environment take
precedence over values in `.env`.

Supported variables:

- `OBSIDIAN_DIR`: destination directory for meeting notes.
- `WHISPER_MODEL`: Faster-Whisper model identifier, such as `medium`, `small`,
  or `tiny`.

Defaults:

- macOS: the existing iCloud Obsidian directory.
- Windows: `~/Documents/Obsidian`.
- Whisper model: `medium`.

The model value is passed directly to `WhisperModel`; downloading and model
compatibility errors remain visible from the existing startup behavior.

## Files and scope

- Update `meeting_transcriber.py` to load `.env` and read both variables.
- Add `python-dotenv` to `requirements.txt`.
- Add `.env.example` with documented values and platform path examples.
- Update `README.md` with setup instructions and model choices.
- Preserve the initial transcription flow, 30-second block size, speaker-change
  detection, and note format.
- Do not add tests or runtime folders to the repository.

## Verification

- Compile the script successfully.
- Run it with an explicit `OBSIDIAN_DIR` and `WHISPER_MODEL` in the environment
  and verify the resolved configuration at startup without editing the Python
  file.
- Confirm that `.env` values are used when process variables are absent.
- Confirm that the repository remains free of generated runtime and test
  directories.
