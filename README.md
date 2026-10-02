# ai-router

One terminal command that sends each prompt to the cheapest model that can answer it,
through the official CLIs of four providers: Gemini and GitHub Copilot (free tiers) for light
work, Claude Code and Codex (Claude Pro, ChatGPT Plus) for heavy work.

## Install

Needs Python 3.11+ and any of these CLIs, installed and logged in:

| Provider | Install | Login |
|---|---|---|
| Claude Code | `npm i -g @anthropic-ai/claude-code` | `claude`, then sign in |
| Codex | `npm i -g @openai/codex` | `codex login` |
| Gemini | `npm i -g @google/gemini-cli` | API key from aistudio.google.com/apikey in `GEMINI_API_KEY`; Google sign-in no longer works for the CLI's free tier |
| Copilot | `npm i -g @github/copilot` | `copilot login` (or an existing `gh` login) |

Remove a provider you do not use from `providers` in `airouter/config.toml`.

```powershell
git clone https://github.com/CakBaskara/ai-router
cd ai-router
py -m pip install -e .
```

This puts `ai` (`ai.exe` on Windows) in the Python `Scripts` folder.

## Use

| Command | What happens |
|---|---|
| `ai` | Chat in the terminal; every message is routed (see below) |
| `ai -t heavy` / `ai -p codex` | Chat with the tier or provider locked |
| `ai "apa itu decorator?"` | One-shot answer, then back to the shell |
| `Get-Content err.log \| ai` | Prompt from stdin |
| `ai -i "tambah test untuk PERIOD_API"` | Opens an interactive Claude Code session in the current folder with the chosen model |
| `ai -i` | Interactive session, medium tier |
| `ai -t heavy "..."` | Force a tier: `light`, `medium`, `heavy` |
| `ai -p codex "..."` | Force a provider: `gemini`, `copilot`, `claude`, `codex` |
| `ai -n "..."` | Dry run: show the route, run nothing |
| `ai --no-llm "..."` | Never ask a model to classify |

## How a route is chosen

```
prompt
  │
  ├─ rules (free): keywords, length, code or log, -i adds one point
  │     score ≤ 0 → light · 1–2 → medium · ≥ 3 → heavy
  │
  ├─ no keyword matched → ask Haiku for one word (≈1.7k tokens, ≈5 s)
  │
  ├─ tier → providers, from config.toml
  │     light, medium → gemini, copilot first (free), then claude, codex
  │     heavy         → claude opus / codex gpt-6.1-sol first, free ones last
  │     the paid pair takes turns: the one used less today goes first
  │
  └─ a failure or exhausted quota moves to the next provider.
     Interactive sessions use the first provider only.
```

One-shot runs are read-only and leave no saved session
(`claude --no-session-persistence`, `codex exec --ephemeral -s read-only`).

## Chat

`ai` with no prompt opens a full-screen chat (Textual) in the current folder; `ai --plain`
gives the line-by-line version. Each message goes through the same rules, with these differences:

| Rule | Why |
|---|---|
| Chat never goes below `[chat] min_tier` (light) | Raise it to `heavy` to send every message to Opus or gpt-6.1-sol |
| The first message to each provider session starts with a short note about the router | Without it the model says it cannot switch models and does not know `/models` |
| The tier only goes up within a conversation; `/new` resets it | A model switch rewrites the whole context into cache, so dropping back for a short follow-up costs more than staying |
| Haiku is asked only for the first message; later ones use the local classifier alone | Follow-ups like "lanjut" carry no keywords and keep the current tier |
| The provider sticks within a conversation, except that a heavy message leaves a free provider | Switching back and forth would resend the recap each time |

Claude runs as one long-lived process per model (`--input-format stream-json`), started
before the first message, so a turn does not pay Claude Code's startup each time. Codex has no
such mode and starts per message. Sessions are resumed (`claude -p --resume`,
`codex exec resume`), so each provider keeps its own history. When a message goes to a provider that missed some turns, those turns are sent
first as a short recap (last 12 entries, 2000 characters each).

Edits and shell commands run without asking. Claude runs with `--permission-mode auto`, the
same mode as an interactive Claude Code session in auto mode: a safety classifier still blocks
risky actions. Codex runs with `-s workspace-write`, so its commands stay inside the folder.
Copilot runs every tool except a deny list (git push/commit/reset/clean/checkout, `gh repo`,
`gh pr`, `npm publish`, and file deletion); its `--assisted-approval` judge refuses even safe
commands when nobody can answer, so it is not used. Gemini runs with `--approval-mode auto_edit`:
it edits files but cannot run shell commands, because its `tools.exclude` command filter did not
block anything in yolo mode when tested.

All four read the same rules. `ai` copies `~/.claude/CLAUDE.md` to `~/.gemini/GEMINI.md` and
`~/.copilot/copilot-instructions.md` on every start, so edit only the first. Codex keeps its own
`~/.codex/AGENTS.md`. In a repository, Copilot reads `CLAUDE.md`/`AGENTS.md`, and Gemini is set
to read them too (`context.fileName` in `~/.gemini/settings.json`).

Measured speed for a short answer: Claude about 2 s (long-lived process), Codex about 5 s,
Copilot 8 to 37 s, Gemini 14 to 50 s. Free-first saves quota at the cost of waiting.

The full-screen chat uses the VS Code Dark Modern palette. On Windows, Shift+Enter adds a line:
Textual drops the Shift state of Enter, so the app reads it from the console event first. This
works in Windows Terminal and the classic console; the VS Code terminal sends a plain Enter, so
there use Ctrl+J or a trailing `\`. Ctrl+A selects the input, or the whole conversation when
the input is empty; Ctrl+C copies the selection. A prompt sent while an answer is still running
is queued and goes out as soon as that answer ends.

### Attachments

| How | What happens |
|---|---|
| Alt+V | Attaches the image on the clipboard, or the files copied in Explorer. Windows Terminal and the VS Code terminal swallow Ctrl+V when the clipboard holds only an image, so Alt+V is the reliable key; Ctrl+V tries the clipboard first and falls back to pasting text |
| Click 📎 right of the input | Opens the Windows file dialog; pick one or more files |
| Drag a file onto the terminal | The pasted path becomes an attachment instead of text |
| Backspace in an empty input | Removes the last attachment |

Attachments are copied to `logs/attachments` (gitignored, 20 MB per file, BMP/TIFF converted to
PNG) so every CLI is allowed to read them, then sent the way each CLI expects: Claude gets image
blocks in its stream-json input, Codex `-i`, Copilot `--attachment` (images and PDF), Gemini an
`@path` reference with `--include-directories`. Other file types are listed in the prompt for the
model to open with its tools. One image with a number was read correctly by all four.

| Command | Effect |
|---|---|
| `/new` | New conversation |
| `/model` | Show tier, provider, model |
| `/model light\|medium\|heavy` | Lock the tier |
| `/model auto` | Back to automatic routing |
| `/claude`, `/codex` | Lock the provider |
| `/models` | Numbered list: Claude models from `config.toml`, Codex models from `codex debug models` |
| `/model 3`, `/model luna` | Lock the model picked from that list (number, exact name, or a unique part of it) |
| `/codex gpt-5.6-sol` | Lock the provider and that exact model (passed as is, not checked) |
| `/reload` | Reload the code now, keeping the conversation |
| `/exit` | Quit |

Full-screen keys: Enter sends, Shift+Enter or a trailing `\` adds a line, Esc stops the
answer, Ctrl+O opens the model picker (or click the model name top right), Ctrl+N starts a
new conversation, Ctrl+Q quits.

The full-screen chat checks `airouter/*.py` and `config.toml` every 2 seconds. When they change
and no answer is running, it reloads the code and keeps the conversation and sessions. A file
with a syntax error is not loaded; a warning shows instead.

In `--plain`, end a line with `\` to continue on the next one; Ctrl+C stops the current answer.

## Learned classifier

When no keyword matches, a local Naive Bayes model (`airouter/learn.py`, pure Python) guesses
the tier first. It decides alone only when its confidence is above a threshold picked from
held-out results: the lowest confidence at which past labels were still right at least
`ml_target_accuracy` of the time. Otherwise Haiku is asked, and its answer becomes a new label.

Labels come from `airouter/seed_labels.jsonl` (hand-written start set), every Haiku answer, and
corrections in chat: changing the tier or model after an answer labels your last message
with that tier, counted three times. Learned labels live in `logs/labels.jsonl`.

`ai --ml` prints the label count, held-out accuracy, the current threshold and how often the
model decides alone.

## Configure

Edit `airouter/config.toml`: provider order, the model per tier, and the keyword lists.
`AI_ROUTER_CONFIG` points to a different file.

## Log

Every run appends a line to `logs/routes.jsonl` (gitignored): tier, score, reasons,
whether a model classified it, provider, model, exit code, seconds, first 500 characters
of the prompt. `AI_ROUTER_LOG` moves it. Read it to tune the keyword lists.

## Test

```powershell
py -m pip install pytest
py -m pytest -q
```

## License

MIT, see `LICENSE`.
