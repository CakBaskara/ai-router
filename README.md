# ai-router

I didn't want to burn Opus quota on "what's the git command for X". So `ai` looks at each
prompt, guesses how hard it is, and sends it to the cheapest model that can handle it:
Haiku or a small Codex model for small stuff, Opus or a big Codex model for real work.

```text
> ai "apa itu decorator di python?"
→ light · claude haiku (low) · light: apa itu

> ai "kenapa frame ISO hilang tiap 10 detik? ini lognya ..."
→ heavy · codex gpt-6.1-sol (high) · heavy: kenapa
```

Built and used on Windows 11.

## Install

You need Python 3.11+ and at least one of these CLIs, logged in:

- Claude Code: `npm i -g @anthropic-ai/claude-code`
- Codex: `npm i -g @openai/codex`

```powershell
git clone https://github.com/CakBaskara/ai-router
cd ai-router
py -m pip install -e .
```

## Use

| Command | What it does |
|---|---|
| `ai` | Open the chat |
| `ai "..."` | Ask once, get one answer |
| `ai -t heavy "..."` | Force a tier: `light`, `medium`, `heavy` |
| `ai -p codex "..."` | Force a provider |
| `ai -n "..."` | Show the route only, don't run it |
| `ai --ml` | See how the router's model is doing |
| `ai --style` | How long and padded the chat answers are, per provider |
| `ai --quota` | Claude and Codex usage left in the 5-hour and weekly windows |
| `ai --migrate-data` | Preview copying runtime data from the old installation |

In the chat, Enter sends, Shift+Enter adds a line, Esc or ■ stops an answer, and Ctrl+O picks a
model and effort in one panel: ↑/↓ and Enter pick the model, ←/→ or a click on the slider set the effort,
and the panel stays open until Esc. Ctrl+E and clicking the model chip open the same panel; `/effort high` and
`/effort auto` work too. A pinned effort is fitted to the levels each model supports. Alt+V pastes a screenshot, and 📎 attaches a file. A pasted or dragged file path stays
plain text. Type `/help` for the rest.

While a model works, its reply shows only a loading line with the elapsed time; when it is done, only its
final answer appears, the text after its last tool call. Set `show_work = true` under `[chat]` to stream tool
calls and text as they happen instead.

The quota in the top bar is re-read every 3 seconds. Claude and Codex usage is also polled every minute and whenever a
reply starts or ends, so both providers stay current whichever one answers, and use from other apps shows up. Claude usage comes from the same endpoint Claude Code's `/usage`
reads, with the login in `~/.claude/.credentials.json`.

Links to local files in chat open with their associated application; web links open in the default browser.

## How it decides

1. Keywords first. "kenapa", "refactor" mean heavy; "bug", "fix" mean medium; "apa itu"
   means light.
2. No keyword? A small local model ([Granite Embedding 97M](https://huggingface.co/ibm-granite/granite-embedding-97m-multilingual-r2))
   guesses. If it's not sure, it asks Haiku.
3. Each tier goes to Claude or Codex, whichever I've used less today. Free providers, when
   turned on, go first for light and medium.
4. If one fails, the next one takes over.
5. In the chat, an answer from a top model (Claude Fable 5 or newer, Codex GPT-6-Astra or newer) is
   held back and reviewed by a fresh run of the same model. It gets
   revised until the review passes it, for at most 3 rounds, or until the model has less than 20% of
   its 5-hour quota left. Only the final version is shown.

The local model learns as you go. It learns from Haiku's answers, from random spot checks,
and from you: switching the model after an answer counts as a correction.
In chat, 10% of confident ML predictions are checked by Haiku in the background, including
follow-up messages. The check uses the ML prediction before the chat's minimum or retained
tier is applied; pinned tiers, keyword decisions, and `--no-llm` skip these checks.
Every 25 new prompts, Sonnet labels them in the background. `ai --learn` does it right away.

## How answers stay short

Every chat message carries a short style note: answer first, no padding, plain words. Each answer is
scored for padding (praise openers, "semoga membantu" closers, restating the question, long answers to
short questions) and for how you react next ("singkat aja", "maksudnya?", or stopping it with Esc).
After 30 answers, Sonnet rewrites the note from the worst ones. The new note is tried on the next 30
answers and kept only if it scores better; otherwise the old one comes back. The note lives in
`style.md` in the runtime logs directory, and you can edit it yourself. `ai --style` shows the scores.

## Good to know

- In the chat, edits and commands run without asking. Each CLI's own safety settings still
  apply.
- Codex command approval requests appear in the chat with one-time accept and reject buttons.
  Restart `ai` after updating to enable this dialog; console-only clients decline these requests.
- Messages sent while Claude or Codex is replying join the active work. A message arriving
  after the turn has ended starts a follow-up reply. Model-switch commands take effect between turns.
- Claude and Codex chat sessions use the current project, explicitly configured `chat.dirs`, and attachment
  storage. The ai-router installation folder is no longer added automatically. The bundled `chat.dirs` is empty;
  a migrated configuration keeps its existing directories. Restart sessions to pick up changes to permissions.
- Settings use the bundled `airouter/config.toml` as defaults, with user settings overlaid as described below.

## Runtime data and configuration

Runtime data is stored outside the installed package. On Windows the defaults are:

| Files | Location |
|---|---|
| User configuration | `%APPDATA%\AI-Router\config.toml` |
| Routes, labels, quota, reply style and learning state | `%LOCALAPPDATA%\AI-Router\logs\` |
| Attachments | `%LOCALAPPDATA%\AI-Router\attachments\` |
| Downloaded embedding models | `%LOCALAPPDATA%\AI-Router\models\` |

Linux and macOS use the corresponding per-user configuration and data directories from `platformdirs`.
Paths are only created when writing. Bundled configuration and seed labels remain package assets.

Configuration selection is `AI_ROUTER_CONFIG`, if set, then the user configuration, if present, then the
bundled defaults. Missing keys in the selected configuration inherit bundled defaults; arrays replace the
default array. An explicit missing or invalid configuration reports an error instead of silently falling back.
The TUI watches both the defaults and user configuration for changes.

| Environment variable | Overrides |
|---|---|
| `AI_ROUTER_CONFIG` | Selected configuration file |
| `AI_ROUTER_DATA_DIR` | Runtime data root, including logs, attachments and models |
| `AI_ROUTER_LOG` | Routes file; quota, replies and style remain beside this file |
| `AI_ROUTER_LABELS` | Labels file; learning settings, vectors, thresholds and audit remain beside this file |
| `AI_ROUTER_ATTACH` | Attachment directory |
| `AI_ROUTER_MODELS` | Models directory; each model retains its own subdirectory |

Set `AI_ROUTER_DATA_DIR` before migration or starting `ai` to keep large runtime files on another drive.
Individual path overrides take priority over this root. The selected overrides must also be present in future
sessions. This change does not move provider-owned Claude or Codex session storage.

Stop running `ai` sessions before migrating. Preview first:

```powershell
ai --migrate-data
```

For a different old checkout, pass its logs directory with `--migrate-from "path\to\old\ai-router\logs"`.
The preview lists every source and destination without creating directories or writing data. It includes the
old sibling `airouter/config.toml` as user configuration unless `AI_ROUTER_CONFIG` already selects a file.
After reviewing the destinations, use the same options plus `--apply`:

```powershell
ai --migrate-data --apply
```

Migration copies files and preserves originals as a backup. It never overwrites a destination: identical
files are left alone, and differing files are reported as conflicts. Conflicts or inspection errors prevent
copying from starting. If copying fails partway through, completed files remain; resolve the error and rerun
the command. Repeating a completed migration does not change files. Locks, partial downloads, symlinks and
Windows reparse points are skipped. Resolve conflicts manually; source files are never deleted automatically.

## License

MIT
