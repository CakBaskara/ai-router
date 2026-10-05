# ai-router

I didn't want to burn Opus quota on "what's the git command for X". So `ai` looks at each
prompt, guesses how hard it is, and sends it to the cheapest model that can handle it:
free Gemini or Copilot for small stuff, Claude Code or Codex for real work.

```text
> ai "apa itu decorator di python?"
→ light · gemini gemini-3.8-flash (low) · light: apa itu

> ai "kenapa frame ISO hilang tiap 10 detik? ini lognya ..."
→ heavy · codex gpt-6.1-sol (high) · heavy: kenapa
```

Built and used on Windows 11.

## Install

You need Python 3.11+ and at least one of these CLIs, logged in:

- Claude Code: `npm i -g @anthropic-ai/claude-code`
- Codex: `npm i -g @openai/codex`
- Gemini: `npm i -g @google/gemini-cli`, with a free key from aistudio.google.com/apikey in `GEMINI_API_KEY`
- Copilot: `npm i -g @github/copilot`

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

In the chat, Enter sends, Shift+Enter adds a line, Esc stops an answer, and Ctrl+O picks a
model. Alt+V pastes a screenshot. You can also click 📎 or drag a file in. Type `/help` for
the rest.

## How it decides

1. Keywords first. "kenapa", "refactor" mean heavy; "bug", "fix" mean medium; "apa itu"
   means light.
2. No keyword? A small local model ([Granite Embedding 97M](https://huggingface.co/ibm-granite/granite-embedding-97m-multilingual-r2))
   guesses. If it's not sure, it asks Haiku.
3. Light and medium go to the free providers first. Heavy goes to Claude or Codex,
   whichever I've used less today.
4. If one fails, the next one takes over.

The local model learns as you go. It learns from Haiku's answers, from random spot checks,
and from you: switching the model after an answer counts as a correction.
`ai --ml-train` lets Sonnet label your old prompts. I run it every evening.

## Good to know

- In the chat, edits and commands run without asking. Each CLI's own safety settings still
  apply.
- Messages sent while Claude or Codex is replying join the active work. A message arriving
  after the turn has ended starts a follow-up reply. Model-switch commands take effect between turns.
- Claude and Codex chat sessions include the current project and the ai-router installation folder as writable
  locations. Restart a session to pick up changes to its directory permissions.
- Free is slower: Claude takes about 2 s, Codex 5 s, Copilot up to 37 s, Gemini up to 50 s.
- Settings live in `airouter/config.toml`. Every route is logged to `logs/routes.jsonl`.

## License

MIT
