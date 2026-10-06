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

Gemini and Copilot are supported but off by default. To use them, add them to `providers` and
`routing.free` in `airouter/config.toml`:

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
| `ai --style` | How long and padded the chat answers are, per provider |
| `ai --quota` | Claude and Codex usage left in the 5-hour and weekly windows |

In the chat, Enter sends, Shift+Enter adds a line, Esc or ■ stops an answer, and Ctrl+O picks a
model. Alt+V pastes a screenshot, and 📎 attaches a file. A pasted or dragged file path stays
plain text. Type `/help` for the rest.

## How it decides

1. Keywords first. "kenapa", "refactor" mean heavy; "bug", "fix" mean medium; "apa itu"
   means light.
2. No keyword? A small local model ([Granite Embedding 97M](https://huggingface.co/ibm-granite/granite-embedding-97m-multilingual-r2))
   guesses. If it's not sure, it asks Haiku.
3. Each tier goes to Claude or Codex, whichever I've used less today. Free providers, when
   turned on, go first for light and medium.
4. If one fails, the next one takes over.
5. In the chat, a heavy answer is held back and reviewed by a fresh run of the same model. It gets
   revised until the review passes it, for at most 3 rounds, or until the model has less than 20% of
   its 5-hour quota left. Only the final version is shown.

The local model learns as you go. It learns from Haiku's answers, from random spot checks,
and from you: switching the model after an answer counts as a correction.
Every 25 new prompts, Sonnet labels them in the background. `ai --learn` does it right away.

## How answers stay short

Every chat message carries a short style note: answer first, no padding, plain words. Each answer is
scored for padding (praise openers, "semoga membantu" closers, restating the question, long answers to
short questions) and for how you react next ("singkat aja", "maksudnya?", or stopping it with Esc).
After 30 answers, Sonnet rewrites the note from the worst ones. The new note is tried on the next 30
answers and kept only if it scores better; otherwise the old one comes back. The note lives in
`logs/style.md`, and you can edit it yourself. `ai --style` shows the scores.

## Good to know

- In the chat, edits and commands run without asking. Each CLI's own safety settings still
  apply.
- Codex command approval requests appear in the chat with one-time accept and reject buttons.
  Restart `ai` after updating to enable this dialog; console-only clients decline these requests.
- Messages sent while Claude or Codex is replying join the active work. A message arriving
  after the turn has ended starts a follow-up reply. Model-switch commands take effect between turns.
- Claude and Codex chat sessions include the current project and the ai-router installation folder as writable
  locations. Restart a session to pick up changes to its directory permissions.
- Free is slower: Claude takes about 2 s, Codex 5 s, Copilot up to 37 s, Gemini up to 50 s.
- Settings live in `airouter/config.toml`. Every route is logged to `logs/routes.jsonl`.

## License

MIT
