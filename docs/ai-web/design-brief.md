# Design brief for ai --web

Design a browser-based chat companion for ai-router, with Windows as the first target. Produce one clickable prototype or equivalent design file from the flow below. Use synthetic prompts, replies, paths, and log excerpts. The primary reviewer is the ai-router maintainer.

## Desktop wireframe

```text
┌──────────────────────────────────────────────────────────────────────────┐
│ ai-router     Workspace: demo-firmware              Quota   Settings     │
├─────────────────┬────────────────────────────────────────────────────────┤
│ + New chat      │ Audio timing investigation                            │
│                 │                                                       │
│ Today           │ You     Check the timing regression in the demo.      │
│ ▸ Audio timing  │ Claude  Baseline found. Next: compare the two runs.    │
│   Parser review │                                                       │
│                 │ Claude reached its quota.                             │
│                 │ [Continue with Codex]  [Restore last request]         │
│                 │                                                       │
│                 ├────────────────────────────────────────────────────────┤
│                 │ Next message → Claude ▾                              │
│                 │ [ Write a message…                              Send ]│
└─────────────────┴────────────────────────────────────────────────────────┘
```

After selecting **Continue with Codex**, keep the transcript and history selection where they were. Focus the composer and update the destination label. The reviewer can open the context panel before sending.

```text
┌─────────────────┬───────────────────────────────┬────────────────────────┐
│ History         │ Same conversation             │ Handoff context    [×] │
│ ▸ Audio timing  │                               │ Workspace demo-firmware│
│                 │ Claude reached its quota.     │ Updated 14:32          │
│                 │ Codex selected for next turn. │                        │
│                 │                               │ Last completed task    │
│                 │                               │ Check baseline timing  │
│                 ├───────────────────────────────┤                        │
│                 │ Next message → Codex ▾        │ Latest turn failed     │
│                 │ [View handoff context]        │ Verify work before     │
│                 │ [ Draft…                Send ]│ retrying               │
└─────────────────┴───────────────────────────────┴────────────────────────┘
```

## Prototype frames

1. **Claude working.** Show provider attribution, the Stop control, and, if included, a collapsible tool activity row.
2. **Claude quota exhausted.** Show the error and **Continue with Codex**. Include a generic failure variant without a quota claim.
3. **Codex selected.** Show focus in the composer, the retained draft, and the optional context drawer. Include an empty state when a checkpoint is unavailable.
4. **Codex continues.** After explicit Send, show the Codex reply in the same transcript. Add an approval overlay as a separate state with action details and equally legible accept/reject controls.

## Interaction and visual direction

Use a quiet desktop utility layout. Put visual emphasis on the composer destination and the primary action in the current state. Use provider names in text and keep color secondary. Render code and paths in monospace, with long paths wrapping or expandable in the context panel. Show only the quota information needed to make a provider decision.

Design at 1440×900 and check the core flow at 960×700. On the narrower layout, use a drawer for history or context where needed. Keep visible keyboard focus, sensible tab order, and Esc dismissal in dialogs. Use clear labels on icon-only controls and expose state changes such as the selected provider and a failed turn to assistive technology.

## Review deliverables

- A shareable clickable walkthrough for the four frames and the approval state.
- A short note identifying the components, responsive behavior, and interactions.
- A walkthrough that answers three questions: Which provider gets my next message? What context goes with it? Has anything already executed?

The prototype does not need a running ai-router instance. A local browser prototype is sufficient. A hosted demo, if wanted for review, should use fixtures.
