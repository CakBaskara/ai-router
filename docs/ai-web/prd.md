# ai-router local web UI

Status: proposal for maintainer review

## Problem

ai-router can continue one conversation across Claude Code and Codex. In a terminal, it can be hard to see which provider will receive the next message, inspect the context being carried over, and recover a draft after a turn fails. The browser UI should make those transitions visible while keeping the conversation in the same workspace.

## Product scope

- Give Windows users an optional `ai --web` entry point that opens a local browser tab for the directory where the command was run.
- Reuse the existing Python chat, routing, history, provider adapters, and permission handling. The terminal remains an available entry point.
- Make one flow work end to end: start with Claude, encounter a quota error, select Codex, inspect available handoff context, and send the next message in the same chat.
- Defer desktop packaging, hosted accounts, cross-device sync, shared skill management, and long-term project memory.

## User flow

1. Run `ai --web` from a project directory. The page shows the workspace and opens a chat with the current routing mode.
2. Select Claude and send a message. Show the actual provider beside each reply and the selected provider beside the composer. In Auto mode, label the destination as Auto until routing selects a provider.
3. If a turn fails with a confirmed quota error, offer **Continue with Codex** if Codex is configured. For an unclassified failure, show the actual error and offer **Switch provider**. Keep the failed turn in the transcript.
4. Switching changes the destination for the next message and focuses the composer. Keep any existing draft. If there is no draft, offer **Restore last request** as a separate action. Sending requires an explicit user action.
5. Offer **View handoff context** beside the composer. Show the recent recap and, if available and enabled, the checkpoint with its timestamp and workspace. Explain that the displayed content is prepared for the receiving provider. If the context changes before send, the preview must be refreshed before presenting it as current.
6. Send to Codex in the same conversation. Show the new provider on the resulting message and preserve the prior Claude transcript. If a tool requests approval, display the action with its scope and explicit accept/reject controls.

## Layout

- Left: new chat, conversation history, and an indication of which workspace owns the active conversation.
- Center: messages with provider attribution, turn status, approval requests, and the failed-turn recovery action.
- Composer: draft, provider or Auto selector, handoff-context link when available, and Send or Stop.
- Right: context panel on demand. Collapse it into a modal or drawer on narrower windows. Keep the composer visible and allow the transcript to scroll independently.

See [the design brief and wireframes](design-brief.md) for the design handoff.

## Local runtime requirements

The browser client should consume structured chat events through a small adapter to the existing Python chat layer. Carry provider and turn identifiers on events so replies, cancellation, and approvals are associated with the correct turn. Start with one active turn per chat. Keep provider credentials and CLI processes in Python. Serve the UI from loopback on the same origin as its API, use a per-launch session secret, and validate the origin of state-changing requests and any streaming connections that can send commands. Use the configured provider policy and surface its current mode. Any decision to change defaults should be reviewed separately. Approval requests must be scoped to the originating turn, and an expired or disconnected approval must resolve as a denial.

Contribute the UI as an optional entry point in the existing repo. Choose a thin browser client for the prototype. Decide whether to use React/Vite and which Python web transport to adopt after the interaction review.

## Acceptance criteria

- In the design prototype, a reviewer can complete the handoff flow and inspect the context entirely with keyboard controls.
- Starting `ai --web` in an implementation opens a tab pointed at the chosen local workspace. The workspace is visible during the entire flow.
- Claude and Codex replies remain attributed to their actual provider. Switching does not clear the transcript, overwrite an existing draft, or start a turn.
- Failed or cancelled turns with possible side effects display a reminder to inspect results before retrying. Restoring a request only fills the composer.
- Test confirmed quota exhaustion, generic failure, no available second provider, unavailable checkpoint, cancellation, and approval acceptance, rejection, and timeout.
- Switching preserves the conversation ID and applies to the next explicit submit. Automated testing can use simulated events, followed by a Windows smoke test with both installed CLIs and a public fixture repo.

## Relation to the checkpoint PR

The design can prototype the full flow with fixture data. The web MVP can use the existing recap. The richer checkpoint panel depends on [PR #1](https://github.com/CakBaskara/ai-router/pull/1) or an equivalent accepted interface. The UI still needs a useful state when checkpoints are disabled or unavailable.
