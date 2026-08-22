---
name: session-wrap
description: End-of-session checklist that leaves the repo and docs in a state the next session can pick up cold. Use whenever the user says they are done, wrapping up, stopping for now, out of time, or asks what to do next session. Also use proactively when a session's stated deliverable is complete.
---

# Session wrap

Goal: the next session starts with zero archaeology. Do all steps, in order, then stop.

## 1. Verify the floor

```
uv run pytest -m fast
uv run ds-agents run --dataset toy
```

If either fails, that is the first item in NEXT.md and the commit message says `[toy broken]`.
Do not spend the rest of the session fixing it unless the user says so.

## 2. Update docs/NEXT.md

Overwrite it with:

```
# Next session

## Start here
One paragraph: where we are, what the last session delivered, what is half-done.

## First prompt
The literal prompt to paste to start the next session. Specific enough to begin without rereading history.

## Open questions
Bullets. Decisions the user needs to make, or things we were unsure about.

## Parking lot
Ideas that came up and were deliberately not pursued.
```

## 3. Update docs/PLAN.md

Check off what was done. If scope changed, say so in one line under the phase.

## 4. Update docs/DECISIONS.md

Any architecture choice made this session that is not already there. One paragraph each:
what was decided, what the alternative was, why.

## 5. Commit

One commit for the session's work if it is coherent, otherwise a few. Message format:

```
<phase>: <what changed in one line>

- bullets on what was added or changed
- any known gaps
```

## 6. Report back

Tell the user in three lines: what shipped, what is open, rough credit usage if known.
Do not summarize the whole session. They were there.
