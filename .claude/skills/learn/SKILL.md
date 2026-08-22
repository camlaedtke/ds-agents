---
name: learn
description: Take a deliberate learning break on a concept this project depends on. Use when the user says they want to understand something, asks "what is X", says they are lost or want to slow down, wants to review flagged concepts, or asks to flag something for later. Also use proactively when a session is about to make a design decision that rests on a load-bearing concept still marked `flagged` in docs/LEARNING.md.
---

# Learning break

The ledger is `docs/LEARNING.md`. Every concept this project leans on gets an entry there so that
not understanding something never stops a session. This skill is how an entry gets closed.

## Modes

**`/learn`** (no argument) — Read `docs/LEARNING.md`. Show the `flagged` entries as a short list,
newest and most load-bearing first. Recommend exactly one and say why in a sentence, tied to what
the project is about to do next per `docs/PLAN.md` and `docs/NEXT.md`. Then wait.

**`/learn <slug>`** — Do the deep dive. See below.

**`/learn flag <thing>`** — Append a new entry in the file's documented format and stop. No
explanation. This is the low-friction path: it costs one line and defers the whole cost.

**`/learn status`** — One table: slug, priority, status. Nothing else.

## Doing a deep dive

Non-negotiable: **explain it against this repository, not in the abstract.** A generic definition
is a failure of this skill. The user can get that anywhere.

1. Find where the concept actually touches this repo. Delegate the search — spawn `Explore`
   (Haiku) with the concept and ask for the files, functions, and doc paragraphs that depend on
   it. Do not read the whole tree yourself.

2. Explain in this order, and stop at the first point where the user says they have enough:
   - **The one-sentence version.** No jargon.
   - **What breaks here if you get it wrong.** Concrete and project-specific: which node, which
     metric, which eval result becomes a lie.
   - **The real mechanism**, using code from this repo as the example. If nothing is built yet,
     use the draft contract in `docs/ARCHITECTURE.md` and say plainly that it is a draft.
   - **The part you can keep delegating.** Name it explicitly. The point of this project is
     to delegate a lot; the user should leave knowing what they chose not to learn.

3. Offer one check: a question, or a two-minute thing they could run or read. Optional, and say so.

4. Update the entry in `docs/LEARNING.md`: status `flagged` -> `explained`, and add a
   `- Covered: <date>, <one line on what was actually understood>` line. If the user says they
   could rebuild it, use `owned` instead.

## Rules

- Never let a learning break turn into building. If the dive surfaces a code change, write it into
  `docs/NEXT.md` and move on.
- If a load-bearing entry is still `flagged` and the session is about to bet a design decision on
  it, say so once, in one sentence, and offer the dive. If the user declines, proceed and note the
  assumption in `docs/DECISIONS.md`. Do not ask twice.
- Plain language. This repo's docs have no marketing tone and neither does this.
