---
name: explain-visual
description: Build a standalone interactive HTML explainer for a concept in this project, traced through a concrete example from the actual code. Use whenever the user says they are confused, lost, or overwhelmed, asks what something is or how it works, asks to be reminded what we are building, says "explain", "walk me through", "I don't get", or flags an entry in docs/LEARNING.md. Also use proactively when a session is about to build on a concept the user has never had explained. Do not use for writing production code.
---

# Visual explainer

The user learns from concrete traced examples and interactive visuals, not abstract description.
They are not a software engineer and have ADHD. A wall of prose is a failed explanation here.

Output is a standalone HTML file in `docs/explainers/`, opened in the browser. Never explain
a concept in chat prose when this skill applies; build the file.

## 1. Ground it in this repo, not in general

Before writing anything, read the actual code the concept lives in. `src/ds_agents/state.py` for
anything about the contract, `nodes/` for node behaviour, `evals/` for metrics. Explain what the
code *does*, not what the concept means in a textbook. If the code contradicts the docs, say so;
that is a finding, not a distraction.

Trace one concrete example the whole way through. Default to the toy dataset
(`tests/fixtures/toy/`) with its planted `status_code` leak, because the user already knows it.
Real column names and real numbers. Never `foo`, `bar`, or "some feature".

## 2. Pick a shape

| the confusion is | build |
|---|---|
| what happens in what order | a stepper: one panel per step, back/next, highlight what changed |
| what does this thing do | before/after panes: state going in, state coming out, new fields highlighted |
| why is it built this way | a toggle: correct version vs the naive version, so the failure is visible |
| what do these numbers mean | a scenario picker: 3 to 4 named situations, same metric row recomputed |
| what does this code do | clickable code lines, plain-language explanation panel below |

Prefer showing a wrong version next to the right one. The user understands a design decision
best when they can watch the naive version break.

## 3. Rules for the file

- One self-contained `.html`: inline CSS and JS, no build step, no CDN, no network.
- Works offline, opens with a double click.
- Dark background (`#1a1a18`), light text (`#f0eee6`), mono font for anything that is code or state.
  Accent colours: teal `#5DCAA5` for normal flow, coral `#D85A30` for the reviewer and for wrong
  versions, amber `#E8A94B` for state and for newly added fields, purple `#AFA9EC` for the harness.
- Highlight what is NEW at each step. That is the whole pedagogical trick.
- Max ~3 interactive controls. More is noise.
- Every visual gets one plain-language sentence above it saying what to look at.
- No jargon without a parenthetical the first time: "a reducer (the rule LangGraph uses to merge
  a node's return value into the state)".

## 4. Naming and follow-through

Save as `docs/explainers/<slug>.html` where slug matches the `docs/LEARNING.md` entry if one
exists (`langgraph-reducers.html`). Then:

1. Update that LEARNING.md entry: status `flagged` -> `explained`, add a `Covered:` line with the
   date, one sentence on what the explainer shows, and the filename.
2. If the concept had no LEARNING.md entry, add one, already marked explained.
3. If building the explainer surfaced a real bug or a doc/code mismatch, add it to `docs/NEXT.md`.
   Do not fix it here; this skill does not touch `src/`.
4. Tell the user the file path and one line on what to click. Nothing else.

## 5. End by checking, not by summarising

Close with a single question that tests whether it landed: ask them to explain one specific piece
back in their own words, or ask which part is still fuzzy. One question. Do not recap the explainer
in prose underneath it, which defeats the point.

## Anti-patterns

- Explaining in chat and offering to build a file. Just build the file.
- A generic tutorial that would work for any LangGraph project.
- Placeholder data. If you do not know a real value, read the code or run the toy pipeline.
- Six controls and a settings panel. This is a teaching aid, not an app.
- Ending with "let me know if you have questions".
