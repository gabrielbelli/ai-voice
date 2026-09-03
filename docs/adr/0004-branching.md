# ADR 0004 — Two long-lived branches: `prerelease` is where work happens, `main` is what was validated

**Status:** accepted
**Date:** 2026-09-03

## Decision

Two branches, both long-lived. No branch per task.

| branch | holds | publishes |
|---|---|---|
| `prerelease` | day-to-day work, committed directly | `:pre`, `:pre-<sha>` |
| `main` | versions validated on real hardware | `:latest`, `:main-<sha>` |

A topic branch is available when a change genuinely needs isolating — a risky
refactor, or two people on one file — but it is the exception, not the routine.
Most work is a commit on `prerelease`.

## Why this shape rather than a branch per task

Because the release boundary already existed and the *names* were the problem.

The remote has always had a `prerelease` branch that publishes `:pre` and never
`:latest`, so pulling the default tag cannot pick up an unvalidated build. What
was confusing is that the local branch feeding it was called `main`:

```
git push origin main:prerelease        # local main IS remote prerelease
```

So "main" meant two different things depending on which side of the remote you
were reading, and `main` locally was simultaneously the working branch and the
name of the validated one. Renaming the local branch to `prerelease` costs
nothing and removes the ambiguity entirely.

A branch per task was tried for about an hour and was more ceremony than this
repository needs. The problem it was reaching for is real, but it is not
solved by branching everything.

## The problem that IS real, and what actually fixes it

Two sessions edited `services/ui/app/static/ui.html` at the same time without
either knowing about the other. It surfaced only because the user mentioned it
in passing. The 12 hunks of one change and 38 of the other could not afterwards
be separated — neither a plain nor a three-way apply would land — so they
shipped as a single commit describing both.

Branching every task would have caught that as a merge conflict. So would the
cheaper habit:

> **Before dispatching work into this repository, check whether the working
> tree is already dirty, and check again before committing.**

`git status` would have shown it at any point. Use a topic branch when two
writers are actually expected; do not pay for that insurance on every commit.

## Rules

1. **Work commits to `prerelease`.** It is what the deploy script builds from,
   so it must stay green: tests pass and the images build.
2. **`main` only ever moves by merging `prerelease` after it has been validated
   on hardware** — deployed to orko, exercised, and seen to work. That is what
   `:latest` promises to anyone pulling it.
3. **A topic branch when isolation is genuinely needed**, named `feat/`,
   `docs/` or `wip/`, and merged with `--no-ff` so the grouping survives.
   `wip/` branches may be broken and must say so in the commit message.
4. **Check the tree is clean before starting and before committing.** The one
   collision this repository has had would have been caught by that alone.

## Consequences

- Local `main` is no longer the working branch. `prerelease` tracks
  `origin/prerelease` directly, so the push is `git push origin prerelease`
  rather than a `main:prerelease` refspec that had to be remembered.
- `origin/main` does not exist yet, because nothing has been through a
  validation pass under this rule. It is created by the first merge that has.
- `deploy.py` builds from whatever is checked out; on this machine that is
  `prerelease`, which is correct — orko runs `:pre`.
- The two branches that prompted the earlier version of this ADR,
  `feat/ui-player-and-controls` and `docs/api-and-glossary-adrs`, are merged
  into `prerelease` and can be deleted.
