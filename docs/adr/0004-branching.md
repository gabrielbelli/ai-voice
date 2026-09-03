# ADR 0004 — Work happens on branches; `main` is not a scratchpad

**Status:** accepted
**Date:** 2026-09-03

## Decision

Work goes on a branch and reaches `main` as a reviewed unit. `main` is not
where a feature is figured out.

```
feat/<thing>     a feature or a fix
docs/<thing>     documentation and ADRs, when they stand alone
wip/<thing>      unfinished work parked so an interrupted session survives
```

## Why, from what actually happened

The web UI, its five follow-up fixes and four ADRs all landed straight on
`main`, and two of those commits exist only because the previous one was wrong:

- a paste handler that pasted twice, fixed the commit after it shipped
- a download URL missing its folder, fixed the commit after that
- a deploy that raced CI and ran the *previous* image, so a fix already on
  `main` was reported as unfixed

That is a normal amount of back-and-forth for a feature being built live. It is
not a good permanent record. On a branch it is one merge; on `main` it is the
history everyone reads.

There is a second, sharper reason. Two sessions edited
`services/ui/app/static/ui.html` at the same time without either knowing about
the other, and the collision was found only because the user mentioned it. Two
branches would have made that a merge with a conflict — visible, mechanical,
and impossible to miss — instead of two sets of uncommitted changes silently
interleaved in one working tree.

## How it maps onto the release layout, which is unusual here

The remote has **no `main`**. Local `main` publishes to the remote's
`prerelease` branch:

```
git push origin main:prerelease
```

That is deliberate and predates this ADR: `prerelease` publishes images tagged
`:pre` and never `:latest`, so pulling the default tag cannot pick up an
unvalidated build. The remote's `main` is reserved for versions that have been
validated on real hardware.

So the flow is:

```
feat/x  ->  main (local)  ->  origin/prerelease  ->  :pre images  ->  orko
                                                        |
                                            validated -> origin/main -> :latest
```

CI runs on `prerelease` and on pull requests, so a branch pushed as a PR is
built and tested without publishing anything.

## Rules

1. **Anything more than a one-line fix starts on a branch.** The exception is a
   correction to something already on `main` that would otherwise leave it
   broken — a green tree matters more than a tidy one.
2. **A branch is merged, not rebased away.** The back-and-forth is real history
   and belongs in the branch; the merge is what `main` reads.
3. **`main` must always be deployable.** It is what `deploy.py` builds from.
4. **`wip/` branches are allowed to be broken** and must say so in the commit
   message. `wip/web-ui` is one: an interrupted workflow's output, parked
   rather than lost.

## Consequences

- The four commits that prompted this were retroactively split onto
  `feat/ui-player-and-controls` and `docs/api-and-glossary-adrs`, and `main`
  was reset to `origin/prerelease`. Nothing was lost; both branches carry their
  work and this ADR arrived on one of them.
- A branch that is never merged is worse than no branch. `wip/web-ui` is
  already close to that line and should be either merged or deleted once the
  UI work settles.
