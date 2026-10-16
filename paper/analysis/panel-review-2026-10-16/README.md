# Simulated peer review of the paper

## Material Passport

- Origin skill: academic-research-skills / academic-paper-reviewer v1.11.1 (ARS 3.23.0), mode `full`
- Contract: `reviewer/reviewer_full/v2` (sprint contract, paper-blind Phase 1 then paper-visible Phase 2)
- Date: 2026-10-16
- Reviewed: `paper/main.tex` at commit `2439f39` (unchanged on `main` at `ac04bac`)
- Verification: contract validation, five Phase 1 and five Phase 2 conformance checks, panel layer-1 check, synthesis recomputation (`check_panel_synthesis.py`, exit 0) and provenance replay all passed.
- Calibration: `NOT_CALIBRATED`. No target venue was bound (`criteria_binding_unavailable`), so there is no venue-fit verdict.
- Scope: a simulated review by model agents, not human peer review.

**Decision: Major Revision.** The decision is mechanical: condition F2 fired because four mandatory dimensions (methodology, domain accuracy, argumentative coherence, contribution) scored block. Every block is marked repairable, no fatal block was declared, and the devil's-advocate CRITICAL table is empty. The core 13.4-point calibration result re-derives from the released records.

## Files

| File | Content |
|---|---|
| [synthesis.md](synthesis.md) | Editorial Decision Package: decision letter, consensus and disagreements, three blocking issues, revision roadmap (21 required items R1–R21, 39 suggested S1–S39), reviewer summaries, acronym-check attachment |
| [phase0_field_analysis.md](phase0_field_analysis.md) | Field analysis and the four reviewer configuration cards |
| `<seat>.phase1.md` | Paper-blind scoring plans (each seat's pre-commitment) |
| `<seat>.phase2.md` | Paper-visible reviews: `eic` journal fit, `methodology` R1, `domain` R2, `perspective` R3, `da` devil's advocate |
| [contract.json](contract.json), [metadata.json](metadata.json) | The contract as injected, and the Phase 1 metadata envelope |
| [provenance_input.json](provenance_input.json), [provenance.json](provenance.json) | Typed panel provenance (six axes, no binary independence claim) |

## Process caveats

- All seats ran on one model family and provider, so their errors may be correlated.
- Phase 1 blindness was partial: the harness injects the project's `CLAUDE.md` and the user's memory index into every agent. A transcript audit found no repository access during Phase 1 and no seat reading another seat's card.
- Seats ran the conformance linter on their own drafts before submitting (format fixes only).
- Reviewers could read `paper/results/` and git history; R2 also used web sources.

## Post-review note

EIC-W18 and roadmap item S10 say `recency_collision.json` is missing from the versioned record. It is versioned at [`paper/analysis/accuracy-review-2026-10-13/recency_collision.json`](../accuracy-review-2026-10-13/recency_collision.json); the reviewers checked only `paper/results/`. The remaining fix is to give that path where §7.13 cites the file. The review files themselves are unedited.
