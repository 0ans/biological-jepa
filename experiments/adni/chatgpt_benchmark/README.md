# LLM head-to-head benchmark — compliance note

The head-to-head comparison against a general LLM used real ADNI participants.
Because the ADNI Data Use Agreement restricts redistribution of
participant-level data, the per-patient artifacts (prompts, predictions,
ground-truth follow-ups) have been **removed from this repository and from its
git history**. They are retained privately by the author.

What remains here are aggregate-only results (`comparison_result.json`,
`cell_tally.json`) with no participant identifiers. Note that the committed
aggregates are from an **n=6 demonstration run** (`comparison_result.json`
itself says "not a statistical benchmark"); the batched-prompt builder lived
with the participant-level artifacts and is retained privately by the author —
it is not part of this repository. `python -m biojepa.hard_tests2` regenerates
our models' side of the clinical comparison (T1–T6) from a local ADNI copy,
but no LLM prompts are built or shipped here.
