# LLM head-to-head benchmark — compliance note

The head-to-head comparison against a general LLM used real ADNI participants.
Because the ADNI Data Use Agreement restricts redistribution of
participant-level data, the per-patient artifacts (prompts, predictions,
ground-truth follow-ups) have been **removed from this repository and from its
git history**. They are retained privately by the author.

What remains here are aggregate-only results (`comparison_result.json`,
`cell_tally.json`) with no participant identifiers. The benchmark is fully
reproducible locally: `python -m biojepa.hard_tests2` and the 97-patient batch
builder regenerate equivalent artifacts from a local ADNI copy.
