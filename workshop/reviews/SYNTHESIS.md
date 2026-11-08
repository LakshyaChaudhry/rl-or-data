# Synthesis

Four reviewer agents worked independently, then each re-read the revised 7-page PDF. This note merges those reports. It does not add a fifth simulated reviewer.

## Agreement

- The printed accuracies, seed rows, paired intervals, budgets, and the three newly added truncation bounds match the logs the reviewers opened. No reviewer found a fabricated result.
- The curated 10.9 point gap was being asked to do too much work in the first draft. The revised abstract and conclusion say it is truncation-flagged and that the registered sampling budget did not run. Reviewers A and B called that lead-sentence problem fixed.
- Curation's failure stays visible. The paper does not claim to contradict Xiong et al. or Samineni et al.
- Scope stays at one counting task, Qwen3-4B-Base, LoRA, and three seeds.
- NeSy-FM fit is weak. Reviewer B, on the second pass, still says a NeSy-FM reviewer would have grounds to call the paper off-topic. The PDF does not claim a neuro-symbolic contribution and does not name a venue.
- Format: after the caption move, Reviewer D's major defect is fixed. Both PDFs are inside the official-list 8-page cap. The preserved arXiv PDF hash is unchanged.
- Adapters and the shared RFT sample files are not publicly obtainable. Reviewer C left that as an open blocker for artifact checks. The PDF states the private-adapter status instead of claiming a release.

## Disagreement, and how it was resolved

Reviewer A wanted the mechanical "yes" marks qualified. The table caption now says a "yes" does not override the truncation flag or the adverse bounds, and the body prints the bounds that knock out the 3.7 and 3.2 point rows. Reviewer A still notes that three-seed standard deviations are noisy. The paper already says the rule is not a formal test. It does not add a new variance formula that was not in the preprint.

Reviewer B wanted the title changed. The title no longer asks "Is it RL or Data?" Reviewer B's second pass withdraws that item as a blocker and says the distinctness ceiling is the scoped measurement the limitations already describe.

Reviewer D and the editor agree the anonymous PDF is conditional. It is not the file to send anywhere until a venue states double-blind review. Its footer forbids public sharing because that is what `\usepackage[submission]{aaai2027}` prints.

## What the second pass left open, and what was edited afterward

The second pass still flagged "refreshes three times" in the introduction and limitations against "twice" in the setup. Those sentences now say the secondary arm has three rounds and resamples the current policy twice. Section headings no longer say mixed data "helps" or that three rounds "recover" 30%. The abstract now flags the 9.2 point mixed pair and gives the iterated-RFT adverse bound of +0.1 [-3.0, +3.2]. Those edits were compiled after the second pass. The reviewers did not re-score them. They do not add numbers.

Still open, and accepted as limits rather than silent fixes:

- The full 27-cell validation grid, generator operator list, decoration-stripping rules, clip epsilon, and compute hours remain in the preprint and the repo, not all in these 7 pages.
- The public repository does not contain the adapter weights or the shared RFT jsonl files.
- A street address is not in the preprint and was not invented.
- Page 4 is tight because two full-width tables sit on it. There is no overfull line.

## Gate

Editorial readiness: usable as a provisional condensation.

Submission readiness: not ready. See `workshop/aaai27-provisional/READINESS.md`.
