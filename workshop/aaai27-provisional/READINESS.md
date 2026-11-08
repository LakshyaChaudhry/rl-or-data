# Readiness gate

Date of this note: 2026-11-08.

**Not submission-ready.** Editorial formatting of the provisional PDF is in good shape. Venue choice, deadline, anonymity, and the author's under-review status are unresolved. Those items block a real submission. They did not block editing or review.

Primary file: `latex/main.pdf` (named, kit option `preprint`).
Conditional file: `latex/main-anonymous.pdf` (kit option `submission`). Do not circulate the anonymous file. Its first page says public sharing is prohibited. That sentence is the kit's own submission footer, not a workshop rule we verified.

## Editorial format (checked on the rebuilt PDFs)

| Check | Result |
|---|---|
| Pages | 7 US Letter pages. References start on page 6 and continue on page 7. Main text excluding references ends on page 6, inside the official-list cap of 8 pages excluding references. |
| Compile | `pdflatex` + `bibtex` + `pdflatex` twice. Exit 0. No overfull boxes in the final `main.log`. |
| Fonts | All embedded fonts are Type 1. No Type 3. No Identity-H. |
| Style | Unmodified `aaai2027.sty` and `aaai2027.bst` from the official kit. No `hyperref`. No page numbers. No ICLR footer. |
| Captions | Under all four tables, in both PDFs. |
| Metadata | Info dictionary has no Title and no Author. Named PDF has `/TemplateVersion (2027.1)`. The author name is on page 1 of the named PDF only. |
| Anonymity of the conditional PDF | Text has no author name, affiliation, email, GitHub URL, or arXiv id. |
| Original files | `workshop/preserved-arxiv-v1/2610.25812.pdf` sha256 is still `7d28a0b7186e789ce7680f0b37edfc924c50b1f28a235b1cc4b9358c50740074`. |

Local recompile of the unpacked arXiv source: 20 US Letter pages after `pdflatex`, `bibtex`, and two more `pdflatex` runs. That local PDF does not copy the arXiv metadata (the arXiv PDF has Title and Author set). It is a compile check, not a replacement for the arXiv PDF.

## Unresolved decisions (blockers to submission)

1. **No venue is selected.** NeSy-FM is a tentative candidate only. The paper does not propose a neuro-symbolic method. A NeSy-FM reviewer would have grounds to call it off-topic. MATH4AI and DAI were not chosen. This draft does not claim any of them.
2. **NeSy-FM dates conflict.** Official list, raw HTML retrieved 2026-11-08: "All deadlines are 11:59 PM Anywhere on Earth (UTC-12). Submission Deadline: November 20, 2026." Notifications December 2, 2026. Organizer site, same day: submissions due November 30, 2026, Anywhere on Earth (UTC-12), notifications December 10, and "refer to this website for the latest information." The retrieved organizer text states the timezone and does not state an hour. This note does not invent 23:59 for November 30. Internal planning target, not an official deadline: Friday, November 20, 2026, during daytime Pacific Time.
3. **Clocks that were not stated are unknown.** MATH4AI on the official list: November 20, 2026, no clock. DAI on the official list: November 20, 2026, no clock. The DAI organizer site says deadlines follow Anywhere on Earth and does not state an hour. None of those dates is reported here as 23:59 AoE.
4. **Anonymity for NeSy-FM is not stated** on the official-list section or the organizer homepage retrieved on 2026-11-08. MATH4AI's official list says single-blind. DAI's organizer site says double-blind. The named PDF is the file to read. The anonymous PDF is only a conditional compile.
5. **Under-review status is unknown.** ICLR 2027 style in the preprint does not prove an ICLR submission. arXiv v2 was HTTP 404 on 2026-11-08. v1 returned HTTP 200. ICLR's author guidelines, retrieved the same day, say parallel submission to conferences or journals is not allowed, that arXiv is allowed, and that a workshop without proceedings does not by itself violate that policy. That sentence is not permission to submit. The author has to say whether any version is under review before anyone submits.
6. **No simultaneous-workshop permission is on file.** NeSy-FM and MATH4AI do not state a dual-workshop rule in the text checked on 2026-11-08. Unknown is not permission. DAI's organizer site allows concurrent submission when the other venue allows it. This draft was not prepared as a DAI paper, and DAI was not selected.
7. **Adapters and the shared RFT completions are not in the tree.** The PDF says the adapters were described as private and were not re-checked. That is a reproducibility limit of the public artifact, not a gap to be filled by a new training run.

## Review process

Four separate reviewer agents read the PDF and wrote independent reports, then each was resumed for a second pass on the revised PDF. They did not read one another's reviews. A short set of line edits after that second pass fixed a "three times" versus "twice" inconsistency, softened two section headings, and put the mixed-pair truncation flag and the iterated-RFT adverse bound into the abstract. Those last lines were recompiled. The reviewers did not score them. No numerical claim in those lines is new. They repeat numbers already in the body and in `truncation_bounds.md`.

Remaining scientific limits are the ones the paper states: one task, one model, LoRA, three seeds, a flagged and budget-unmatched curated comparison, a failed curation prediction, and no OOD conclusion.
