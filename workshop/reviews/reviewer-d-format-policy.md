# Reviewer D: formatting and venue policy

This review covers format and venue policy only. The paper was not edited. No other review was consulted. Sources read: `workshop/aaai27-provisional/latex/main.tex`, `main-anonymous.tex`, `shared.tex`, `aaai2027.sty` (preamble and option handling), both workshop PDFs, and the preserved arXiv PDF. The style files were compared with the author kit named in `AUTHOR_KIT_SHA256.txt`. Official venue facts used here are the ones re-fetched on 2026-11-08 and supplied with this assignment.

The draft is not ready to submit. The editorial format is close to the AAAI 2027 kit and is inside the NeSy-FM official-list page cap, with one major caption-placement defect. Venue, deadline, anonymity, and under-review status are unresolved author decisions. Those open items are separate from the caption defect.

## Blockers (author and venue decisions)

### B1. No venue is selected

The author has not selected a venue. `main.tex` loads `\usepackage[preprint]{aaai2027}` and the comment calls the file a named provisional draft. `main-anonymous.tex` loads `\usepackage[submission]{aaai2027}` and the comment says to use it only if a venue requires it. Neither PDF, and neither source file, names NeSy-FM, MATH4AI, DAI, AAAI, or any other event. Page limits, anonymity, and the clock on the deadline cannot be locked until a venue is chosen.

### B2. The two NeSy-FM sources conflict

Both statements below are from the facts re-fetched on 2026-11-08. They disagree, so neither is treated here as the rule that wins.

- Official workshop list, NeSy-FM section, from https://aaai.org/conference/aaai/aaai-27/aaai-27-workshop-program-list/ : full papers up to 8 pages of main content excluding references. The deadline text on that list is "All deadlines are 11:59 PM Anywhere on Earth (UTC-12). Submission Deadline: November 20, 2026." That section states no anonymity rule and no dual-submission sentence.
- Organizer site, https://llm-symbol.github.io/NeSy-FM-AAAI2027/ : submissions due November 30, 2026 Anywhere on Earth, notifications December 10, page accounting "excluding appendices," appendices unlimited, and a direction to use that website for the latest information. The supplied organizer facts state Anywhere on Earth and do not state an hour. This review does not invent 11:59 PM for that date.

The conflict is the calendar date (November 20 versus November 30) and what a page cap excludes (references versus appendices). This 6-page file does not exceed either length rule. See the page-count section. The conflict still blocks a real submission, because the author cannot know which deadline to meet, and a later appendix would be scored differently under the two rules.

### B3. Dates with no clock are not Anywhere on Earth

- MATH4AI on the official list: November 20, 2026, with no clock. That date is not read as Anywhere on Earth.
- DAI on the official list: November 20, 2026, with no clock. The DAI organizer site says deadlines follow Anywhere on Earth and does not state an hour. The official-list date is not read as Anywhere on Earth on the strength of the organizer sentence.

This review was not given page limits or anonymity rules for MATH4AI or DAI, and none are applied.

### B4. Under-review status is unknown

Whether the manuscript is under review anywhere is unknown. That is a blocker to an actual submission. It is not a reason to skip this format review.

The preserved preprint is set up with `iclr2027_conference.sty` and `\iclrfinalcopy` (`workshop/preserved-arxiv-v1/PROVENANCE.txt`). Its PDF running head is the word "Preprint." (preserved PDF, page 1 and later pages). A text extract of that PDF contains no "ICLR" string. An ICLR-styled preprint with a "Preprint." head does not prove an ICLR submission. The workshop PDFs do not say "under review."

The NeSy-FM official-list section contains no dual-submission sentence. This review does not invent a ban. The author still has to confirm status before submitting anywhere.

## How the two PDFs are supposed to be used

`main.pdf` is the named provisional file. The preprint option turns the copyright notice off and leaves the author visible. That is what page 1 shows: Lakshya Chaudhry, Purdue University, lchaudhr@purdue.edu, and no footer.

`main-anonymous.pdf` is conditional. It should be used only if a chosen venue requires anonymity. Page 1 replaces the author block with "Anonymous submission" and places this notice at the bottom of the left column:

"This is an anonymized submission for review purposes only. Distribution, citation, or public sharing of this manuscript is strictly prohibited. Copyright and publication details will appear in the final version if accepted."

That notice is the `submission` option in `aaai2027.sty` (the option redefines `\copyright@text` and clears `\showauthors@on`). It is the footer the kit inserts. It is also a handling constraint: this PDF's own first page forbids public sharing. The named PDF does not carry that notice.

A later AAAI Press camera-ready would need the default package option so the copyright slug is present. The kit forbids disabling that notice on a publication copy. The named file is a preprint draft, and the copyright-off state matches that draft. It is not a defect of this provisional file.

## Page count against the 8-page official-list rule

`pdfinfo` reports 6 pages for each workshop PDF, US Letter, 612 by 792 pt.

References flow after the text. There is no `\clearpage`, `\newpage`, or `\pagebreak` in the author source. On the named PDF, pages 1 through 5 are main text. Page 6, left column, continues with Reproducibility and the AI use statement, then the heading "References," and the bibliography fills the rest of the left column and the right column (named PDF, page 6). Main content excluding references is five full pages plus part of page 6.

The anonymous PDF is also 6 pages. The page 1 notice uses some of the left column, so the conclusion runs onto page 6, followed by Reproducibility, the AI use statement, and then References (anonymous PDF, page 6). Main content excluding references is still inside page 6.

Both files are inside "up to 8 pages of main content excluding references." There is no appendix. If the organizer rule (everything counts except appendices, and appendices are unlimited) were the one that applied, the file would still be 6 pages. Length is not the conflict. Which rule would govern a future appendix is part of B2.

## Format checks that pass

Style files were not locally modified in a way that changes layout. `sha256sum /tmp/AuthorKit27.zip` is `9406a545d5d213757245eb37118fe652b2fccf64f555aff28b49ebe2a780c960`, the same digest as `AUTHOR_KIT_SHA256.txt`. `diff -q` reports no difference between the paper's `aaai2027.sty` and `aaai2027.bst` and the copies in `/tmp/AuthorKit27/AuthorKit27/`. Digests:

- `aaai2027.sty` `01990f8d3716f8e39c5b28a7478f3ac6e28630d1161de6105a7217f5e7f60e43`
- `aaai2027.bst` `5db7765ba99de5c1e4686f9b3940a0add9c5e702f2164514462bec130ccb6e3c`

The style announces `\ProvidesPackage{aaai2027}[2027/10/21 AAAI 2027 Submission format]`. It forces `\pdfpagewidth=8.5in` and `\pdfpageheight=11in`, sets a 7.0 in by 9.0 in text block, `\twocolumn`, and `\pagestyle{empty}`. It loads `newtxtext`, `helvet`, and `courier`. It errors if `hyperref` is loaded. The author preamble in `shared.tex` follows the kit skeleton: `url` with the `hyphens` option, `graphicx`, `\urlstyle{rm}`, `\def\UrlFont{\rm}`, `natbib` with no options, `caption` with no options, `\frenchspacing`, `booktabs`, and `\pdfinfo{/TemplateVersion (2027.1)}`. It does not load `times`, `helvet`, `courier`, `hyperref`, `geometry`, or another layout package. `\setcounter{secnumdepth}{2}` is inside the range the kit explicitly allows (0, 1, or 2).

Both PDFs:

- US Letter, two columns, no page numbers, no running header on the named file.
- Fonts from `pdffonts` are all Type 1, embedded and subset: TeX Gyre Termes (text), Nimbus Mono (typewriter), Computer Modern math. No Type 3 font. No TrueType font was required. Helvetica is loaded by the style and is unused because the text has no sans-serif.
- No `hyperref`. Decompressed PDFs contain no `/Annot`, `/URI`, or `/Link`, and `mutool show outline` is empty. The `url` package breaks the repository URL across lines. It does not embed a link.
- No overfull box in `main.log` or `main-anonymous.log`. Word boxes from `pdftotext -bbox` run from x = 54.0 pt to x = 558.0 pt on every page, which is a 0.75 in left margin and a 0.75 in right margin on a 612 pt page. Tops of pages 2 through 6 sit at about 54 to 57 pt. Bottoms sit near 704 pt, leaving about 88 pt. That matches the kit's 0.75 in side margins and its text block. Nothing was found in the gutter.
- Body cross-references resolve. Both PDFs say "Table 1" through "Table 4" in an order that matches the four captions. Neither PDF contains `??`.
- The bibliography uses `aaai2027.bst` (the style sets `\bibliographystyle{aaai2027}`). References are the last material in the file.
- The abstract contains no citations.

Metadata (`mutool show trailer/Info` and `pdfinfo`):

- Named PDF: Creator TeX, Producer pdfTeX-1.40.25, `/TemplateVersion (2027.1)`, creation and modification `D:20261108035004Z`. No `/Title`, `/Author`, `/Subject`, or `/Keywords`. The author is on the page, which is what the preprint option is for. The author is not in the info dictionary. The kit's own instructions say there is no need to add Title and Author tags.
- Anonymous PDF: same producer and creator, dates `D:20261108035005Z`. No Title, Author, or TemplateVersion. The missing TemplateVersion is the `submission` option, which defines `\pdfinfo` to `\relax` before `shared.tex` runs. That is kit behavior, including for the kit's own anonymous template, which also calls `\pdfinfo` under the `submission` option.

Named PDF and the ICLR footer: the named PDF text contains no "ICLR", no "iclr", no "under review", and no "Preprint." running head. Page 1 has the title and the author block and no footer. The preserved arXiv PDF is a different file (20 pages, 434975 bytes, metadata Title and Author set, running head "Preprint."). That head was not carried into the workshop PDF.

## Major

### M1. Table captions sit above the tables

The AAAI 2027 instructions say the table number and caption must appear under the table, in 10 point roman, and the kit's own examples place `\caption` after the `tabular` (`CameraReady2027.tex`, the caption paragraph and the two sample tables). In `shared.tex` each of the four `table*` environments places `\caption` before `\small` and before `tabular` (the arms table, the greedy-accuracy table, the contrasts table, and the per-seed table).

The PDFs follow that source order. Captions are above the grids on named pages 3 (Tables 1 and 2), 4 (Table 3), and 5 (Table 4), and on the corresponding anonymous pages (Table 1 on page 3, Tables 2 and 3 on page 4, Table 4 on page 5). The caption text is outside the `\small` that sets the grid, so the caption itself is still at the body size. The 9 point grid is the kit's allowed reduction ("tables ... may be altered to 9 point type"). The defect is position, not the 9 point grid, and not `\resizebox` (none is used).

This is major because the kit states it as a must, and it affects every table in both PDFs. It does not push the paper over 8 pages. Moving each caption to after its `tabular` is the kit's own pattern. This review does not edit the paper.

## Minor

### m1. Named title block has no postal address

Named PDF, page 1, gives Purdue University and the email, and no street address. The kit asks for a complete address, including email when available, in 9 point roman under the author name. This matters for an AAAI Press camera-ready. It does not affect the anonymous PDF, which hides the affiliation. It is minor for this provisional draft.

### m2. The anonymous file can still be identified if anonymity is later required

The anonymous PDF text contains no "Lakshya", "Chaudhry", "Purdue", "lchaudhr", "github.com", or "2610.25812", and the info dictionary has no Author or Title. That is the mechanical strip the `submission` option plus the `\ifanon` branch are written to do.

Page 6 of the anonymous PDF still says the file omits the repository URL and the preprint identifier because both name the author, and the AI use statement still refers to "the preprint." The reported experiment is specific. The supplied NeSy-FM section states no anonymity requirement, so this is not a current violation. It is a reason to leave `main-anonymous.pdf` unused unless a chosen venue requires an anonymous file. If that happens, those sentences on page 6 need another pass, and the public-sharing footer on page 1 still forbids distributing that file.

## Nits

### n1. Title-case pronoun

The kit asks for Chicago title case and says pronouns are capitalized. The title on page 1 of both PDFs is "Is it RL or Data? A Controlled Comparison of GRPO and Rejection-Sampling Fine-Tuning". "it" is lowercase. "or" is correctly lowercase. Both halves of "Rejection-Sampling" and "Fine-Tuning" are capitalized. "A" after the question mark is capitalized.

### n2. TemplateVersion string is one of the two strings in the kit

The named PDF info dictionary says `2027.1`. `AnonymousSubmission2027.tex` in the same zip uses `2027.1` in its live preamble. `CameraReady2027.tex` uses `2027.2` in its live preamble, while the how-to snippet inside that file uses `2027.1`. `aaai2027.sty` does not contain either string and does not check it. `2027.1` is not evidence of a patched style. The anonymous PDF omits the key for the reason given above.

### n3. Log noise, with references already correct

`main.log` reports underfull vbox badness 2080 and 1577, and an underfull hbox of badness 1102 at `shared.tex` lines 247 to 249 (the repository URL line on named page 6). `main-anonymous.log` reports underfull vbox badness 10000 and 2617, and "Label(s) may have changed. Rerun to get cross-references right." Anonymous page 4, the page associated with the badness 10000 warning, is filled with Tables 2 and 3 and sections 3.3 and 3.4. It is not a blank column. There is no overfull box and no unresolved `??`. The visible table numbers match the captions. A rerun would clear the label warning. It is not a broken reference in the PDF that was checked.

## Claims: no false neuro-symbolic claim, and no chosen venue

The introduction on page 1 of both PDFs says the paper does not claim a new architecture, a symbolic solver inside the model, or a general result about reinforcement learning with verifiable rewards. The same page calls the checker an ordinary program. The conclusion (named PDF, page 5, continuing onto the anonymous PDF's page 6) says none of the statements is a general claim that reinforcement learning is more than the data. A text search of both workshop PDFs finds no "neuro-symbolic", "NeSy", "workshop", "AAAI", "ICLR", or "under review".

The draft does not dishonestly claim a neuro-symbolic contribution. It also does not claim a venue. Choosing NeSy-FM later would be a scope decision by the author. The sentences above are the paper's own limit, and they are on the first page.

## Preserved original

`sha256sum /workspace/workshop/preserved-arxiv-v1/2610.25812.pdf` returns `7d28a0b7186e789ce7680f0b37edfc924c50b1f28a235b1cc4b9358c50740074`. That is the required digest. The file is 434975 bytes and 20 pages. `2610.25812.tar.gz` matches the digest in `workshop/preserved-arxiv-v1/SHA256SUMS` (`c1dbeed9e16c70126e9dc0c9d4bb49f1a4c63875954c8cfe7bda26746a0d3980`). The workshop PDFs are separate files in `workshop/aaai27-provisional/latex/` (`main.pdf` 245274 bytes, `main-anonymous.pdf` 244045 bytes). The preserved PDF was not overwritten.

## Readiness

Editorial format: both PDFs are US Letter, two-column, Type 1 only, free of hyperref and of an ICLR footer, with references flowing after the text, and with main content inside the 8-page NeSy-FM official-list cap. The style files match the author kit. The remaining format defect is that all four table captions sit above the tables.

Unresolved venue and author decisions: no venue is selected, the NeSy-FM official list and the organizer site disagree on the deadline and on what a page cap excludes, the anonymous PDF is conditional and its first page forbids public sharing, and under-review status is unknown. This draft is not ready to submit.

## Second pass

Rechecked after the PDFs were rebuilt. Named PDF info dictionary creation time `D:20261108040049Z`. Anonymous PDF `D:20261108040050Z`. This pass re-read `main.pdf`, `main-anonymous.pdf`, and `shared.tex`, and re-ran `sha256sum` on the preserved arXiv PDF. The paper was not edited.

### M1 is fixed

Captions now sit under the tables. In `shared.tex`, each of the four `table*` environments ends the `tabular` before `\caption`, and `\label` follows the caption. `\small` wraps only the grid. Both PDFs show the caption after the last row of that grid:

- Table 1, page 3, after the C2 format-reward row.
- Table 2, page 4, after the Qwen3-4B row.
- Table 3, page 4, after the GRPO-Curated minus IterRFT row.
- Table 4, page 5, after the IterRFT-Curated row.

The same order is in the anonymous PDF. The first-pass defect, captions above the grids, is gone.

### Page count

`pdfinfo` reports 7 pages for each PDF, US Letter, 612 by 792 pt. The first pass was 6 pages. References still start on page 6, after Reproducibility and the AI use statement, and they continue onto page 7. Page 7 is bibliography only: word boxes stop at x = 292.5 pt and y = 273.2 pt, so the text is a short left column. Main content excluding references ends on page 6. That remains inside the NeSy-FM official-list cap of 8 pages of main content excluding references. Counting references as well, the file is 7 pages. There is still no appendix. The extra page does not cross either length rule recorded in the first pass.

### Overfull boxes, fonts, metadata

`main.log` and `main-anonymous.log` contain zero lines matching `Overfull`. Underfull boxes remain and are not overflow. Both logs report an underfull vbox of badness 10000 while shipping page 4. The named log also reports underfull hboxes of badness 1102 and 3229 at `shared.tex` lines 251 to 253, the repository URL paragraph on page 6. Word boxes on pages 1 through 6 run from x = 54.0 pt to x = 558.0 pt.

Every font in both PDFs is Type 1, embedded and subset. There is no Type 3 font. Decompressed PDFs contain no `/Annot`, `/URI`, or `/Link`. The named info dictionary has `/TemplateVersion (2027.1)` and no Title or Author. The anonymous info dictionary has no Title, Author, or TemplateVersion. Neither PDF contains "ICLR", "under review", or a "Preprint." running head. Named page 1 shows the author and no footer. Anonymous page 1 says "Anonymous submission" and places the kit prohibition at the bottom of the left column, including the words "strictly prohibited."

### Anonymity leaks

The anonymous PDF text contains no "Lakshya", "Chaudhry", "Purdue", "lchaudhr", "github.com", or "2610.25812". The named PDF still has those, on page 1 and in the reproducibility paragraph on page 6, which is what the preprint option is for. The anonymous reproducibility paragraph now says the repository is one this file does not name. It no longer says the omitted identifier names the author. The anonymous file still says "the longer preprint" in the body and "the preprint" in the AI use statement, without giving an identifier. That residue matters only if a later venue requires anonymity. The anonymous page 1 notice still forbids public sharing of that file.

The title on page 1 of both PDFs is now "GRPO and Rejection-Sampling Fine-Tuning on a Procedural Counting Task". The first-pass nit about the lowercase pronoun "it" no longer matches the file.

### Preserved original

`sha256sum /workspace/workshop/preserved-arxiv-v1/2610.25812.pdf` is still `7d28a0b7186e789ce7680f0b37edfc924c50b1f28a235b1cc4b9358c50740074`.

### Readiness

Editorial format: M1 is fixed. Both PDFs are US Letter and two-column, Type 1 only, free of overfull boxes and of an ICLR footer, with references after the text. Main content is inside the 8-page NeSy-FM official-list cap.

Unresolved venue and author-status decisions: no venue is selected, the NeSy-FM official list and the organizer site still disagree on the deadline and on what a page cap excludes, the anonymous PDF is conditional and its first page forbids public sharing, and under-review status is unknown. This draft is not ready to submit.
