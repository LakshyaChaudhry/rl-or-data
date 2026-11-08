# Local recompile of arXiv:2610.25812v1

Source: `workshop/preserved-arxiv-v1/arxiv-source/` copied to `/tmp/orig-compile` and compiled there. The preserved tree was not modified.

Commands: `pdflatex`, `bibtex`, `pdflatex`, `pdflatex` (TeX Live 2023, pdfTeX 3.141592653-2.6-1.40.25).

Result: `main.pdf`, 20 pages, US Letter (612 x 792 pt). Exit code 0 after the bibliography pass. The first pass warned about undefined citations. Those warnings were gone after `bibtex` and the later passes.

`pdfinfo` on this local PDF shows an empty Title and an empty Author. The downloaded arXiv PDF, which was not overwritten, has Title and Author set and was produced by arXiv GenPDF. This local file is evidence that the unpacked source compiles. It is not a byte match for the arXiv PDF.

Preserved arXiv PDF sha256: `7d28a0b7186e789ce7680f0b37edfc924c50b1f28a235b1cc4b9358c50740074`.
