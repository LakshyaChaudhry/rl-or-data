# Changelog: arXiv:2610.25812v1 to the provisional AAAI draft

The preprint is unchanged. This note covers `workshop/aaai27-provisional/` only.

## What changed

- The file is set in the official AAAI 2027 author kit (`aaai2027.sty`, package date 2027/10/21), US Letter, two columns. The ICLR 2027 style, `\iclrfinalcopy`, and the "Preprint." running head are not used.
- The title is now "GRPO and Rejection-Sampling Fine-Tuning on a Procedural Counting Task." The preprint title asked "Is it RL or Data?" The limitations already refused that general question. The new title names the comparison and the task.
- The abstract and conclusion lead with the narrow measurement. They state the curated rollout mismatch (19,200 versus 14,016), the 5% test truncation flag on the 10.9 point pair, and that 10.9 is not a clean headline. They label the easy and mixed gaps as later comparisons and flag the mixed GRPO seeds. They do not say mixed data "helps" or that three rounds "recover" the gap.
- Table captions sit under the tables, at body size. The grids are 9 point.
- The mechanical success rule is labeled as such. A "yes" in the contrast table does not override the truncation flag or the adverse bounds.
- Adverse truncation bounds that qualify the smaller contrasts are now in the text, from `results/phase5/appendix/truncation_bounds.md`: RFT mixed-minus-easy test lower bound +1.2 [-2.4, +4.7]; iterated-RFT test lower bound +0.1 [-3.0, +3.2]; curated OOD lower bound -11.8 [-18.7, -5.2].
- The draft says the RFT sweep ran on seed 1, records the Mixed validation tie at 67%, and says each iterated-RFT round starts a fresh optimizer. Round 1 is the base draw. The current policy is resampled twice, not three times.
- The extractor fallback is the last integer. Reference models use a chat template. The cap formula and the 4,096 counterfactual are one sentence. Library versions from the preprint appendix are in the reproducibility note. The two post hoc intervals point at `exploratory_grpo_vs_rft.md`, not only at `contrasts.md`.
- Lu et al.'s strongest reported variant, which also keeps easy samples and was not tested here, is back in the related work.
- No new experiment, citation, or number was added. Figures from the preprint were left out because those PDFs embed Identity-H fonts, which the AAAI kit rejects, and because the numbers they show are in the tables.
- A conditional anonymous PDF uses the kit `submission` option. It is not the primary file. Its first page carries the kit notice that public sharing is prohibited.

## What was kept

- Sole author: Lakshya Chaudhry, Purdue University.
- One counting task, Qwen3-4B-Base, LoRA, three seeds.
- Pre-registered versus post hoc versus secondary labels.
- Curation does not help RFT (-2.7 points). The registered prediction fails.
- The paper does not contradict Xiong et al. 2025 or Samineni et al. 2025.
- OOD truncation blocks an OOD conclusion.
- Cross-machine re-evaluation can move test accuracy by up to 2.3 points.
- RFT seeds share one base-model draw. Selected RFT hyperparameters differ across data comparisons.
- No neuro-symbolic architecture is claimed. No venue is named in the PDF.
