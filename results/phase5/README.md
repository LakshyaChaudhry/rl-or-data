# Phase 5 results — tasks/06b: iterated RFT (secondary), truncation bounds, exploratory cap re-eval, base gsm8k mean@8

A committed snapshot of `outputs/` (gitignored) as written by `make analysis` at ed1d408 on 2026-09-21
from `runs/store_mirror/runs`, plus the small provenance files of the new runs under `runs/` (no samples,
no weights). Figures (d) and (g) were regenerated at 61e1cf3 (tasks/07: the ninth arm had pushed the C2
panel out of (d); (g) gained the IterRFT legend entry) — tables and sheets are byte-identical to ed1d408.
Analysis config `configs/analysis/default.yaml`, hash `3ba220e48a66`; bootstrap seed 0,
10,000 resamples; SPEC v1.8. Sanity: 13 checks pass with the new arm included. No file exceeds 1 MB.

**Everything Phase 4 reported is unchanged.** `uv run python scripts/compare_phase_tables.py results/phase4
results/phase5 --expect-changed base,gsm8k_mean8` → every row of the six Phase 4 CSV tables reappears byte
for byte; the one expected change is base gsm8k_500 mean@8, from `missing` to a value. The confirmatory
sections of `hypotheses.md` (H1–H3, P1–P2, controls) are word-for-word those of Phase 4; the new section
follows them. `results_packet.md` / `results_flat.csv` of Phase 4 were not regenerated (the packet has a
300-line budget sized for the Phase 4 arm set).

## What was run (queue of 2026-09-20/21, 12.07 GPU-h on 1× H100 PCIe, git 4d97b07, clean tree)

- **Iterated RFT on the frozen train_curated**, seeds 1–3 — **secondary: registered after unblinding**
  (PREREGISTRATION §6). 3 rounds × 64 samples per prompt = 14,016 rollouts (round 1 = first 64 per prompt
  of the shared base draw; rounds 2–3 sampled from the current policy), each round continues the previous
  adapter, lr 1e-5 / 4 epochs reused from RFT-Curated (no sweep; §4 deviation), final = after round 3.
  Kept samples per round: 2,262 / 2,614 / 2,927 · 2,262 / 2,642 / 2,951 · 2,262 / 2,573 / 2,992.
- **Exploratory re-evaluation at a cap of 8,704** (2 × the locked 4,352; §4 deviation), greedy, test_300 and
  ood_hard_200: base, RFT-Curated, GRPO-Curated, IterRFT × seeds (20 units).
- **Base gsm8k_500 mean@8**: 0.7365 [0.7138, 0.7583], n = 500 × 8, seed 1, truncation 0.05 %, config
  3407e83a5380 (owed since Phase 1).

## Numbers (seeds 1–3; greedy; bootstrap 95 % CI; tr = truncation; no verdict is written anywhere)

| | test_300 (n = 300) | ood_hard_200 (n = 200) |
|---|---|---|
| IterRFT-Curated, per seed | 0.697 / 0.700 / 0.707 (tr 1.3 / 1.7 / 1.7 %) | 0.280 / 0.270 / 0.315 (tr 14.5 / 16.0 / 14.5 %) |
| IterRFT-Curated, μ ± σ [CI] | 0.701 ± 0.005 [0.654, 0.747] | 0.288 ± 0.024 [0.235, 0.342] |
| (i) IterRFT − RFT-Curated, Δ per seed | +0.023 / +0.043 / +0.030 | −0.020 / −0.020 / +0.050 |
| (i) mean Δ ± seed std [paired CI] | +0.032 ± 0.010 [+0.002, +0.062] | +0.003 ± 0.040 [−0.035, +0.038] |
| (ii) GRPO-Curated − IterRFT, Δ per seed | +0.097 / +0.060 / +0.073 | +0.080 / +0.020 / −0.020 |
| (ii) mean Δ ± seed std [paired CI] | +0.077 ± 0.019 [+0.037, +0.117] | +0.027 ± 0.050 [−0.022, +0.073] |

SPEC §10, evaluated mechanically: (i) |Δ test| 0.032 > 2 × pooled seed std 0.017, same sign on ood → met;
(ii) 0.077 > 0.025, same sign on ood → met, **with all three GRPO-Curated seeds over the §7 truncation
flag**. H3 Δ +0.109 = (ii) +0.077 + (i) +0.032 exactly (shares 70 % / 30 %, descriptive only; no
threshold is registered). Analysis config hash 3ba220e48a66; per-unit config hashes in `tables/units.csv`.

## Read before using any number

1. **Registered after unblinding.** (i) and (ii) are secondary; they are not part of H1–H3.
2. **Unequal seed variance.** RFT-Curated's seeds share one base draw (its seed std omits sampling
   variance); IterRFT's include it. The pooled seed std in the criterion mixes the two.
3. **The ood sign of both contrasts rests on one seed each way**: (i) − / − / +, (ii) + / + / −; both ood
   CIs include 0. Criterion (b) reads only the sign of the mean.
4. **Greedy decoding is not reproducible across hosts** (`appendix/exploratory_cap.md`, found as a
   by-product of the cap re-eval). Same host: 1,396 of 1,396 untruncated completions re-generate byte for
   byte, 0 correctness flips. Different host, identical settings, packages, sampling code and GPU model:
   724 of 3,067 (23.6 %) identical, 6–10 % of problems flip correctness per unit, roughly symmetrically;
   single-run accuracy moves by up to 2.3 points on test_300 (up to 5.0 on ood) from the flips among
   untruncated completions alone. The arms
   were evaluated on different hosts (base, RFT, GRPO, IterRFT each on its own), and an arm's seed std
   does not contain this. It is small next to H3 (+0.109) and (ii) (+0.077) and comparable to the small
   contrasts: (i) +0.032, H1-RFT +0.037, H2 numerator −0.027. One same-host check exists: at cap 8,704,
   RFT-Curated and IterRFT were both generated on the tasks/06b box, and (i)\* is +0.031 [−0.000, +0.063].
5. **Truncation bounds** (`appendix/truncation_bounds.md`, post hoc): on test_300 the sign of Δ is the
   same at both ends for H1 (both), H2 denominator, H3, (i) [+0.001, +0.048] and (ii) [+0.061, +0.152];
   not for the H2 numerator [−0.047, +0.004]. On ood_hard_200 the interval contains 0 for 7 of 7
   contrasts: uninformative, including for criterion (b).
6. **Exploratory cap 8,704\*** (never a §10 number): GRPO-Curated test 0.823 / 0.833 / 0.803\* (primary
   0.793 / 0.760 / 0.780), ood 0.405 / 0.395 / 0.415\* (primary 0.360 / 0.290 / 0.295); H3\* +0.150 on test,
   +0.115 [+0.060, +0.170] on ood; (ii)\* +0.119 test, +0.107 [+0.058, +0.155] ood. For the models whose
   primary run was on another host, 'change' mixes the cap with item 4; only the 'cut → right' counts are
   attributable to the cap. GRPO-Curated is still cut at 8,704 on 23–31 % of ood for seeds 2–3.
7. Defect, not fixed mid-queue so the SHA stays 4d97b07 across seeds: the run-level `meta.json` of the
   three iterated-RFT runs records `wall_clock_s: 0.001`; `gpu_hours_actual` (2.61 / 2.66 / 2.72) and the
   per-round meta files are right.

## What is here

- `hypotheses.md` — Phase 4 sheet unchanged + `## Secondary — iterated RFT`, **verdict line blank**.
- `tables/` — as Phase 4, with the IterRFT rows, S1/S2 in `contrasts.*`, base gsm8k mean@8 filled.
- `appendix/truncation_bounds.{md,csv}`, `appendix/exploratory_cap.{md,csv}` — never in `tables/`.
- `how_could_this_be_wrong.md` — adds S1 and S2 (no sweep of its own; registered after unblinding).
- `figures/` — (a)–(i); the IterRFT arm in purple, labelled secondary in the legend.
- `runs/` — provenance of the 3 iterated-RFT runs (per round: config, hash, meta, budgets, diagnostics,
  train log; final eval units), the 20 exploratory units and the base gsm8k unit.
  `SHA256SUMS.tasks06b-20260921`: the manifest the local mirror was verified against (620/620).
