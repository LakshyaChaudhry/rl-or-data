---
name: preflight
description: Launch gate for expensive runs. Use before starting any training, sweep, generation, caching or grading job that takes more than about 1 GPU-hour or about $10, before telling the user a run is "ready to launch", and when verifying another agent's branch ahead of a run.
---

# Preflight

A run that has to be thrown away costs more than every check here combined. Work through the list, then give the user a go or no-go. Do not launch until they approve.

Check by reading the code and resolved configs and by running things. Do not rely on what a task file or another agent's summary says the code does.

## 1. Comparability across arms

For every pair of arms or conditions that will be compared, diff the resolved configs and confirm the only differences are the intended ones. Check each of these explicitly:

- prompt template and chat template
- max new tokens / completion cap, and prompt length cap
- EOS and stop handling, in both generation and the reward or loss
- answer extraction and the verifier or grader
- sampler settings (temperature, top-p, n samples) and the sampling backend
- seeds, and how they are threaded
- base checkpoint id and revision, adapter config
- data splits and how each arm draws from them
- step count, batch size and learning-rate schedule where those are meant to match

List every difference found, intended or not.

## 2. Outputs will be usable

- Run a smoke test on a handful of real examples, end to end, through the same code path as the full run.
- Look at the actual outputs. Measure the truncation rate and the extraction-failure rate. If the cap cuts off answers, the run cannot be graded later.
- Confirm the grader or verifier gives the right label on a few outputs you check by hand, including one clear success and one clear failure.
- Confirm everything needed downstream is saved (full completions, activations at the positions required, per-sample records), so nothing has to be regenerated.

## 3. The run will survive

- It is launched detached (sbatch, tmux or nohup) and keeps going if the laptop sleeps or the SSH session drops.
- Checkpoints and results go to persistent storage, and a restart resumes instead of starting over.
- Output paths are new. Nothing existing is overwritten, and no other running job writes to the same directory.
- There is enough disk for the outputs. Check free space against an estimate.
- Memory fits: the smoke test ran at the real batch size, or the batch size is known to fit.
- Any idle-shutdown timer will not fire mid-run, and it will fire after the run ends.

## 4. Shared machines

- Check the queue first. Use only free GPUs. Never stop or alter another user's job.
- Submit through the scheduler, not as a bare process on a GPU.

## 5. Cost and time

- Estimate wall-clock from the smoke test's measured throughput, not from a guess.
- Give GPU-hours and dollar cost, and the number of runs this covers.
- Say what the smallest version of the run is that would still answer the question, if the full one is large.

## 6. Results are committed to in advance

- If the project pre-registers, the predictions and thresholds are committed before launch.
- The code being run is committed and the hash is recorded in the run directory.

## Report

Give the user:

1. **Verdict**: go, or no-go with the blocking items.
2. **Differences between arms**: the full list from section 1.
3. **Smoke test**: what ran, truncation rate, extraction-failure rate, one sample output.
4. **Estimate**: time, GPU-hours, cost.
5. **Not checked**: anything on this list that could not be verified, and why.

Then wait for approval before launching.
