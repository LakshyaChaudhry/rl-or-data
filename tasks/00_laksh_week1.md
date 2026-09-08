# tasks/00 — Laksh's own Week 1 (not for the agent)

Do these by hand, in order. Each is small. Log each in notebook/LAB_NOTEBOOK.md.

1. **Read SPEC.md end to end.** Write, in your own words, the three hypotheses and what result would
  falsify each. Put that paragraph in notebook/PREREGISTRATION.md §1.
2. **core/verify.py** (pure Python). Then author the remaining cases in tests/core/test_verify.py.
  `make test` should pass verify tests. Time: ~1 hour.
3. **Concept check (fundamentals chat or here):** tokenizer → forward → logits [T,V] → log_softmax →
  gather. Then write **core/logprobs.py**. Test it against F.cross_entropy on a 2-layer random GPT2
   (transformers `GPT2Config(n_layer=2, n_embd=64)`); this runs on CPU in seconds. Time: ~2–3 hours.
4. **core/evaluate.py**: pass_at_k (write the product-form derivation in the docstring), bootstrap_ci,
  compute_metrics. Author the remaining evaluate tests. Time: ~2 hours.
5. **Review the agent's generator PR (tasks/01)** using the ritual: docstring in your own words for
  generator.execute_pipeline; ask the agent to explain one design choice you didn't specify; run
   `make test`; sample 20 problems and solve 5 by hand to confirm answers.
6. **AWS:** redeem credits, request quotas, note expiry and region in the notebook (setup/AWS_LAUNCH.md §0). <- i changed this to work on lambda

Deliverable by end of Week 1: verify, logprobs, evaluate implemented and tested; generator merged;
base-model eval run on GPU; cap.yaml locked; pre-registration §1–2 filled.