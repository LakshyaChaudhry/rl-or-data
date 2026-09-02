"""AGENT-OWNED acceptance tests for tasks/01. All must pass before tasks/01 is done."""

import pytest

pytestmark = pytest.mark.skip(reason="tasks/01 not started")

# Required tests (implement all):
# 1. determinism: generate_pool(cfg, seed) twice -> identical problem_ids and texts
# 2. independent reference executor (written in this file, NOT importing generator.execute_pipeline)
#    agrees with execute_pipeline on every problem in a 2,000-problem pool
# 3. canonical_id ignores key order and whitespace; changes when any pipeline element changes
# 4. split disjointness by problem_id AND by structure (pipeline minus range)
# 5. knob monotonicity: mean total_steps and range span increase across S<M<L / step configs
# 6. every operator appears with >=3 distinct surface renderings across the pool
# 7. answers are bounded ints (product uses product-mod-m), no NaN/inf, no empty sets after filtering
#    (generator must reject/resample pipelines whose filtered set is empty)
