import pytest

from rlordata.core import evaluate as ev
from tests.conftest import skip_unless_implemented

pytestmark = pytest.mark.core


@pytest.mark.parametrize(
    "n,c,k,expected", [(8, 8, 1, 1.0), (8, 0, 4, 0.0), (8, 1, 1, 1 / 8), (8, 1, 8, 1.0)]
)
def test_pass_at_k(n, c, k, expected):
    skip_unless_implemented(ev.pass_at_k, 8, 4, 2)
    assert abs(ev.pass_at_k(n, c, k) - expected) < 1e-9


# TODO(Laksh): bootstrap CI contains point estimate; CI narrows with more problems; per-tier weights.
