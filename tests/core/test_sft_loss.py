import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.core

# TODO(Laksh): all-masked batch raises; equals -mean(logp[mask]); invariant to right padding.
