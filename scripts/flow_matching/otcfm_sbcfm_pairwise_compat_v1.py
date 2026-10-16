"""Apply the author's SB Gaussian formula to already selected normalized pairs.

The released parent handles trajectory pair selection, Sinkhorn coupling,
leaveout bridging and time rescaling before calling calc_mu_sigma. Its SB
override unnecessarily asserts the whole data module is non-trajectory even
though that method consumes only x0, x1 and local t. A read-only pair view
permits the original formula to run; global trajectory state is never changed.
This is an explicit source repair, not evidence of historical author behavior.
"""
from functools import wraps


class PairwiseInterval:
    is_trajectory = False

    def __init__(self, model):
        self._model = model

    def __getattr__(self, name):
        return getattr(self._model, name)


def pairwise_formula_adapter(original):
    @wraps(original)
    def compatible(self, x0, x1, t):
        view = PairwiseInterval(self) if self.is_trajectory else self
        return original(view, x0, x1, t)
    compatible._iia_sbcfm_pairwise_adapter = True
    return compatible


def install(owner):
    original = owner.calc_mu_sigma
    if not getattr(original, '_iia_sbcfm_pairwise_adapter', False):
        owner.calc_mu_sigma = pairwise_formula_adapter(original)
