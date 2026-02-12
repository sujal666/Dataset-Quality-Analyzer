from __future__ import annotations

from analyzers.domain_check import _tabular_mixing_severity


def test_tabular_domain_mixing_default_is_info_until_extreme_entropy() -> None:
    assert _tabular_mixing_severity(entropy=0.70, threshold=0.65) == "info"
    assert _tabular_mixing_severity(entropy=0.90, threshold=0.65) == "warning"

