"""Run the README example against the importable package without plotting extras."""
from pathlib import Path
import re
import sys

import numpy as np


def test_readme_cnm_without_optional_dependencies(monkeypatch):
    monkeypatch.setitem(sys.modules, "pandas", None)
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    example = re.search(r"^```python\n(.*?)^```$", readme, flags=re.MULTILINE | re.DOTALL)
    assert example, "README must include its executable CNM example"
    variables = {}
    exec(compile(example.group(1), "README.md", "exec"), variables)
    nu, sed = variables["frequency_GHz"], variables["emissivity"]
    assert nu.shape == sed.shape == (200,)
    assert np.all(np.diff(nu) > 0)
    assert np.all(np.isfinite(sed)) and np.all(sed >= 0) and sed.max() > 0


def test_gaunt_table_values():
    from SpyDust.free_free import gaunt_factor

    np.testing.assert_allclose(gaunt_factor(1e-4, [1e-4, 1e-3]), [5.524282, 4.258079])
