import importlib.util
import json
import math
from pathlib import Path


def _load_validator_module():
    script = Path(__file__).resolve().parents[2] / "scripts" / "validate_wds_loader.py"
    spec = importlib.util.spec_from_file_location("validate_wds_loader", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_validator_expected_freq_uses_sampling_temperature(tmp_path):
    validator = _load_validator_module()
    small = tmp_path / "small"
    large = tmp_path / "large"
    small.mkdir()
    large.mkdir()
    (small / "metadata.json").write_text(json.dumps({"sample_count": 4}))
    (large / "metadata.json").write_text(json.dumps({"sample_count": 100}))

    expected = validator._expected_frequencies(
        [str(small), str(large)],
        datasets_weights=[1.0, 1.0],
        sampling_temperature=0.5,
    )

    assert math.isclose(expected["small"], 2 / 12, rel_tol=1e-9)
    assert math.isclose(expected["large"], 10 / 12, rel_tol=1e-9)
