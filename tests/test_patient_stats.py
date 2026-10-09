import numpy as np
import pytest
from biojepa.stats import paired_patient_bootstrap


def test_resamples_patients_not_visits():
    # Patient A has many identical visits, B has just one. Both should count once.
    ids = ["A"] * 20 + ["B"]
    y = np.zeros(21)
    a = np.array([2.] * 20 + [0.])
    b = np.array([0.] * 20 + [4.])
    result = paired_patient_bootstrap(ids, y, a, b, n_boot=300, seed=5)
    assert result["n_patients"] == 2
    assert result["n_observations"] == 21
    assert result["difference_a_minus_b"] == pytest.approx(-1.)
    assert result["resampling_unit"] == "patient"


def test_reproducible_and_missing_values_masked():
    args = (["a","a","b","c"], [0, 1, 2, 0],
            [1, 0, 3, np.nan], [0, 1, 2, 0])
    one = paired_patient_bootstrap(*args, n_boot=200, seed=7)
    two = paired_patient_bootstrap(*args, n_boot=200, seed=7)
    assert one == two
    assert one["n_patients"] == 2


def test_missing_and_single_subject_rejected():
    with pytest.raises(ValueError, match="two distinct"):
        paired_patient_bootstrap(["a","a"], [1,2], [1,2], [0,1])
    with pytest.raises(ValueError, match="aligned"):
        paired_patient_bootstrap(["a","b"], [1], [1], [1])
