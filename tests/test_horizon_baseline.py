"""Regression tests: supervised GRU must know the requested future horizon."""
import numpy as np
import pytest
import torch
from types import SimpleNamespace
from biojepa.model.baselines import GRUProg, gru_collate
from biojepa.evaluate import gru_rollout

def study():
    subject = SimpleNamespace(times=np.array([0., .5, 1.]), dyn=np.array([[26.,1.],[25.,1.5],[24.,2.]]))
    specs = [SimpleNamespace(name="MMSE"), SimpleNamespace(name="CDRSB")]
    return SimpleNamespace(subjects=[subject], dyn_specs=specs)

def test_horizon_changes_input_but_not_history():
    s = study()
    pairs = [dict(subject=0, i=1, j=2, dt=.5),
             dict(subject=0, i=1, j=2, dt=2.)]
    seq, lengths = gru_collate(s, pairs, lambda sid: s.subjects[sid].dyn,
                               np.array([[72., 1.]],dtype=np.float32), dt_dim=8)
    assert seq.shape == (2,2,22)
    np.testing.assert_allclose(seq[0,:,:4], seq[1,:,:4])
    assert not np.allclose(seq[0],seq[1])
    assert lengths.tolist() == [2,2]
    model = GRUProg(seq.shape[-1], 2)
    output = model(torch.tensor(seq),torch.tensor(lengths))
    assert output.shape == (2,2)

def test_invalid_gap_rejected():
    s=study()
    with pytest.raises(ValueError, match="Forecast gap"):
        gru_collate(s,[dict(subject=0,i=1,j=2,dt=-1.)],
                    lambda sid:s.subjects[sid].dyn,np.zeros((1,2),dtype=np.float32))
