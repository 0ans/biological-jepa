from scripts.repeated_ablation import run


def test_smoke(tmp_path):
    report = run('synthetic', (0,), 2, 2, str(tmp_path / 'result.json'))
    assert report['n_valid_pairs'] > 0
    assert report['per_seed'][0]['n_patients'] >= 2
