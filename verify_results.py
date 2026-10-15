"""Verify source snapshot, split ordering, predictions, and reported metrics."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from assignment5 import KINDS, read_data, predict_prices, measures, date_values

root=Path(__file__).resolve().parent
metrics=json.loads((root/'results/metrics.json').read_text())
source=root/'data/finance-charts-apple.csv'
assert hashlib.sha256(source.read_bytes()).hexdigest()==metrics['data_sha256']
dates,y=read_data(source); cut=metrics['n_train']
assert dates[cut-1] < dates[cut]
params={k:metrics['models'][k]['selected_params'] for k in KINDS}
with threadpool_limits(limits=1):
    models=predict_prices(dates[:cut],y[:cut],dates,params)
with (root/'results/predictions.csv').open() as f: saved=list(csv.DictReader(f))
for kind,entry in models.items():
    expected=np.array([float(row[kind]) for row in saved])
    assert np.allclose(entry['predictions'],expected,atol=1e-8,rtol=1e-9),kind
    model=entry['estimator']
    assert model.x_scaler.n_samples_seen_==cut
    assert model.y_scaler.n_samples_seen_==cut
    assert model.model.fit_status_==0
    for split,ix in [('train',slice(None,cut)),('test',slice(cut,None))]:
        recomputed=measures(y[ix],entry['predictions'][ix])
        for key,value in recomputed.items():
            assert np.isclose(value,metrics['models'][kind][split][key],atol=1e-8,rtol=1e-9)
assert np.max(abs(models['rbf']['predictions']-models['precomputed_rbf']['predictions'])) < 1e-8
with threadpool_limits(limits=1):
    # Demonstrate scalar query handling for both string and numeric dates.
    for kind,entry in models.items():
        model=entry['estimator']
        a=model.predict(dates[-1]);b=model.predict(date_values([dates[-1]])[0,0])
        assert a.shape==(1,) and np.allclose(a,b)
print('PASS: source hash, ordered split, training-only scalers, all six refits,')
print('      metrics, RBF equivalence, scalar queries, and solver convergence.')
