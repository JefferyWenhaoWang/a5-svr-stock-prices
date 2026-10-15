"""Assignment 5: compare all SVR kernel interfaces on historical AAPL prices.

Run: python assignment5.py --out results
Training and inference use scikit-learn's libsvm-backed SVR.
The CSV snapshot is bundled; reruns need no data downloads.
"""
import os
os.environ.setdefault('MPLCONFIGDIR', str(__import__('pathlib').Path(__file__).resolve().parent/'.mplconfig'))
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import argparse
import csv
from datetime import date
from functools import partial
import hashlib
import importlib.metadata
import itertools
import json
from pathlib import Path
import platform
import time
import warnings
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics.pairwise import rbf_kernel, laplacian_kernel
from sklearn.model_selection import TimeSeriesSplit
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

KINDS = ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed_rbf', 'laplacian']
NAMES = {'linear':'Linear', 'poly':'Polynomial', 'rbf':'RBF', 'sigmoid':'Sigmoid',
         'precomputed_rbf':'Precomputed RBF', 'laplacian':'Callable Laplacian'}
HORIZON = 60

def date_values(values):
    a = np.asarray(values).reshape(-1)
    if a.dtype.kind in 'iuf':
        return a.astype(float).reshape(-1,1)
    return np.array([date.fromisoformat(str(v)[:10]).toordinal() for v in a], dtype=float).reshape(-1,1)

class ScaledSVR:
    """Training-only scalers, with consistent preprocessing for each kernel."""
    def __init__(self, kind, params):
        self.kind, self.params = kind, dict(params)

    def fit(self, dates, prices):
        x, y = date_values(dates), np.asarray(prices, dtype=float).reshape(-1,1)
        self.x_scaler, self.y_scaler = StandardScaler(), StandardScaler()
        self.train_x = self.x_scaler.fit_transform(x)
        scaled_y = self.y_scaler.fit_transform(y).ravel()
        p = dict(self.params)
        gamma = p.pop('gamma', 1.0)
        self.gamma = gamma
        if self.kind == 'precomputed_rbf':
            self.model = SVR(kernel='precomputed', **p, tol=1e-5, cache_size=200, max_iter=2000000)
            train_input = rbf_kernel(self.train_x, self.train_x, gamma=gamma)
        elif self.kind == 'laplacian':
            self.model = SVR(kernel=partial(laplacian_kernel, gamma=gamma), **p,
                             tol=1e-5, cache_size=200, max_iter=2000000)
            train_input = self.train_x
        else:
            self.model = SVR(kernel=self.kind, gamma=gamma, **p,
                             tol=1e-5, cache_size=200, max_iter=2000000)
            train_input = self.train_x
        with warnings.catch_warnings():
            warnings.simplefilter('error', ConvergenceWarning)
            self.model.fit(train_input, scaled_y)
        return self

    def predict(self, dates):
        query = self.x_scaler.transform(date_values(dates))
        if self.kind == 'precomputed_rbf':
            query = rbf_kernel(query, self.train_x, gamma=self.gamma)
        return self.y_scaler.inverse_transform(self.model.predict(query).reshape(-1,1)).ravel()

def predict_prices(dates, prices, x=None, params=None):
    """Fit all six settings and predict query dates; replaces a three-kernel loop.

    dates and x may be ISO dates or numeric time coordinates. x=None requests
    fitted values at the training dates. Returns a dict keyed by kernel name;
    each entry has a fitted estimator and a one-dimensional predictions array.
    """
    x = dates if x is None else x
    output = {}
    for kind in KINDS:
        p = (params or {}).get(kind, {'C':10.0, 'epsilon':.05, 'gamma':1.0,
                                     'degree':3, 'coef0':1.0})
        model = ScaledSVR(kind,p).fit(dates,prices)
        output[kind] = {'estimator':model,'predictions':model.predict(x)}
    return output

def candidates(kind):
    common = {'C':[1.,10.,100.], 'epsilon':[.02,.1]}
    if kind == 'poly': common.update(gamma=[.1,1.,10.], degree=[2,3], coef0=[0.,1.])
    if kind == 'rbf' or kind == 'laplacian': common.update(gamma=[.1,1.,10.])
    if kind == 'sigmoid': common.update(gamma=[.1,1.,10.], coef0=[-1.,0.,1.])
    keys=list(common)
    return [dict(zip(keys, values)) for values in itertools.product(*(common[k] for k in keys))]

def measures(y,p):
    e=p-y
    return {'rmse':float(np.sqrt(mean_squared_error(y,p))),
            'mae':float(mean_absolute_error(y,p)), 'r2':float(r2_score(y,p)),
            'bias':float(np.mean(e)), 'max_abs':float(np.max(abs(e)))}

def read_data(filename):
    with open(filename,newline='') as f: rows=list(csv.DictReader(f))
    rows=sorted(rows,key=lambda r:r['Date'])
    dates=np.array([r['Date'] for r in rows])
    prices=np.array([float(r['AAPL.Close']) for r in rows])
    assert len(set(dates)) == len(dates)
    assert np.all(np.isfinite(prices)) and np.all(prices>0)
    assert np.all(np.diff(date_values(dates).ravel())>0)
    return dates,prices

def benchmark(selected, dates, prices, test_dates, repetitions=41):
    """Randomized order, one warm-up per model, median/IQR, full pipeline times.

    Fit times include both scalers and (for precomputed/callable) Gram creation.
    Predict times include scaling, kernel construction, and inverse scaling.
    Prediction batch is all 60 test dates. No CV, disk, import, or plot time.
    """
    rng=np.random.default_rng(7773)
    trained={k:ScaledSVR(k,selected[k]).fit(dates,prices) for k in KINDS}
    for model in trained.values(): model.predict(test_dates)
    observations={k:{'fit_ms':[],'predict_ms':[]} for k in KINDS}
    for _ in range(repetitions):
        for kind in rng.permutation(KINDS):
            start=time.perf_counter_ns()
            model=ScaledSVR(kind,selected[kind]).fit(dates,prices)
            end=time.perf_counter_ns()
            observations[kind]['fit_ms'].append((end-start)/1e6)
            start=time.perf_counter_ns()
            for __ in range(20): model.predict(test_dates)
            end=time.perf_counter_ns()
            observations[kind]['predict_ms'].append((end-start)/20/1e6)
    result={}
    for k,d in observations.items():
        result[k]={}
        for label,times in d.items():
            result[k][label]={'median':float(np.median(times)),
                             'q25':float(np.quantile(times,.25)),
                             'q75':float(np.quantile(times,.75)), 'raw':times}
    return result

def run(data_file,out):
    out.mkdir(parents=True,exist_ok=True)
    dates,y=read_data(data_file)
    cut=len(y)-HORIZON
    td,ty=dates[:cut],y[:cut]
    split=TimeSeriesSplit(n_splits=3,test_size=HORIZON)
    folds=list(split.split(td))
    result={'data_file':str(data_file.name),'data_sha256':hashlib.sha256(data_file.read_bytes()).hexdigest(),
            'n':len(y),'n_train':cut,'n_test':HORIZON,'first_date':dates[0],
            'last_train':td[-1],'first_test':dates[cut],'last_date':dates[-1],
            'feature':'elapsed calendar time (equivalent to date ordinal after standardization)',
            'target':'AAPL.Close, nominal historical unadjusted closing price in USD',
            'python':platform.python_version(),'platform':platform.platform(),
            'machine':platform.machine(),'versions':{p:importlib.metadata.version(p) for p in ['numpy','scipy','scikit-learn','matplotlib','threadpoolctl']},
            'folds':[{'n_train':len(tr),'train_end':td[tr[-1]],'validation_start':td[va[0]],'validation_end':td[va[-1]],'n_validation':len(va)} for tr,va in folds],
            'models':{}, 'benchmark_repetitions':41,'predict_inner_repetitions':20}
    selected={}; all_cv=[]
    for kind in [k for k in KINDS if k!='precomputed_rbf']:
        best_score=float('inf'); best=None; records=[]
        started=time.perf_counter()
        for params in candidates(kind):
            scores=[]; failure=None
            try:
                for tr,va in folds:
                    model=ScaledSVR(kind,params).fit(td[tr],ty[tr])
                    scores.append(measures(ty[va],model.predict(td[va]))['rmse'])
            except (ConvergenceWarning, ValueError) as exc:
                failure=str(exc)
            score=float(np.mean(scores)) if failure is None else None
            row={'kernel':kind,'params':params,'fold_rmse':scores,'mean_rmse':score,'failure':failure}
            records.append(row)
            if score is not None and score < best_score:
                best_score,best=score,params
        if best is None: raise RuntimeError(f'Every {kind} candidate failed')
        selected[kind]=best
        result['models'][kind]={'name':NAMES[kind], 'selected_params':best,
             'cv_rmse':best_score, 'candidate_count':len(records),
             'failed_candidates':sum(r['failure'] is not None for r in records),
             'search_seconds':time.perf_counter()-started}
        all_cv.extend(records)
        print(kind, result['models'][kind],flush=True)
    # Identical RBF function; reuse selected parameters for an implementation check.
    selected['precomputed_rbf']=dict(selected['rbf'])
    result['models']['precomputed_rbf']={'name':NAMES['precomputed_rbf'],
        'selected_params':selected['precomputed_rbf'],'cv_rmse':result['models']['rbf']['cv_rmse'],
        'candidate_count':0,'failed_candidates':0,'search_seconds':0.0}
    fitted=predict_prices(td,ty,dates,params=selected)
    pred={}
    for kind,entry in fitted.items():
        model,p=entry['estimator'],entry['predictions']
        pred[kind]=p
        result['models'][kind].update(train=measures(ty,p[:cut]),test=measures(y[cut:],p[cut:]),
              support_vectors=len(model.model.support_),solver_iterations=int(model.model.n_iter_),
              fit_status=int(model.model.fit_status_))
    rbfdiff=float(np.max(abs(pred['rbf']-pred['precomputed_rbf'])))
    assert rbfdiff<.01,rbfdiff
    result['rbf_max_prediction_difference_dollars']=rbfdiff
    result['baseline_fixed_last_close']=measures(y[cut:],np.full(HORIZON,ty[-1]))
    # Same forecast origin and horizon: the baseline is not updated during testing.
    result['baseline_last_close_price']=float(ty[-1])
    # Test definiteness on the final standardized training Gram matrices.
    xn=fitted['rbf']['estimator'].x_scaler.transform(date_values(td))
    sig=selected['sigmoid']
    gram=np.tanh(sig['gamma']*(xn@xn.T)+sig['coef0'])
    eigenvalues=np.linalg.eigvalsh(gram)
    result['sigmoid_gram_min_eigenvalue']=float(eigenvalues[0])
    result['sigmoid_gram_max_eigenvalue']=float(eigenvalues[-1])
    result['rbf_train_gram_mebibytes']=cut*cut*8/1024**2
    result['test_gram_mebibytes']=HORIZON*cut*8/1024**2
    print('Benchmarking selected models...',flush=True)
    result['timings']=benchmark(selected,td,ty,dates[cut:])
    (out/'metrics.json').write_text(json.dumps(result,indent=2))
    (out/'cross_validation.json').write_text(json.dumps(all_cv,indent=2))
    with (out/'predictions.csv').open('w',newline='') as f:
        writer=csv.writer(f); writer.writerow(['Date','split','actual_close']+KINDS)
        for i,d in enumerate(dates): writer.writerow([d,'train' if i<cut else 'test',y[i]]+[pred[k][i] for k in KINDS])
    make_plots(dates,y,pred,result,out)
    print(json.dumps(result,indent=2),flush=True)

def make_plots(dates,y,pred,result,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    plt.rcParams.update({'font.size':9,'axes.titlesize':10,'axes.labelsize':9,
                         'axes.spines.top':False,'axes.spines.right':False,
                         'savefig.facecolor':'white','font.family':'DejaVu Sans'})
    d=np.array(dates,dtype='datetime64[D]'); cut=result['n_train']
    colors=['#22628E','#885FAD','#C45F2A','#8A7133','#497A75','#347247']
    figs=out/'figures';figs.mkdir(exist_ok=True)
    def save(fig,name):
        fig.savefig(figs/(name+'.png'),dpi=220,bbox_inches='tight')
        fig.savefig(figs/(name+'.pdf'),bbox_inches='tight')
        plt.close(fig)
    fig,axes=plt.subplots(3,2,figsize=(10.0,8.3),sharex=True,sharey=True,layout='constrained')
    lo=min(y.min(),*(p.min() for p in pred.values()))-3
    hi=max(y.max(),*(p.max() for p in pred.values()))+3
    for ax,kind,color in zip(axes.flat,KINDS,colors):
        ax.plot(d,y,color='#3D3D3D',lw=.8,label='Actual close')
        ax.plot(d[:cut],pred[kind][:cut],color=color,lw=1.5,label='Training fit')
        ax.plot(d[cut-1:],pred[kind][cut-1:],color=color,ls='--',lw=1.6,label='60-day test prediction')
        ax.axvspan(d[cut],d[-1],color='#E7EDF2',alpha=.8,zorder=-1)
        ax.set_title(NAMES[kind]);ax.grid(alpha=.18);ax.set_ylim(lo,hi)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6));ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.set_ylabel('Price (USD)')
    axes[0,0].legend(loc='best',fontsize=7,frameon=False)
    save(fig,'all_kernel_fits')
    fig,axes=plt.subplots(1,2,figsize=(10,3.5),layout='constrained')
    ax=axes[0]
    ax.plot(d[cut:],y[cut:],color='black',lw=2,label='Actual close')
    for kind,color in zip(KINDS,colors):
        if kind=='precomputed_rbf': continue
        ax.plot(d[cut:],pred[kind][cut:],color=color,lw=1.2,label=NAMES[kind])
    ax.axhline(result['baseline_last_close_price'],color='#666666',ls=':',label='Fixed last close')
    ax.set_title('Held-out 60 trading days');ax.set_ylabel('Price (USD)')
    ax.xaxis.set_major_locator(mdates.MonthLocator());ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.legend(fontsize=6.8,frameon=False);ax.grid(alpha=.18)
    names=[NAMES[k] for k in KINDS]+['Fixed last close']
    values=[result['models'][k]['test']['rmse'] for k in KINDS]+[result['baseline_fixed_last_close']['rmse']]
    axes[1].barh(names,values,color=colors+['#777777'])
    axes[1].invert_yaxis();axes[1].set_xlabel('Test RMSE (USD)');axes[1].set_title('Forecast error')
    for i,v in enumerate(values):axes[1].text(v+.12,i,f'{v:.2f}',va='center',fontsize=8)
    axes[1].set_xlim(0,max(values)*1.15)
    save(fig,'test_comparison')
    fig,axes=plt.subplots(1,2,figsize=(10,3.35),layout='constrained')
    for ax,label,title in zip(axes,['fit_ms','predict_ms'],['Complete fit time','Prediction time for 60 dates']):
        med=np.array([result['timings'][k][label]['median'] for k in KINDS])
        lower=np.array([result['timings'][k][label]['q25'] for k in KINDS])
        upper=np.array([result['timings'][k][label]['q75'] for k in KINDS])
        ax.barh([NAMES[k] for k in KINDS],med,color=colors,xerr=np.array([med-lower,upper-med]),capsize=3)
        ax.invert_yaxis();ax.set_xlabel('Milliseconds');ax.set_title(title)
        ax.grid(axis='x',alpha=.18)
    save(fig,'runtime_comparison')

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,default=Path(__file__).resolve().parent/'data/finance-charts-apple.csv')
    parser.add_argument('--out',type=Path,default=Path('results'))
    args=parser.parse_args()
    with threadpool_limits(limits=1): run(args.data,args.out)
