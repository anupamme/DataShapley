import numpy as np

from DShap import DShap
from shap_utils import convergence_plots, label_generator
'''
1. noise level: 
2. select a simplistic model
3. create data: nominal
4. add noise
5. calculate shapely value.
6. plot shapely value in both cases.
7. 
'''
problem = 'classification'
model = 'logistic'
num_test = 10
directory = './temp'
d = 1
train_size = 5
train_total = 50
_param = 1.0

import pdb
pdb.set_trace()
X_raw = np.random.multivariate_normal(mean=np.zeros(d), cov = np.eye(d), 
                                          size=train_size + train_total)
_, y_raw, _, _ = label_generator(
        problem, X_raw, param = _param)
X, y = X_raw[:train_size], y_raw[:train_size]
X_test, y_test = X_raw[train_size:], y_raw[train_size:]

dshap = DShap(X, y, X_test, y_test, num_test, model_family=model, metric='accuracy',
              directory=directory, seed=2, problem=problem)
dshap.run(100, 0.1, g_run=False)
dshap.merge_results()
convergence_plots(dshap.marginals_tmc)
convergence_plots(dshap.marginals_g)

dshap.performance_plots([dshap.vals_tmc, dshap.vals_g, dshap.vals_loo], num_plot_markers=20,
                       sources=dshap.sources)