#%%
import numpy as np
from kdg import kdf
from kdg.utils import gaussian_sparse_parity
import pandas as pd
from sklearn.ensemble import RandomForestClassifier as rf
import seaborn as sns
import matplotlib.pyplot as plt
#%%
p = 20
p_star = 3
sample_size = 5000
n_test = 1000
scales = [0.001, 0.1, 1, 10, 100, 1000, 10000, 10000]
reps = 10

n_estimators = 500
df = pd.DataFrame()
error_kdf = []
# %%
for scale in scales:
    print('Doing scale ', scale)
    
    acc = []
    for _ in range(reps):
        X, y = gaussian_sparse_parity(
                sample_size,
                p_star=p_star,
                p=p
            )
        X_test, y_test = gaussian_sparse_parity(
                n_test,
                p_star=p_star,
                p=p
            )

        #train kdf
        model_kdf = kdf(
            bw_scale=scale,
            kwargs={'n_estimators':n_estimators}
        )
        model_kdf.fit(X, y)
        acc.append(
                np.mean(
                    model_kdf.predict(X_test) == y_test
                )
            )
    
    error_kdf.append(
        1 - np.mean(acc)
    )
#%%
sns.set_context('talk')
fig, ax = plt.subplots(1,1, figsize=(8,8))

ax.plot(scales, error_kdf, c="b", label='KDF')
right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)

ax.set_xscale('log')
ax.set_xlabel('Scales')
ax.set_ylabel('error')

ax.legend(frameon=False)

plt.savefig('plots/high_dim_gaussian_bw_scales.pdf')