#%%
import numpy as np
from kdg import kdf
from kdg.utils import sparse_parity
import pandas as pd
from sklearn.ensemble import RandomForestClassifier as rf
#%%
p = 20
p_star = 3
'''sample_size = np.logspace(
        np.log10(10),
        np.log10(5000),
        num=10,
        endpoint=True,
        dtype=int
        )'''
sample_size = [1000,5000,10000]
n_test = 5000
reps = 10
covarice_types = {'diag', 'full', 'spherical'}
criterion = 'bic'
n_estimators = 500
df = pd.DataFrame()
reps_list = []
accuracy_kdf = []
accuracy_kdf_ = []
accuracy_rf = []
sample_list = []
# %%
for sample in sample_size:
    print('Doing sample %d'%sample)
    for ii in range(reps):
        X, y = sparse_parity(
            sample,
            p_star=p_star,
            p=p
        )
        X_test, y_test = sparse_parity(
            n_test,
            p_star=p_star,
            p=p
        )

        #train kdf
        model_kdf = kdf(
            covariance_types = covarice_types,
            criterion = criterion, 
            kwargs={'n_estimators':n_estimators}
        )
        model_kdf.fit(X, y)
        accuracy_kdf.append(
            np.mean(
                model_kdf.predict(X_test) == y_test
            )
        )

        #train feature selected kdf
        model_kdf = kdf(
            covariance_types = covarice_types,
            criterion = criterion, 
            kwargs={'n_estimators':n_estimators}
        )
        model_kdf.fit(X[:,:3], y)
        accuracy_kdf_.append(
            np.mean(
                model_kdf.predict(X_test[:,:3]) == y_test
            )
        )

        #train rf
        model_rf = rf(n_estimators=n_estimators).fit(X, y)
        accuracy_rf.append(
            np.mean(
                model_rf.predict(X_test) == y_test
            )
        )
        reps_list.append(ii)
        sample_list.append(sample)

df['accuracy kdf'] = accuracy_kdf
df['feature selected kdf'] = accuracy_kdf_
df['accuracy rf'] = accuracy_rf
df['reps'] = reps_list
df['sample'] = sample_list

df.to_csv('high_dim_res_aic_kdf.csv')
# %% plot the result
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np 

filename1 = 'high_dim_res_aic_kdf.csv'

df = pd.read_csv(filename1)

sample_size = [1000,5000,10000]

err_rf_med = []
err_rf_25_quantile = []
err_rf_75_quantile = []

err_kdf_med = []
err_kdf_25_quantile = []
err_kdf_75_quantile = []

err_kdf_med_ = []
err_kdf_25_quantile_ = []
err_kdf_75_quantile_ = []
#clr = ["#e41a1c", "#f781bf", "#306998"]
#c = sns.color_palette(clr, n_colors=3)


for sample in sample_size:
    err_rf = 1 - df['accuracy rf'][df['sample']==sample]
    err_kdf = 1 - df['accuracy kdf'][df['sample']==sample]
    err_kdf_ = 1 - df['feature selected kdf'][df['sample']==sample]

    err_rf_med.append(np.median(err_rf))
    err_rf_25_quantile.append(
            np.quantile(err_rf,[.25])[0]
        )
    err_rf_75_quantile.append(
        np.quantile(err_rf,[.75])[0]
    )

    err_kdf_med.append(np.median(err_kdf))
    err_kdf_25_quantile.append(
            np.quantile(err_kdf,[.25])[0]
        )
    err_kdf_75_quantile.append(
        np.quantile(err_kdf,[.75])[0]
    )

    err_kdf_med_.append(np.median(err_kdf_))
    err_kdf_25_quantile_.append(
            np.quantile(err_kdf_,[.25])[0]
        )
    err_kdf_75_quantile_.append(
        np.quantile(err_kdf_,[.75])[0]
    )

sns.set_context('talk')
fig, ax = plt.subplots(1,1, figsize=(8,8))

ax.plot(sample_size, err_rf_med, c="k", label='RF', marker='.',markersize=14)
ax.fill_between(sample_size, err_rf_25_quantile, err_rf_75_quantile, facecolor='k', alpha=.3)

ax.plot(sample_size, err_kdf_med, c="r", label='KDF (BIC)', marker='.',markersize=14)
ax.fill_between(sample_size, err_kdf_25_quantile, err_kdf_75_quantile, facecolor='r', alpha=.3)

ax.plot(sample_size, err_kdf_med_, c="b", label='KDF (BIC and feteaure selected)', marker='.',markersize=14)
ax.fill_between(sample_size, err_kdf_25_quantile_, err_kdf_75_quantile_, facecolor='b', alpha=.3)

right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)

ax.set_xscale('log')
ax.set_xlabel('')
ax.set_ylabel('error')
ax.set_xticks([])
ax.text(1000, -0.040, "1000", fontsize=18)
ax.text(5000, -0.040, "5000", fontsize=18)
ax.text(10000, -0.040, "10000", fontsize=18)
ax.text(2000, -0.080, "Sample Size", fontsize=20)
ax.legend(frameon=False)

plt.savefig('plots/high_dim.pdf')

# %%
