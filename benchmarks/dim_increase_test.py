#%%
import numpy as np
from kdg import kdf
from kdg.utils import sparse_parity
import pandas as pd
from sklearn.ensemble import RandomForestClassifier as rf
# %%
reps = 10
dim = range(1,21)
covarice_types = {'diag', 'full', 'spherical'}
criterion = 'bic'
n_estimators = 500
train_sample = 1000
test_sample = 1000
df = pd.DataFrame()
reps_list = []
accuracy_kdf = []
accuracy_kdf_ = []
accuracy_rf = []
dims = []
# %%
for p_star in dim:
    print('Doing dim %d'%p_star)
    for ii in range(reps):
        X, y = sparse_parity(
            train_sample,
            p_star=p_star,
            p=p_star
        )
        X_test, y_test = sparse_parity(
            test_sample,
            p_star=p_star,
            p=p_star
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

        #train rf
        model_rf = rf(n_estimators=n_estimators).fit(X, y)
        accuracy_rf.append(
            np.mean(
                model_rf.predict(X_test) == y_test
            )
        )
        reps_list.append(ii)
        dims.append(p_star)

df['accuracy kdf'] = accuracy_kdf
df['accuracy rf'] = accuracy_rf
df['dimension'] = dims
df['reps'] = reps_list

df.to_csv('increasing_dim_res_bic_kdf.csv')
# %%
# %% plot the result
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np 

filename1 = 'increasing_dim_res_bic_kdf.csv'

df = pd.read_csv(filename1)

dim = range(1,21)

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


for p_star in dim:
    err_rf = 1 - df['accuracy rf'][df['dimension']==p_star]
    err_kdf = 1 - df['accuracy kdf'][df['dimension']==p_star]

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

sns.set_context('talk')
fig, ax = plt.subplots(1,1, figsize=(8,8))

ax.plot(dim, err_rf_med, c="k", label='RF')
ax.fill_between(dim, err_rf_25_quantile, err_rf_75_quantile, facecolor='k', alpha=.3)

ax.plot(dim, err_kdf_med, c="r", label='KDF (BIC)')
ax.fill_between(dim, err_kdf_25_quantile, err_kdf_75_quantile, facecolor='r', alpha=.3)

right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)

#ax.set_xscale('log')
ax.set_xlabel('dimension')
ax.set_ylabel('error')
ax.legend(frameon=False)
ax.set_xticks([1,5,10,15,20])

plt.savefig('plots/increasing_dim.pdf')
# %%
