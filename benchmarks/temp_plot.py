# %% plot the result
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np 
filename1 = 'compare_weighted_kdf_kdn.csv'

df = pd.read_csv(filename1)
sample_size = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]#[1000,5000,10000]

#sample_size = [1000,5000,10000]

err_kdn_med = []
err_kdn_25_quantile = []
err_kdn_75_quantile = []

err_kdn_med_ = []
err_kdn_25_quantile_ = []
err_kdn_75_quantile_ = []

err_kdf_med = []
err_kdf_25_quantile = []
err_kdf_75_quantile = []

err_kdf_med_ = []
err_kdf_25_quantile_ = []
err_kdf_75_quantile_ = []
#clr = ["#e41a1c", "#f781bf", "#306998"]
#c = sns.color_palette(clr, n_colors=3)


for sample in sample_size:
    err_kdn = 1 - df['accuracy kdn'][df['sample']==sample]
    err_kdf = 1 - df['accuracy kdf'][df['sample']==sample]

    err_kdn_med.append(np.median(err_kdn))
    err_kdn_25_quantile.append(
            np.quantile(err_kdn,[.25])[0]
        )
    err_kdn_75_quantile.append(
        np.quantile(err_kdn,[.75])[0]
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

ax.plot(sample_size, err_kdn_med, c="k", label='KDN')
ax.fill_between(sample_size, err_kdn_25_quantile, err_kdn_75_quantile, facecolor='k', alpha=.3)

#ax.plot(sample_size, err_rf_med_, c="g", label='RF (feature selected)')
#ax.fill_between(sample_size, err_rf_25_quantile_, err_rf_75_quantile_, facecolor='g', alpha=.3)

ax.plot(sample_size, err_kdf_med, c="r", label='KDF')
ax.fill_between(sample_size, err_kdf_25_quantile, err_kdf_75_quantile, facecolor='r', alpha=.3)

#ax.plot(sample_size, err_kdf_med_, c="b", label='KDF (feteaure selected)')
#ax.fill_between(sample_size, err_kdf_25_quantile_, err_kdf_75_quantile_, facecolor='b', alpha=.3)

right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)

ax.set_xscale('log')
ax.set_xlabel('Sample size')
ax.set_ylabel('error')
ax.legend(frameon=False)

plt.savefig('plots/compare_weighted_kdf_kdn_v1.pdf')