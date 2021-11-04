#%%
import numpy as np
from kdg import kdf
from kdg import weighted_kdn
from kdg.utils import gaussian_sparse_parity, trunk_sim
from kdg.utils import generate_gaussian_parity, pdf, hellinger
import pandas as pd #TODO: Add pandas to setup.py file
from sklearn.ensemble import RandomForestClassifier as rf
import tensorflow as tf
import keras
from keras import layers
import seaborn as sns
import matplotlib.pyplot as plt
#%%


#TODO: Move get_colors to 'functions'
def get_colors(colors, inds):
    c = [colors[i] for i in inds]
    return c

#TODO: Move plotting to 'functions'
def plot_xor_nxor(data, labels, title):
    colors = sns.color_palette("Dark2", n_colors=2)
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.scatter(data[:, 0], data[:, 1], c=get_colors(colors, labels), s=50)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=30)
    plt.tight_layout()
    ax.axis("off")
    #plt.show()


p = 20
p_star = 3
#sample_size = np.logspace(
#        np.log10(10),
#        np.log10(20000),
#        num=12,
#        endpoint=True,
#        dtype=int
#        )
sample_size = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]#[1000,5000,10000]
n_test = 1000
reps = 20

cov_type = 'full' #{'diag', 'full', 'spherical'}
criterion = None

n_estimators = 500
df = pd.DataFrame()
reps_list = []
accuracy_kdf = []
accuracy_kdf_ = []
accuracy_kdn = []
accuracy_kdn_ = []
sample_list = []
hellinger_dist_kdf = []
hellinger_dist_kdn = []
# %%
for sample in sample_size: 
    print('Doing sample %d'%sample)
    for ii in range(reps):
        '''
        Earlier implementation of gaussian parity for KDF 
        X, y = gaussian_sparse_parity(
            sample,
            p_star=p_star,
            p=p
        )
        X_test, y_test = gaussian_sparse_parity(
            n_test,
            p_star=p_star,
            p=p
        )''' 

        X, y = generate_gaussian_parity(sample, cluster_std=0.5)
        X_test, y_test = generate_gaussian_parity(1000, cluster_std=0.5)

        plot_xor_nxor(X, y, 'Training: Gaussian XOR')
        plt.savefig('./data_gen/Training_XOR_{}_{}.png'.format(sample, ii))
        plot_xor_nxor(X_test, y_test, 'Gaussian XOR')
        plt.savefig('./data_gen/Testing_XOR_{}_{}.png'.format(sample, ii))
        plt.close('all')

        #Creating setup for hellinger distance tests 
        p = np.arange(-1,1,step=0.006)
        q = np.arange(-1,1,step=0.006)
        xx, yy = np.meshgrid(p,q)
        grid_samples = np.concatenate(
                (
                    xx.reshape(-1,1),
                    yy.reshape(-1,1)
                ),
                axis=1
        ) 

        true_pdf_class1 = np.array([pdf(x, cov_scale=0.5) for x in grid_samples]).reshape(-1,1)
        true_pdf = np.concatenate([true_pdf_class1, 1-true_pdf_class1], axis = 1)

 

        #train kdf
        model_kdf = kdf(
            kwargs={'n_estimators':n_estimators}
        )
        
        model_kdf.fit(X, y)
        accuracy_kdf.append(
            np.mean(
                model_kdf.predict(X_test) == y_test
            )
        )
        
        #Hellinger comparison
        proba_kdf = model_kdf.predict_proba(grid_samples)
        err_kdf = 1 - np.mean(model_kdf.predict(X_test)==y_test)
        hellinger_dist_kdf.append(hellinger(proba_kdf, true_pdf))


        #train kdn
        network = tf.keras.Sequential()
    #network.add(layers.Dense(2, activation="relu", input_shape=(2)))
        network.add(layers.Dense(3, activation='relu', input_shape=(2,)))
        network.add(layers.Dense(3, activation='relu'))
        network.add(layers.Dense(units=2, activation = 'softmax'))

        model_kdn = weighted_kdn(network, weighting_method = 'EFM')
        model_kdn.fit(X, y)

        accuracy_kdn.append(
            np.mean(
                model_kdn.predict(X_test) == y_test
            )
        )

        #Hellinger comparison
        proba_kdn = model_kdn.predict_proba(grid_samples)
        err_kdn = 1 - np.mean(model_kdn.predict(X_test)==y_test)
        hellinger_dist_kdn.append(hellinger(proba_kdn, true_pdf))

        #accuracy_rf.append(
        #    np.mean(
        #        model_kdf.rf_model.predict(X_test) == y_test
        #    )
        #)
        reps_list.append(ii)
        sample_list.append(sample)
        print("Accuracy_KDF: ", accuracy_kdf)
        print("Accuracy_KDN: ", accuracy_kdn)
        #train feature selected kdf
        

df['accuracy kdf'] = accuracy_kdf
#df['feature selected kdf'] = accuracy_kdf_
df['accuracy kdn'] = accuracy_kdn
#df['feature selected rf'] = accuracy_rf_
df['reps'] = reps_list
df['sample'] = sample_list
df['hellinger_dist_kdf'] = hellinger_dist_kdf
df['hellinger_dist_kdn'] = hellinger_dist_kdn

df.to_csv('compare_weighted_kdf_kdn.csv')
# %% plot the result
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np 

filename1 = 'compare_weighted_kdf_kdn.csv'

df = pd.read_csv(filename1)

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

# %%
