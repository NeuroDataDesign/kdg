import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import seaborn as sns
from kdg.utils import generate_gaussian_parity
from kdg import kdf
from kdg import kdn
from sklearn.ensemble import RandomForestClassifier as rf
import tensorflow as tf
import keras 
from keras import layers


def label_noise_trial(n_samples, p=0.10, n_estimators=500):
    """Single label noise trial with proportion p of flipped labels."""
    X, y = generate_gaussian_parity(n_samples, cluster_std=0.5)
    X_test, y_test = generate_gaussian_parity(1000, cluster_std=0.5)

    # Generate noise and flip labels
    n_noise = np.int32(np.round(len(X) * p))
    noise_indices = random.sample(range(len(X)), n_noise)
    y[noise_indices] = 1 - y[noise_indices]


    #train kdn
    network = tf.keras.Sequential()
    #network.add(layers.Dense(2, activation="relu", input_shape=(2)))
    network.add(layers.Dense(3, activation='relu', input_shape=(2,)))
    network.add(layers.Dense(3, activation='relu'))
    network.add(layers.Dense(units=2, activation = 'softmax'))
    network.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(3e-4))
    
    model_kdn = kdn(network, covariance_types = cov_type, criterion = criterion)
    model_kdn.fit(X, y)
    
    error_kdn = 1 - np.mean(model_kdn.predict(X_test)==y_test)


    network.fit(X, tf.keras.utils.to_categorical(y), epochs=100, batch_size=32, verbose=False)
    predicted_label = np.argmax(network.predict(X_test), axis=1)
    error_nn = 1 - np.mean(predicted_label==y_test)

    #model_kdf = kdf(kwargs={"n_estimators": n_estimators})
    #model_kdf.fit(X, y)
    #error_kdf = 1 - np.mean(model_kdf.predict(X_test) == y_test)

    #model_rf = rf(n_estimators=n_estimators)
    #model_rf.fit(X, y)
    #error_rf = 1 - np.mean(model_rf.predict(X_test) == y_test)
    return error_kdn, error_nn


### Run the experiment with varying proportion of label noise
df = pd.DataFrame()
reps = 10
n_estimators = 500
n_samples = 5000

cov_type = 'full' #{'diag', 'full', 'spherical'}
criterion = None

err_kdn = []
err_nn = []
proportions = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
proportion_list = []
reps_list = []

for p in proportions:
    print("Doing proportion {}".format(p))
    for ii in range(reps):
        err_kdn_i, err_nn_i = label_noise_trial(
            n_samples=n_samples, p=p, n_estimators=n_estimators
        )
        err_kdn.append(err_kdn_i)
        err_nn.append(err_nn_i)
        reps_list.append(ii)
        proportion_list.append(p)
        print("KDN error = {}, NN error = {}".format(err_kdn_i, err_nn_i))

# Construct DataFrame
df["reps"] = reps_list
df["proportion"] = proportion_list
df["error_kdn"] = err_kdn
df["error_nn"] = err_nn

err_kdn_med = []
err_kdn_25_quantile = []
err_kdn_75_quantile = []
err_nn_med = []
err_nn_25_quantile = []
err_nn_75_quantile = []

for p in proportions:
    curr_kdn = df["error_kdn"][df["proportion"] == p]
    curr_nn = df["error_nn"][df["proportion"] == p]

    err_kdn_med.append(np.median(curr_kdn))
    err_kdn_25_quantile.append(np.quantile(curr_kdn, [0.25])[0])
    err_kdn_75_quantile.append(np.quantile(curr_kdn, [0.75])[0])

    err_nn_med.append(np.median(curr_nn))
    err_nn_25_quantile.append(np.quantile(curr_nn, [0.25])[0])
    err_nn_75_quantile.append(np.quantile(curr_nn, [0.75])[0])

# Plotting
sns.set_context("talk")
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.plot(proportions, err_kdn_med, c="r", label="KDN")
ax.fill_between(
    proportions, err_kdn_25_quantile, err_kdn_75_quantile, facecolor="r", alpha=0.3
)
ax.plot(proportions, err_nn_med, c="k", label="NN")
ax.fill_between(
    proportions, err_nn_25_quantile, err_nn_75_quantile, facecolor="k", alpha=0.3
)

right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)

ax.set_xlabel("Label Noise Proportion")
ax.set_ylabel("Error")
plt.title("Gaussian Parity Label Noise")
ax.legend(frameon=False)
plt.savefig("plots/label_noise_kdn_5000.pdf")
plt.show()
