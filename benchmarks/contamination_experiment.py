import seaborn as sns
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_blobs
from scipy.stats import multivariate_normal
from kdg.utils import hellinger
from kdg import kdf


def generate_distribution(points, grid_density=200, contamination_prop=0.25):
    r""" Generate two Gaussian distributions with uniform noise from KDE paper."""
    centers = np.array([[0, -3], [0, 3]])

    num_contamination = np.int32(points * contamination_prop)
    num_samples = points - num_contamination
    samples_per_blob = num_samples // 2
    
    # Generate Gaussian data
    X0, y = make_blobs(n_samples=samples_per_blob,
                       n_features=2,
                       centers=centers,
                       cluster_std=1)

    # Generate Contamination
    Xc = np.random.uniform(low=-6, high=6, size=(num_contamination, 2))

    # Generate true pdf
    x = np.linspace(-6, 6, grid_density)
    y = np.linspace(-6, 6, grid_density)
    xx, yy = np.meshgrid(x, y)
    pos = np.dstack((xx, yy))
    cov = np.array([[1, 0], [0, 1]])
    rv1 = multivariate_normal(centers[0], cov)
    rv2 = multivariate_normal(centers[1], cov)
    true_pdf = rv1.pdf(pos) + rv2.pdf(pos)
    true_pdf /= true_pdf.sum()

    return X0, Xc, true_pdf


def replicate_figure(X0, Xc, grid_density=100, centers=None):
    r""" Replicate KDE paper figure display. """
    plt.figure(figsize=(8, 8))
    plt.scatter(X0[:, 1], X0[:, 0], facecolors='none', edgecolors='black')
    plt.scatter(Xc[:, 1], Xc[:, 0], marker='x', color='red')
    plt.xlim([-6, 6])
    plt.ylim([-6, 6])
    plt.gca().set_aspect('equal')

    # Create contours
    centers = np.array([[0, -3], [0, 3]])
    x = np.linspace(-6, 6, grid_density)
    y = np.linspace(-6, 6, grid_density)
    xx, yy = np.meshgrid(x, y)
    pos = np.dstack((xx, yy))
    cov = np.array([[1, 0], [0, 1]])
    rv1 = multivariate_normal(centers[0], cov)
    rv2 = multivariate_normal(centers[1], cov)
    Z = rv1.pdf(pos) + rv2.pdf(pos)
    Z /= Z.sum()
    plt.contour(yy, xx, Z, levels=[Z.max() / 16, Z.max() / 8, Z.max() / 4, Z.max() / 2, Z.max() * 3 / 4])


def contamination_experiment(points, grid_density=200, n_estimators=500, uniform_size=200, contamination_prop=0.25, debug=False):
    """ Run a trial of the contamination experiment """
    # Generate Data
    X0, Xc, true_pdf = generate_distribution(points,
                                             grid_density=grid_density,
                                             contamination_prop=contamination_prop)
    class0 = np.random.uniform(low=-6, high=6, size=(uniform_size, 2))
    X_train = np.vstack((X0, Xc))
    y_train = np.ones(X_train.shape[0])    
    X_train = np.vstack((X_train, class0))
    y_train = np.hstack((y_train, np.zeros(class0.shape[0])))
    true_pdf_class1 = true_pdf.reshape(-1, 1)
    x = np.linspace(-6, 6, grid_density)
    y = np.linspace(-6, 6, grid_density)
    xx, yy = np.meshgrid(x, y)
    grid_samples = np.concatenate(
        (
            xx.reshape(-1, 1),
            yy.reshape(-1, 1)
        ),
        axis=1
        )

    model_kdf = kdf(kwargs={'n_estimators':n_estimators})
    model_kdf.fit(X_train, y_train)
    pdf_class1 = model_kdf.predict_pdf(grid_samples)[:, 1]
    #pdf_class0 = model_kdf.predict_pdf(grid_samples)[:, 0]

    h = hellinger(pdf_class1.reshape(-1, 1), true_pdf_class1)

    if debug:
        #### Show True Points ####
        replicate_figure(X0, Xc)
        plt.gca().set_title('True Distribution')

        #### Show true pdf ####
        plt.figure()
        plt.imshow(true_pdf.T)
        plt.colorbar()
        plt.title('True pdf')
    
        #### Show predicted pdf ####
        plt.figure()
        plt.imshow(np.flip(pdf_class1.reshape((grid_density, grid_density)).T, axis=0))
        plt.colorbar()
        plt.title('KDF Class 1 PDF')

    return h

    
#### Experiment from RKDE Paper ####
#points = 220
#n_uniform = 200
#h = contamination_experiment(points, grid_density=200, n_estimators=50, uniform_size=n_uniform, contamination_prop=0.09, debug=True)
#print(h)
#plt.show()

#### Run Experiment with Varying Samples ####
sample_size = np.logspace(
    np.log10(10),
    np.log10(10000),
    num=5,
    endpoint=True,
    dtype=int
    )
reps = 10
contamination_prop = 0.09

df = pd.DataFrame()
n_estimators=500
#n_uniform=500
p = 0.10
reps_list = []
sample_list = []
hellinger_dist_kdf = []

for sample in sample_size:
    print('Doing sample {}'.format(sample))
    for ii in range(reps):
        h = contamination_experiment(sample,
                                     grid_density=200,
                                     n_estimators=n_estimators,
                                     uniform_size=samples,  # TODO: class 0 uniform noise size?
                                     contamination_prop=p)
        hellinger_dist_kdf.append(h)
        
        reps_list.append(ii)
        sample_list.append(sample)
        print('Hellinger KDF: {}'.format(h))

df['reps'] = reps_list
df['sample'] = sample_list
df['hellinger_dist_kdf'] = hellinger_dist_kdf

hellinger_kdf_med = []
hellinger_kdf_25_quantile = []
hellinger_kdf_75_quantile = []

for sample in sample_size:
    curr = df['hellinger_dist_kdf'][df['sample']==sample]
    hellinger_kdf_25_quantile.append(
        np.quantile(curr, [.25])[0]
    )
    hellinger_kdf_75_quantile.append(
        np.quantile(curr, [.75])[0]
    )
    hellinger_kdf_med.append(np.median(curr))

sns.set_context('talk')
fig, ax = plt.subplots(1,1, figsize=(8,8))
ax.plot(sample_size, hellinger_kdf_med, c="r", label='KDF')
ax.fill_between(sample_size, hellinger_kdf_25_quantile, hellinger_kdf_75_quantile, facecolor='r', alpha=.3)

right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)

ax.set_xscale('log')
ax.set_xlabel('Sample size')
ax.set_ylabel('Hellinger Distance')
ax.legend(frameon=False)
plt.savefig('plots/contamination_test.pdf')

plt.show()
