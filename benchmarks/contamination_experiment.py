import seaborn as sns
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import loadmat
from sklearn.datasets import make_blobs
from scipy.stats import multivariate_normal
from kdg.utils import hellinger
from kdg import kdf


def hellinger1d(p, q):
    """ Hellinger distance for 1D/flattened inputs. """
    return np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / np.sqrt(2)


def generate_distribution(points, grid_density=200, contamination_prop=0.25):
    """ Generate two Gaussian distributions with uniform noise."""
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
    true_pdf /= 2

    return X0, Xc, true_pdf


def replicate_figure(X0, Xc, grid_density=100, centers=None):
    """ Replicate figure similar to KDE paper."""
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
    #Z /= Z.sum()
    Z /= 2
    plt.contour(yy, xx, Z, levels=[Z.max() / 16, Z.max() / 8, Z.max() / 4, Z.max() / 2, Z.max() * 3 / 4])


def contamination_experiment(points, grid_density=200, n_estimators=500, uniform_size=200, contamination_prop=0.25, debug=False):
    """ Run a trial of the contamination experiment """
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
    h = hellinger1d(pdf_class1.reshape(-1, 1).flatten(), true_pdf_class1.flatten())

    # Visualization utilities
    if debug:
        # Show True Points
        replicate_figure(X0, Xc)
        plt.gca().set_title('True Distribution')

        # Show true pdf ####
        plt.figure()
        plt.imshow(np.flip(true_pdf.T, axis=0))
        plt.colorbar()
        plt.title('True pdf')
    
        # Show predicted pdf
        plt.figure()
        plt.imshow(np.flip(pdf_class1.reshape((grid_density, grid_density)).T, axis=0))
        plt.colorbar()
        plt.title('KDF Class 1 PDF')

    return h

#### Experiment from RKDE Paper ####
points = 1000
n_uniform = 500
h = contamination_experiment(points, grid_density=200, n_estimators=500, uniform_size=n_uniform, contamination_prop=0.09, debug=True)
print('Test hellinger = {}'.format(h))
plt.show()

#### Run Experiment with Varying Samples ####
sample_size = np.logspace(
    np.log10(100),
    np.log10(10000),
    num=10,
    endpoint=True,
    dtype=int
    )
reps = 10
df = pd.DataFrame()
n_estimators = 500
n_uniform = 1000
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
                                     uniform_size=n_uniform,
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

for sample in sample_size:  # assumes Matlab used the same sample sizes
    curr = df['hellinger_dist_kdf'][df['sample']==sample]
    hellinger_kdf_25_quantile.append(
        np.quantile(curr, [.25])[0]
    )
    hellinger_kdf_75_quantile.append(
        np.quantile(curr, [.75])[0]
    )
    hellinger_kdf_med.append(np.median(curr))

## Load from matlab
mat = loadmat('../rkde_code/rkde_exp.mat')

hellinger_rkde_med = np.squeeze(mat['hellinger_rkde_med'])
hellinger_rkde_25_quantile = np.squeeze(mat['hellinger_rkde_25_quantile'])
hellinger_rkde_75_quantile = np.squeeze(mat['hellinger_rkde_75_quantile'])
df.to_csv('sim_res/contam_exp_1000noise.csv')


## Plotting
sns.set_context('talk')
fig, ax = plt.subplots(1,1, figsize=(8,8))
ax.plot(sample_size, hellinger_kdf_med, c="r", label='KDF')
ax.fill_between(sample_size, hellinger_kdf_25_quantile, hellinger_kdf_75_quantile, facecolor='r', alpha=.3)
ax.plot(sample_size, hellinger_rkde_med, c="k", label='RKDE')
ax.fill_between(sample_size, hellinger_rkde_25_quantile, hellinger_rkde_75_quantile, facecolor='k', alpha=.3)

right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)

ax.set_xscale('log')
ax.set_xlabel('Sample size')
ax.set_ylabel('Hellinger Distance')
ax.legend(frameon=False)
plt.savefig('plots/contamination_test_1000.pdf')

plt.show()
