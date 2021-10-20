from .base import KernelDensityGraph
from sklearn.mixture import GaussianMixture
from tensorflow import keras
from keras import layers
from keras import Model
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
import numpy as np
from scipy.stats import multivariate_normal
import warnings
from sklearn.covariance import LedoitWolf
import matplotlib.pyplot as plt

class kdcnn(KernelDensityGraph):

    def __init__(self,
        network,
        num_fc_layers,
        covariance_types = 'full', 
        criterion = None,
        compile_kwargs = {
            "loss": "categorical_crossentropy",
            "optimizer": keras.optimizers.Adam(3e-4)
            },
        fit_kwargs = {
            "epochs": 10,
            "batch_size": 256,
            "verbose": True
            }
        ):
        super().__init__()
        self.polytope_means = {}
        self.polytope_cov = {}
        self.network = network
        self.num_fc_layers = num_fc_layers
        self.encoder = None # convolutional encoder
        self.compile_kwargs = compile_kwargs
        self.fit_kwargs = fit_kwargs
        self.covariance_types = covariance_types
        self.criterion = criterion
        self.num_fc_neurons = 0

    def _get_polytopes(self, X):
        r"""
        Get the polytopes (neural network activation paths) for a given set of observations.
        
        Parameters
        ----------
        X : ndarray
            Input data matrix.

        num_fc_layers: int
            Number of fully-connected layers in the CNN
            
        Returns
        -------
        polytope_memberships : binary list-of-lists
                               Each list represents activations of nodes in the neural network for a given observation
                               0 = not activated; 1 = activated
        """
        polytope_memberships = []
        last_activations = X
        total_layers = len(self.network.layers)
        fully_connected_layers = np.arange(total_layers-self.num_fc_layers, total_layers, 1)

        # Iterate through neural network manually, getting node activations at each step
        for layer_id in fully_connected_layers:
            weights, bias = self.network.layers[layer_id].get_weights()
            
            # Calculate new activations based on input to this layer
            preactivation = np.matmul(last_activations, weights) + bias
            
            # get list of activated nodes in this layer
            if layer_id == total_layers - 1:
                binary_preactivation = (preactivation > 0.5).astype('int')
            else:
                binary_preactivation = (preactivation > 0).astype('int')
            
            # determine the polytope memberships only based on the penultimate layer
            if layer_id == total_layers - 2:
              polytope_memberships.append(binary_preactivation)

            # # determine the polytope memberships only based on all the FC layers
            # polytope_memberships.append(binary_preactivation)
            
            # remove all nodes that were not activated
            last_activations = preactivation * binary_preactivation
          
        #Concatenate all activations for given observation
        polytope_obs = np.concatenate(polytope_memberships, axis = 1)
        polytope_memberships = [np.tensordot(polytope_obs, 2 ** np.arange(0, np.shape(polytope_obs)[1]), axes = 1)]
        self.num_fc_neurons = polytope_obs.shape[1]
        return polytope_memberships

    def fit(self, X, y):
        r"""
        Fits the kernel density forest.
        Parameters
        ----------
        X : ndarray
            Input data matrix.
        y : ndarray
            Output (i.e. response) data matrix.
        """
        
        self.labels = np.unique(y)
        feature_dim = X.shape[1]

        # get the encoder outputs
        self.encoder = Model(self.network.input, self.network.layers[-(self.num_fc_layers + 1)].output)
        X = self.encoder.predict(X)
        
        for label in self.labels:
            print("label : ", label)
            self.polytope_means[label] = []
            self.polytope_cov[label] = []
            
            # Get all training items that match our given label
            X_ = X[np.where(y==label)[0]]
            
            # Calculate polytope memberships for each observation in X_
            polytope_memberships = self._get_polytopes(X_)[0]
            unique_polytopes = np.unique(polytope_memberships) # get the unique polytopes
            print("Number of Polytopes : ", len(polytope_memberships))
            print("Number of Unique Polytopes : ", len(unique_polytopes))

            polytope_member_count = [] # store the polytope member counts
            for polytope in unique_polytopes:
                idx = np.where(polytope_memberships==polytope)[0] # collect the samples that belong to the current polytope
                polytope_member_count.append(len(idx))
                
                if len(idx) < 10: # don't fit a gaussian to singleton polytopes
                    continue
                
                # get the activation pattern of the current polytope
                current_polytope_activation = np.binary_repr(polytope, width=self.num_fc_neurons)[::-1] 

                # compute the weights
                weights = []
                for member in polytope_memberships:
                    member_activation = np.binary_repr(member, width=self.num_fc_neurons)[::-1] 
                    
                    # # weight based on the total number of matches
                    # weight = np.sum(np.array(list(current_polytope_activation))==np.array(list(member_activation)))/self.num_fc_neurons
                
                    ## weight based on the first mistmatch 
                    match_status = np.array(list(current_polytope_activation))==np.array(list(member_activation))
                    if len(np.where(match_status.astype('int')==0)[0]) == 0:
                        weight = 1.0
                    else:
                        first_mismatch_idx = np.where(match_status.astype('int')==0)[0][0]
                        weight = first_mismatch_idx / self.num_fc_neurons
                    
                    weights.append(weight)

                weights = np.array(weights)

                X_tmp = X_.copy()
                polytope_mean_ = np.average(X_tmp, axis=0, weights=weights) # compute the weighted average of the samples 
                X_tmp -= polytope_mean_ # center the data

                sqrt_weights = np.sqrt(weights)
                sqrt_weights = np.expand_dims(sqrt_weights, axis=-1)
                X_tmp *= sqrt_weights # scale the centered data with the square root of the weights

                # compute the paramters of the Gaussian underlying the polytope
                 
                ## Gaussian Mixture Model
                # gm = GaussianMixture(n_components=1, covariance_type=self.covariance_types, reg_covar=1e-4).fit(X_tmp)
                # polytope_mean_ = gm.means_[0]
                # polytope_cov_ = gm.covariances_[0]
                
                # LedoitWolf Estimator
                covariance_model = LedoitWolf(assume_centered=True)
                covariance_model.fit(X_tmp)
                polytope_cov_ = covariance_model.covariance_ * len(weights) / sum(weights)

                # store the mean and covariances
                self.polytope_means[label].append(
                        polytope_mean_
                )
                self.polytope_cov[label].append(
                        polytope_cov_
                )

            plt.hist(polytope_member_count, bins=30)
            plt.xlabel("Number of Members")
            plt.ylabel("Number of Polytopes")
            plt.show()

    def _compute_pdf(self, X, label, polytope_idx):
        r"""
        Calculate probability density function using the kernel density network for a given group.
        Parameters
        ----------
        X : ndarray
            Input data matrix.
        label : string
                A single group we want the PDF for
        polytope_idx : index of a polytope, within label
        """
        polytope_mean = self.polytope_means[label][polytope_idx]
        polytope_cov = self.polytope_cov[label][polytope_idx]

        var = multivariate_normal(
            mean=polytope_mean, 
            cov=polytope_cov, 
            allow_singular=True
            )

        likelihood = var.pdf(X)
        return likelihood

    def predict_proba(self, X):
        r"""
        Calculate posteriors using the kernel density network.
        Parameters
        ----------
        X : ndarray
            Input data matrix.
        """
        X = check_array(X)

        likelihoods = np.zeros(
            (np.size(X,0), len(self.labels)),
            dtype=float
        )
        
        for ii,label in enumerate(self.labels):
            for polytope_idx,_ in enumerate(self.polytope_means[label]):
                likelihoods[:,ii] += np.nan_to_num(self._compute_pdf(X, label, polytope_idx))

        proba = (likelihoods.T/(np.sum(likelihoods,axis=1)+1e-100)).T
        return proba

    def predict(self, X):
        r"""
        Perform inference using the kernel density forest.
        Parameters
        ----------
        X : ndarray
            Input data matrix.
        """
        # get the encoder outputs
        X = self.encoder.predict(X)
        return np.argmax(self.predict_proba(X), axis = 1)