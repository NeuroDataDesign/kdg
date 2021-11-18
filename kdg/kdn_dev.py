from functools import total_ordering
from keras import layers
from .base import KernelDensityGraph
from sklearn.mixture import GaussianMixture
from tensorflow import keras
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
import numpy as np
from scipy.stats import multivariate_normal
from scipy.special import comb
import warnings
from sklearn.covariance import LedoitWolf
import matplotlib.pyplot as plt
import itertools

class kdn(KernelDensityGraph):

    def __init__(self,
        network,
        k = 1,
        polytope_compute_method = 'all', # 'all': all the FC layers, 'pl': only the penultimate layer
        T=2, 
        weighting_method = None, # 'TM', 'FM'
        verbose=True
        ):

        super().__init__()
        self.polytope_means = {}
        self.polytope_cov = {}
        self.network = network
        self.k = k
        self.polytope_compute_method = polytope_compute_method
        self.T = T
        self.weighting_method = weighting_method
        self.bias = {}
        self.verbose = verbose

        self.total_layers = len(self.network.layers)

        self.network_shape = []
        for layer in network.layers:
            self.network_shape.append(layer.output_shape[-1])

        self.num_fc_neurons = sum(self.network_shape)

    def _get_polytope_memberships(self, X):
        polytope_memberships = []
        last_activations = X

        # Iterate through neural network manually, getting node activations at each step
        for layer_id in range(self.total_layers):
            weights, bias = self.network.layers[layer_id].get_weights()

            # Calculate new activations based on input to this layer
            preactivation = np.matmul(last_activations, weights) + bias

             # get list of activated nodes in this layer
            if layer_id == self.total_layers - 1:
                binary_preactivation = (preactivation > 0.5).astype('int')
            else:
                binary_preactivation = (preactivation > 0).astype('int')
            
            if self.polytope_compute_method == 'pl':
                # determine the polytope memberships only based on the penultimate layer (uncomment )
                if layer_id == self.total_layers - 2:
                    polytope_memberships.append(binary_preactivation)

            if self.polytope_compute_method == 'all':
                # determine the polytope memberships only based on all the FC layers (uncomment)
                if layer_id < self.total_layers - 1:
                    polytope_memberships.append(binary_preactivation)
            
            # remove all nodes that were not activated
            last_activations = preactivation * binary_preactivation

        # Concatenate all activations for given observation
        polytope_obs = np.concatenate(polytope_memberships, axis = 1)
        polytope_memberships = [np.tensordot(polytope_obs, 2 ** np.arange(0, np.shape(polytope_obs)[1]), axes = 1)]

        self.num_fc_neurons = polytope_obs.shape[1] # get the number of total FC neurons under consideration
        
        return polytope_memberships

    def _get_activation_pattern(self, polytope_id):
        binary_string = np.binary_repr(polytope_id, width=self.num_fc_neurons)[::-1] 
        return np.array(list(binary_string)).astype('int')
    
    def _get_activation_paths(self, polytopes):
        n_samples = polytopes[0].shape[0]
        #reverse to save on calculation time using smaller later layers
        polytope_r = polytopes[::-1]
        place = [1]
        for layer in polytope_r:
            p = place[-1]
            place.append(p*layer.shape[1])

        #set error code
        #this value ensures final output will be negative for all nodes and code will run
        err = np.array([-place[-1]])
        #get paths values
        paths = [np.zeros(1, dtype=int) for n in range(n_samples)]
        for i, layer in enumerate(polytope_r):
            idx = layer*np.arange(layer.shape[1])*place[i]
            temp_paths = [None for n in range(n_samples)]
            for j in range(n_samples):
                active_nodes = idx[j, layer[j,:]>0]
                if len(active_nodes) == 0:
                    temp_paths[j] = err
                else: 
                    temp_paths[j] = np.concatenate([p + active_nodes for p in paths[j]])

            paths = temp_paths
            #print(paths)

        #convert to binary
        activation_paths = np.zeros((n_samples, place[-1]))
        for i, p in enumerate(paths):
            #if error occured, return None
            if any(p < 0): activation_paths[i] = -1
            else: activation_paths[i, p] = 1

        return activation_paths

    def _nCr(self, n, r):
        # return math.factorial(n) / (math.factorial(r) / math.factorial(n-r))
        return comb(n, r)
    
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
        X, y = check_X_y(X, y)
        self.labels = np.unique(y)

        feature_dim = X.shape[1]
        
        for label in self.labels:
            self.polytope_means[label] = []
            self.polytope_cov[label] = []

            X_ = X[np.where(y==label)[0]]
            polytope_memberships = self._get_polytope_memberships(X_)[0]
            unique_polytope_ids = np.unique(polytope_memberships) # get the unique polytopes
            
            if self.verbose:
                print("Number of Polytopes : ", len(polytope_memberships))
                print("Number of Unique Polytopes : ", len(unique_polytope_ids))
            
            polytope_member_count = [] # store the polytope member counts

            for polytope_id in unique_polytope_ids: # fit Gaussians for each unique non-singleton polytopes

                # get the activation pattern of the current polytope
                a_native = self._get_activation_pattern(polytope_id)
                
                # compute the weights
                weights = []

                for member_polytope_id in polytope_memberships:
                    a_foreign = self._get_activation_pattern(member_polytope_id)
                    
                    match_status = a_foreign == a_native
                    match_status = match_status.astype('int')

                    if self.weighting_method == 'TM' or self.weighting_method == None:
                        # weight based on the total number of matches
                        weight = np.sum(match_status)/self.num_fc_neurons

                    if self.weighting_method == 'FM':
                        # weight based on the first mistmatch
                        if len(np.where(match_status==0)[0]) == 0:
                            weight = 1.0
                        else:
                            first_mismatch_idx = np.where(match_status==0)[0][0]
                            weight = first_mismatch_idx / self.num_fc_neurons

                    layerwise_match_status = []
                    start = 0
                    for l in self.network_shape:
                        end = start + l
                        layerwise_match_status.append(match_status[start:end])
                        start = end

                    if self.weighting_method == 'PFM':
                        # weight based on the first mistmatch
                        penultimate_match_status = layerwise_match_status[-1]
                        if len(np.where(penultimate_match_status==0)[0]) == 0:
                            weight = 1.0
                        else:
                            first_mismatch_idx = np.where(penultimate_match_status==0)[0][0]
                            weight = first_mismatch_idx / self.num_fc_neurons

                    if self.weighting_method == 'EFM':
                        #pseudo-ensembled first mismatch - fast update
                        weight = 0
                        for layer in layerwise_match_status:
                            n = layer.shape[0] #length of layer
                            m = np.sum(layer) #matches
                            #k = nodes drawn before mismatch occurs
                            if m == n: #perfect match
                                weight += n/self.num_fc_neurons
                            elif m <= math.floor(n/2): #break if too few nodes match
                                break
                            else: #imperfect match, add scaled layer weight and break
                                layer_weight = m/(self.num_fc_neurons*(n-m+1))
                                weight += layer_weight
                                break

                    weights.append(weight)
                weights = np.array(weights)
                
                if self.weighting_method == None:
                    weights[weights != 1] = 0 # only use the data from the native polytopes
            
                idx = np.where(weights > 0)[0]

                polytope_size = len(idx)
                polytope_member_count.append(polytope_size)
                
                if polytope_size < self.T: # don't fit a gaussian to polytopes that has less members than the specified threshold
                    continue

                scales = weights[idx]/np.max(weights[idx])

                # apply weights to the data
                X_tmp = X_[idx].copy()
                polytope_mean_ = np.average(X_tmp, axis=0, weights=scales) # compute the weighted average of the samples 
                X_tmp -= polytope_mean_ # center the data

                # sqrt_weights = np.sqrt(weights)
                # sqrt_weights = np.expand_dims(sqrt_weights, axis=-1)
                sqrt_scales = np.sqrt(scales).reshape(-1,1) @ np.ones(feature_dim).reshape(1,-1)
                X_tmp *= sqrt_scales # scale the centered data with the square root of the weights

                # compute the paramters of the Gaussian underlying the polytope
                
                # LedoitWolf Estimator (uncomment)
                covariance_model = LedoitWolf(assume_centered=True)
                covariance_model.fit(X_tmp)
                polytope_cov_ = covariance_model.covariance_ * len(scales) / sum(scales)

                # store the mean and covariances
                self.polytope_means[label].append(
                        polytope_mean_
                )
                self.polytope_cov[label].append(
                        polytope_cov_
                )

            ## calculate bias for each label
            likelihoods = np.zeros(
                            (np.size(X_,0)),
                            dtype=float
                        )

            for polytope_idx,_ in enumerate(self.polytope_means[label]):
                likelihoods += np.nan_to_num(self._compute_pdf(X_, label, polytope_idx))

            likelihoods /= X_.shape[0]
            self.bias[label] = np.min(likelihoods)/(self.k * X_.shape[0])

            if self.verbose:
                plt.hist(polytope_member_count, bins=30)
                plt.xlabel("Number of Members")
                plt.ylabel("Number of Polytopes")
                plt.show()

    def _compute_pdf(self, X, label, polytope_idx):
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
        Calculate posteriors using the kernel density forest.
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
            total_polytopes = len(self.polytope_means[label])
            for polytope_idx,_ in enumerate(self.polytope_means[label]):
                likelihoods[:,ii] += np.nan_to_num(self._compute_pdf(X, label, polytope_idx))
            
            likelihoods[:,ii] = likelihoods[:,ii]/total_polytopes
            likelihoods[:,ii] += min(self.bias.values())
            
        proba = (likelihoods.T/(np.sum(likelihoods,axis=1)+1e-100)).T
        return proba

    def predict_proba_nn(self, X):
        r"""
        Calculate posteriors using the vanilla NN
        Parameters
        ----------
        X : ndarray
            Input data matrix.
        """
        X = check_array(X)

        proba = self.network.predict(X)
        return proba

    def predict(self, X):
        r"""
        Perform inference using the kernel density forest.
        Parameters
        ----------
        X : ndarray
            Input data matrix.
        """
        return np.argmax(self.predict_proba(X), axis = 1)