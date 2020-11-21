import numpy as np
from sklearn.cluster import KMeans
from model_extraction.state_models import MajorityVotingClsModel
from model_extraction.utils import pred_label_thresholded


def get_state_data(reduced_model, rnn_model, x_data, max_samples=None):

    # Generate the hidden states and labels for all data points, and for all timesteps
    hidden_states = reduced_model(x_data).numpy()
    hidden_labels = pred_label_thresholded(rnn_model, x_data)

    # Flatten out the timesteps (such that now every timestep is a sample)
    (n_samples, n_timesteps, n_features) = hidden_states.shape
    unrolled_samples = np.reshape(hidden_states, (n_samples*n_timesteps, n_features))
    unrolled_labels = np.reshape(hidden_labels, (n_samples *n_timesteps))

    if max_samples is not None:
        # Extract subset of hidden points
        inds = np.arange(unrolled_samples.shape[0])
        np.random.shuffle(inds)
        inds = inds[:max_samples]
        unrolled_samples = unrolled_samples[inds]
        unrolled_labels = unrolled_labels[inds]

    return unrolled_samples, unrolled_labels


def get_states(reduced_model, rnn_model, cls_names, x_data, n_clusters, max_samples=None):
    '''
    Given an RNN model, retrieve hidden state groupings, and the mapping from a state to a corresponding class
    :param rnn_model: The RNN model from which the states and transitions are extracted
    :param x_data: RNN training data, shape: (n_samples, n_timesteps, n_features)
    :return: (states, state_to_class_map). states - list of state coordinates of cluster centroids.
             state_to_class_map: map from state index to class id
    '''

    hidden_samples, hidden_labels = get_state_data(reduced_model, rnn_model, x_data, max_samples)

    # Cluster the samples and compute mapping to class labels
    clustering = KMeans(n_clusters=n_clusters).fit(hidden_samples)

    # Create model for predicting label from given state and data point
    cluster_cls_model = MajorityVotingClsModel(n_clusters, cls_names, clustering, hidden_samples, hidden_labels)

    return clustering, cluster_cls_model
