from model_extraction.transition_models import StaticModel, DNNTransitionModel, DTTransitionModel
from sklearn.model_selection import train_test_split

'''
Code for training/preprocessing transition models
'''
import numpy as np


def extract_switching_subsamples(sample_labels, sample_data, src, target, window):
    '''
    Extract subsequences where the model transitioned from state src, to state target
    :param sample_labels: state labels for the sample timesteps, shape (n_timesteps, 1)
    :param sample_data: sequence sample, shape (n_timesteps, n_features)
    :param src: source state
    :param target: target state
    :return: a list of sequences where there was a transition from state src to state target
    '''

    # Get the indices of all items in the sample where there was a transition
    transition_indices = [i for i in range(1, sample_labels.shape[0])
                          if sample_labels[i-1] == src and sample_labels[i] == target]

    # Extract the subsamples, by using the transition indices and a context window
    subsamples = [sample_data[i-window:i + 1] for i in transition_indices
                  if (i - window) >= 0]

    return subsamples


def extract_transition_sequences(reduced_model, x_data, clustering, max_samples, window):
    '''
    Retrieve subsequences where the model transitioned from one hidden state to another
    :param rnn_model: Original RNN model being approximated
    :param x_data: Training sequence data used for model extraction
    :param clustering: Clustering of the hidden states
    :return: An array (n_states, n_states), where element [i][j] contains training samples for transitions
             from state i to state j
    '''

    # Generate the hidden states for all data points, and for all timesteps
    hidden_states = reduced_model(x_data).numpy()

    # Flatten out the timesteps (such that now every timestep is a sample)
    (n_samples, n_timesteps, n_features) = hidden_states.shape
    unrolled_samples = np.reshape(hidden_states, (n_samples * n_timesteps, n_features))

    # Compute the clusters that the samples belong to
    # Effectively it is a matrix |n_samples| * |n_timesteps|, where M[i][j] gives the hidden cluster the model was in
    # at sample i, timestep j
    pred_clusters = clustering.predict(unrolled_samples)
    pred_clusters = np.reshape(pred_clusters, (n_samples, n_timesteps))

    clusters = clustering.cluster_centers_
    transition_samples = []

    # Initialise the transition sample array
    for i in range(len(clusters)):
        transition_samples.append([])
        for j in range(len(clusters)):
            transition_samples[i].append([])

    # Populate the transition sample array
    # In the end, transition_samples[i][j] holds subsequences during which the model transitioned from state i to state j
    for i in range(len(clusters)):
        for j in range(len(clusters)):

            new_samples = []

            # Retrieve transition subsamples from current sample (note, a single sequence might have multiple
            # transitions along the timesteps)
            for sample_id in range(len(x_data)):
                new_samples += extract_switching_subsamples(pred_clusters[sample_id], x_data[sample_id], i, j, window)

            if len(new_samples) > 0:

                # Combine all the samples into a numpy array
                new_samples = np.stack(new_samples)

                # If array is too large - truncate it to the maximum limit
                if max_samples is not None:
                    inds = np.arange(new_samples.shape[0])
                    np.random.shuffle(inds)
                    inds = inds[:max_samples]
                    new_samples = new_samples[inds]

                transition_samples[i][j] = new_samples

            else:
                transition_samples[i][j] = None

    return transition_samples


def train_transition_model(x_data, y_data, model="dt", state_names=None, rep_converter=None):
    '''
    Code for training transition model for prediction which state the transition will occur to/from
    :param x_train: input training sequence data
    :param y_data: labels for transitions
    :return: trained transition model
    '''

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.15)

    if model == "dnn":
        transition_model = DNNTransitionModel(x_data=x_train, y_data=y_train, state_names=state_names, representation_converter=rep_converter)
    elif model == "dt":
        transition_model = DTTransitionModel(state_names, representation_converter=rep_converter)
    else:
        raise ValueError("Unexpected model type")

    transition_model.fit(x_train, y_train)

    acc = transition_model.evaluate(x_test, y_test)

    print("Transition model accuracy: ", acc, "     n_train: ", x_train.shape[0])

    return transition_model


def get_transition_models(reduced_model, x_data, clustering, max_samples, window, model="dt", state_names=None, rep_converter=None):
    '''
    Given an RNN model and approximated states (returned by _get_states), produce a list of transition models,
    modelling inter-state transitions
    :param rnn_model: RNN model states and transitions are extracted from
    :param states: States extracted from the RNN model
    :return: A list of size |states|. Element i is a model predicting the next state from state i, given a new input
    '''

    transition_models = []

    # Transition samples is a list of size |states|
    # where transition_samples[i] returns a pair (x_data, y_data) of transitions to corresponding new states
    transition_samples = extract_transition_sequences(reduced_model, x_data, clustering, max_samples, window)

    for s1 in range(clustering.n_clusters):

        s1_transition_samples = []
        s1_transition_labels = []

        # Note: some transitions may not occur, thus, need to retain list of "effective names"
        effective_state_names = []

        for i, s2 in enumerate(range(clustering.n_clusters)):

            # Retrieve all transitions from state 1 to state 2
            transition_sequences = transition_samples[s1][s2]

            # Check if there were no transitions between clusters
            if transition_sequences is None: continue

            dest_labels = np.array([s2 for _ in range(transition_sequences.shape[0])])

            s1_transition_samples.append(transition_sequences)
            s1_transition_labels.append(dest_labels)
            effective_state_names.append(state_names[i])

        if len(s1_transition_samples) > 0:

            # Obtain all transitions from state 1 to any other state, and corresponding labels
            s1_transition_samples = np.concatenate(s1_transition_samples)
            s1_transition_labels = np.concatenate(s1_transition_labels)

            # Train the model on this data
            transition_model = train_transition_model(s1_transition_samples, s1_transition_labels, model,
                                                      state_names=effective_state_names, rep_converter=rep_converter )

        else:
            # Edge case: if there were no transitions from a state, create self-loop
            transition_model = StaticModel(s1)

        transition_models.append(transition_model)

    return transition_models
