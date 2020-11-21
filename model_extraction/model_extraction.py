import tensorflow as tf
import numpy as np
from model_extraction.state_extraction import get_states
from model_extraction.transition_extraction import get_transition_models
from model_extraction.prediction_explanation import PredictionExplanation
from model_extraction.utils import try_to_retrieve

class ExtractedModel:

    def __init__(self, rnn_model, x_data, **params):
        '''
        :param rnn_model: Original RNN model being approximated
        :param x_data: Input data of the RNN
        :param params: extra parameters
        '''

        # Create copy of passed-in parameters
        self.params = params

        # Create names of classes
        self.cls_names = try_to_retrieve("cls_names", **params)

        # Size of context window for detecting a transition, used in training
        self.transition_windows_size = try_to_retrieve("window_size", **params)

        # Maximum number of transition samples for training submodels
        self.max_transition_samples = try_to_retrieve("max_trans_samples", **params)

        # Maximum number of hidden state samples for clustering
        self.max_state_samples = try_to_retrieve("max_state_samples", **params)

        # Retrieve the layer to use for hidden representation
        self.layer_id = try_to_retrieve("layer_id", **params)

        # Retrieve the number of states to use
        self.n_states = try_to_retrieve("n_states", **params)

        # Retrieve the type of transition model to use
        self.tmodel = try_to_retrieve("t_model", **params)

        # Set a representation converter
        self.rep_converter = try_to_retrieve("rep_converter", **params)

        # Extract the submodel which returns the hidden state
        GRU_layer = rnn_model.layers[self.layer_id]
        GRU_layer.return_sequences = True
        reduced_model = tf.keras.Model(inputs=rnn_model.inputs, outputs=[GRU_layer.output])

        # Compute the starting state
        self.starting_state = self._get_start_state(x_data[0], reduced_model)

        # Extract clusterings and transition models
        self.clustering, self.cluster_cls_model = get_states(reduced_model,
                                                             rnn_model,
                                                             self.cls_names,
                                                             x_data,
                                                             self.n_states,
                                                             self.max_state_samples)

        self.transition_models = get_transition_models(reduced_model,
                                                       x_data,
                                                       self.clustering,
                                                       self.max_transition_samples,
                                                       self.transition_windows_size,
                                                       self.tmodel,
                                                       self.get_state_names(),
                                                       self.rep_converter)

        print("Model extracted successfully...")


    def get_state_names(self):
        '''
        Retrieve the names of the clusters/states
        '''
        return self.cluster_cls_model.state_names


    def _get_start_state(self, sequence, reduced_model):
        '''
        :param sequence: sample sequence of shape (n_timesteps, n_features)
        :param reduced_model: hidden prediction model
        '''

        # Compute initial starting cluster
        sequence = np.expand_dims(sequence, axis=0)
        s_hidden = reduced_model(sequence)[0:1, 0, :]
        s_curr = self.clustering.predict(s_hidden)[0]

        return s_curr


    def explain_sequence(self, sequence):
        '''
        :param sequence: numpy array of shape (1, n_timesteps, n_features)
        :return: extracted model outputs, with corresponding explanations for every output
        '''

        # Specify context window to feed into the transition model
        context_window = self.transition_windows_size

        n_samples, n_timesteps, _ = sequence.shape

        if n_samples != 1:
            raise ValueError("Currently only working for a single sample")

        outputs = np.zeros((n_timesteps), dtype=int)
        outputs[:context_window] = self.starting_state
        s_curr = self.starting_state
        explanations = PredictionExplanation()

        for t in range(context_window, n_timesteps):

            # Extract subsequence
            subsequence = sequence[:, t-context_window:t+1, :]

            transition_model = self.transition_models[s_curr]
            preds = transition_model.predict(subsequence)
            s_next = preds[0]

            expl = transition_model.explain(subsequence)
            explanations.add_state(self.get_state_names()[s_curr])
            explanations.add_subsequence(subsequence)
            explanations.add_transition_expl(expl)

            x = sequence[:, t, :]
            pred = self.cluster_cls_model.predict(s_next, x)
            outputs[t] = pred

            s_curr = s_next

        return outputs, explanations



    def batch_predict(self, sequences, return_sequences=False):
        '''
        Batched prediction of sequences
        :param sequences:  Input data of shape (n_samples, n_timesteps, n_features)
        :param return_sequences: Whether to return outputs for all timesteps, or just the final one
        :return: Extracted model outputs of shape (n_samples, n_timesteps) if return_sequences = True,
                                 or (n_samples, ) if if return_sequences = False
        '''

        # Specify context window to feed into the transition model
        context_window = self.transition_windows_size

        n_samples, n_timesteps, _, = sequences.shape
        outputs = np.zeros((n_samples, n_timesteps), dtype=int)
        outputs[:, :context_window] = self.starting_state
        s_curr = outputs[:, context_window]

        for t in range(context_window, n_timesteps):

            # Extract subsequence
            subsequence = sequences[:, t-context_window:t+1, :]

            s_next = np.zeros((s_curr.shape), dtype=int)

            for state in range(self.n_states):
                # Obtain next states
                state_sample_inds = np.where(s_curr == state)[0]

                if len(state_sample_inds) == 0: continue

                transition_model = self.transition_models[state]
                state_samples = subsequence[state_sample_inds]
                preds = transition_model.predict(state_samples)
                s_next[state_sample_inds] = preds

            for state in range(self.n_states):
                # Obtain the corresponding class labels
                state_sample_inds = np.where(s_next == state)[0]

                if len(state_sample_inds) == 0: continue

                x = sequences[:, t, :]
                x = x[state_sample_inds]
                pred = self.cluster_cls_model.predict(state, x)
                outputs[state_sample_inds, t] = pred

            s_curr = s_next

        if return_sequences:
            final_preds = outputs
        else:
            final_preds = outputs[:, -1]

        return final_preds





