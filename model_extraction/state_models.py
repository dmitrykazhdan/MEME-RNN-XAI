import numpy as np
from sklearn.model_selection import train_test_split
from model_extraction.transition_models import StaticModel

'''
Class for predicting the class labels from the states
'''

class MajorityVotingClsModel():

    def __init__(self, n_states, cls_names, clustering, x_data, y_data):

        # Initialisation
        self.n_states = n_states
        self.cls_names = cls_names
        self.state_model_map = {}
        self.cls_dist = []
        self.state_names = []

        # Train prediction model (using majority voting)
        self._fit(clustering, x_data, y_data)


    def _fit(self, clustering, x_data, y_data):

        pred = np.array(clustering.labels_)

        for center_id in range(self.n_states):

            # Find all points of the current cluster
            cluster_ids = np.squeeze(np.argwhere(pred == center_id))

            # Find corresponding data and labels of these points
            data = x_data[cluster_ids]
            labels = y_data[cluster_ids]

            x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.15)

            # Retrieve distribution of points accross classes
            bincount = np.bincount(labels)
            percentage = bincount / np.sum(bincount) * 100
            self.cls_dist.append(percentage)

            # Obtain the index and value of the most frequent label
            most_freq_label = percentage.argmax()
            label_percent = percentage[most_freq_label]

            # Compute name of the state
            if label_percent < 80:
                self.state_names.append("uncertain")
            else:
                self.state_names.append(self.cls_names[most_freq_label])

            # Create corresponding state model
            self.state_model_map[center_id] = StaticModel(most_freq_label)

            # Print out stats
            print("State name:           ", self.state_names[center_id])
            print("State model accuracy: ", self.state_model_map[center_id].evaluate(x_test, y_test))
            print("State size:           ", labels.shape[0])
            print("")


    def predict(self, state, x_data):
        # Predict the label from the given data point in a given state
        return self.state_model_map[state].predict(x_data)
