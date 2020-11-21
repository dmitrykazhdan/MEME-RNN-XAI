from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import numpy as np

'''
Note: some of the code below is suboptimal, and will (potentially) be re-written in the (potentially-near) future
'''

class DNNTransitionModel(object):

    def __init__(self, n_states=None, n_features=None, state_names=None, representation_converter=None, x_data=None, y_data=None):

        self.converter = representation_converter
        self.state_names = state_names

        if (x_data is not None) and (y_data is not None):
            y_data = to_categorical(y_data)
            self.n_states = y_data.shape[-1]
            x_converted = self._convert_representation(x_data)
            self.n_features = x_converted.shape[-1]

        elif (n_states is not None) and (n_features is not None):
            self.n_states = n_states
            self.n_features = n_features

        else:
            raise ValueError("Error. Expecting either (states, features), or (x_data, y_data) for transition model")

        model = self._build_dnn_model(self.n_features, self.n_states)
        self.clf = model


    def _build_dnn_model(self, n_features, n_classes):
        model = Sequential()

        model.add(Dense(200, activation='relu', input_shape=(n_features,)))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(n_classes, activation='softmax'))

        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        return model


    def _convert_representation(self, x):
        if self.converter is None:
            return x
        else:
            return self.converter.convert_representation(x)


    def fit(self, x, y):
        """Fit the model using x as training data and l as class labels

        Arguments
        ---------
        x : array of shape [n_samples, n_timepoints, n_features]
            Training data set for input into DNN classifer

        l : array of shape [n_samples]
            Training labels for input into DNN classifier
        """
        y_cat = to_categorical(y)
        x_transformed = self._convert_representation(x)
        self.clf.fit(x_transformed, y_cat, epochs=100, batch_size=2000, validation_split=0.2, verbose=0)

        # TODO: fix this temporary hack, by computing feature importances from the Keras model as well
        # Note: sklearn feature importance only works with their MLPClassifier
        # Temporary solution: train the same MLPClassifier, together with the DNN
        from sklearn.neural_network import MLPClassifier
        self.mlp = MLPClassifier(hidden_layer_sizes=(200, 50), solver='adam')
        self.mlp.fit(self._convert_representation(x), y)


    def create_explainer(self, x_data, y_data):

        import lime.lime_tabular

        feature_names = self.converter.get_feature_names()
        cat_features =  self.converter.get_categorical_features()

        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=x_data,
            mode="classification",
            training_labels=y_data,
            feature_names=feature_names,
            categorical_features=cat_features,
            discretize_continuous=True,
            discretizer='decile',
            sample_around_instance=False,
            class_names=self.state_names
            )


    def explain(self, x):
        '''
        Explain single datapoint
        Note: assumes point passed in with batch dimension
        :param x: input, of shape (1, n_timesteps, n_features)
        :return: Lime explainer explanation
        '''

        # Compute converted representation, and remove batch dimension
        x_t = self._convert_representation(x)
        x_t = np.squeeze(x_t)

        # Compute predicted label
        label = self.predict(x)

        # Define function for computing scores
        def get_proba(x):
            probs = self.clf(x).numpy()
            return probs

        # Create lime explainer
        exp = self.explainer.explain_instance(x_t,
                                         get_proba,
                                         num_features=5,
                                         labels=label,
                                         num_samples=1000)

        return exp


    def evaluate(self, x, y_true):

        y_pred = self.predict(x)
        acc = (y_pred == y_true).sum() / y_true.shape[0] * 100

        # Compute feature importances
        from sklearn.inspection import permutation_importance
        x_transfored = self._convert_representation(x)
        result = permutation_importance(self.mlp, x_transfored, y_true, n_repeats=20)
        self.f_importances = result

        return acc


    def predict(self, x):
        """Predict the class labels or probability estimates for
        the provided data

        Arguments
        ---------
          x : array of shape [n_samples, n_timepoints, n_features]
              Array containing the testing data set to be classified
        """
        x_transfored = self._convert_representation(x)
        labels = self.clf.predict_classes(x_transfored)
        return labels



class DTTransitionModel(object):

    def __init__(self, cls_names, representation_converter=None):
        self.clf = DecisionTreeClassifier(max_depth=3)
        self.converter = representation_converter
        self.cls_names = cls_names

    def _convert_representation(self, x):
        if self.converter is None:
            # Extract last timestep for predicting
            x_t = x[:, -1, :]
            return x_t
        else:
            return self.converter.convert_representation(x)


    def fit(self, x, l):
        """Fit the model using x as training data and l as class labels

        Arguments
        ---------
        x : array of shape [n_samples, n_timepoints, n_features]
            Training data set for input into DT classifer

        l : array of shape [n_samples]
            Training labels for input into DT classifier
        """
        x_transformed = self._convert_representation(x)
        self.clf.fit(x_transformed, l)


    def evaluate(self, x, y_true):
        y_pred = self.predict(x)
        acc = (y_pred == y_true).sum() / y_true.shape[0] * 100
        return acc


    def predict(self, x):
        """Predict the class labels or probability estimates for
        the provided data

        Arguments
        ---------
          x : array of shape [n_samples, n_timepoints, n_features]
              Array containing the testing data set to be classified
        """
        x_transfored = self._convert_representation(x)
        labels = self.clf.predict(x_transfored)
        return labels


class StaticModel(object):
    '''
    Simple model, predicting the same output for any input
    '''

    def __init__(self, label):
        self.label = label

    def fit(self, x, l):
        pass

    def predict(self, x):
        n_samples = x.shape[0]
        return np.ones((n_samples,)) * self.label

    def evaluate(self, x, y_true):
        y_pred = self.predict(x)
        acc = (y_pred == y_true).sum() / y_true.shape[0] * 100
        return acc


