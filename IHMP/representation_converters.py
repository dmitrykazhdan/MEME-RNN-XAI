import numpy as np

'''
Class for basic representation pre-processing of the MIMIC-III data, 
as described in 'MEME: Generating RNN Model Explanations via Model Extraction'
'''

class MimicRepConverter():

    def _one_hot_encoded_to_cat(self, x, feature_indices):
        '''
        Retrieves the categorical representation of a one-hot encoded feature
        :param x: (n_samples, n_timesteps, n_features)
        :param feature_indices: list of all column indices of the one-hot encoded feature
        :return: array (n_samples, n_timesteps, 1),
                where the one-hot encoded feature has been converted to a categorical representation
        '''

        feature_vals = x[:, :, feature_indices]
        feature_vals = feature_vals.astype(int)
        feature_vals = np.argmax(feature_vals, axis = -1)
        feature_vals = np.expand_dims(feature_vals, axis = -1)
        return feature_vals


    def _use_mimic3_deencoding(self, x):
        '''
        Converts all one-hot encoded features to their categorical representation
        :param x: samples of shape (n_samples, n_timesteps, n_features)
        :return: array of shape (n_samples, n_timesteps, n_features*), with one-hot encoded
                 features converted to categorical
        '''

        # Safe copy of data
        x_transformed = x[:]

        # Retrieve categorical variants of all one-hot encoded features
        capillary_vals          = self._one_hot_encoded_to_cat(x_transformed, [i for i in range(0,  0+2)])
        glascow_eye_opening     = self._one_hot_encoded_to_cat(x_transformed, [i for i in range(4,  4+8)])
        glascow_motor_response  = self._one_hot_encoded_to_cat(x_transformed, [i for i in range(12, 12+12)])
        glascow_total           = self._one_hot_encoded_to_cat(x_transformed, [i for i in range(24, 24+13)])
        glascow_verbal_response = self._one_hot_encoded_to_cat(x_transformed, [i for i in range(37, 37+12)])

        # Retrieve continuous features
        continuous_features     = x_transformed[:, :, [2, 3] + [i for i in range(49, 59)]]

        # Concatenate all the features together
        all_features = [capillary_vals, glascow_eye_opening, glascow_motor_response,
                        glascow_total, glascow_verbal_response, continuous_features]

        x_transformed = np.concatenate(all_features, axis=-1)

        return x_transformed


    def convert_representation(self, x):
        '''
        Convert the input representation to the categorical one
        :param x: input data in original representation
        :return: new data in transformed representation
        '''

        x_transformed = self._use_mimic3_deencoding(x)
        return self._use_mimic3_deencoding(x_transformed)


    def get_feature_names(self):
        '''
        Retrieve the names of the features of the new representation
        '''

        feature_names = ["CRR", "G_Eye", "G_Motor", "G_Total", "G_Verbal"] + \
                        ["DBP", "FI_o2", "Glucose", "HR", "Height", "MBP", "o2S", "RR", "SBP", "Temp.", "Weight",
                         "pH"]

        return feature_names



    def get_categorical_features(self):
        '''
        Retrieve the indices of the categorical features of the new representation
        '''

        cat_features = [0, 1, 2, 3, 4]
        return cat_features