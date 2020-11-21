from pandas import read_csv
import numpy as np
import tensorflow as tf
from sklearn import preprocessing
import argparse
import os

from ROP.model_builder import build_LSTM_model
from model_extraction.model_extraction import ExtractedModel
from model_extraction.utils import evaluate_extracted_model, save_model_results, compute_average_metrics, \
    pred_label_thresholded, check_dir_exists, compute_window_effect, save_window_effect_results


'''
See "https://machinelearningmastery.com/how-to-predict-room-occupancy-based-on-environmental-factors/"
for further details, including possible references to related work
'''


def split_into_timeseries(x_data, y_data, time_window=60):
    '''
    Split the input sequence into multiple subsequences
    :param x_data: input sequence of shape (n_timesteps, n_features)
    :param y_data: array of (n_timesteps, ) of corresponding class labels, 1 per timestep
    :param time_window: length of subsequences
    :return: data array (n_sequences, time_window, n_features), and label array (n_sequences, time_window)
    '''

    n_samples = x_data.shape[0]
    n_seqs = n_samples // time_window       # Note: only full subsequences are retrieved

    sequences = []
    labels = []

    for i in range(n_seqs):
        start, stop = i * time_window, (i+1) * time_window
        sequence = np.expand_dims(x_data[start:stop], axis = 0)

        label = y_data[start:stop]
        label = np.expand_dims(label, axis=0)

        sequences.append(sequence)
        labels.append(label)

    sequences = np.vstack(sequences)
    labels = np.vstack(labels)

    return sequences, labels


def data_loader_roomocc(dataset_path, scale=False, normalise=False):
    '''
    Loads ROP task data
    :param dataset_path: path to downloaded ROP dataset dir
    :param scale: whether to scale data or not
    :param normalise: whether to normalise the data or not
    :return: returns training and testing data and labels
    '''

    data1_file = os.path.join(dataset_path, "datatest.txt")
    data2_file = os.path.join(dataset_path, "datatraining.txt")
    data3_file = os.path.join(dataset_path, "datatest2.txt")

    # Load all data
    test = read_csv(data1_file, header=0, index_col=0, parse_dates=True, squeeze=True)
    train = read_csv(data2_file, header=0, index_col=0, parse_dates=True, squeeze=True)
    val = read_csv(data3_file, header=0, index_col=0, parse_dates=True, squeeze=True)

    test_vals, train_vals, val_vals = test.values, train.values, val.values

    # Extract subset of features, as specified in paper
    x_train, y_train, x_test, y_test, x_val, y_val = train_vals[:, [1, 2, 3, 4]], train_vals[:, 6], \
                                                     test_vals[:, [1, 2, 3, 4]], test_vals[:, 6], \
                                                     val_vals[:, [1, 2, 3, 4]], val_vals[:, 6]

    # Combine training and validation data
    x_train = np.concatenate([x_train, x_val])
    y_train = np.concatenate([y_train, y_val])

    # Scale/Normalise the data
    if scale:
        x_train, x_test = preprocessing.scale(x_train), preprocessing.scale(x_test)
    if normalise:
        x_train, x_test = preprocessing.normalize(x_train), preprocessing.normalize(x_test)


    # Convert data into 3-dimensional arrays of shape (n_samples, n_timesteps, n_features)
    x_train, y_train = split_into_timeseries(x_train, y_train)
    x_test, y_test = split_into_timeseries(x_test, y_test)

    # Convert to appropriate numpy datatype
    x_train = np.asarray(x_train).astype(np.float32)
    x_test = np.asarray(x_test).astype(np.float32)
    y_train = np.expand_dims(y_train, axis=-1).astype(np.float32)
    y_test = np.expand_dims(y_test, axis=-1).astype(np.float32)

    return x_train, y_train, x_test, y_test


def get_model(dataset_path, model_save_path):
    '''
    :param dataset_path: Path to ROP dataset
    :param model_save_path: Path to saved model
    :return: loaded/trained model
    '''

    x_train, y_train, x_test, y_test = data_loader_roomocc(dataset_path)

    # build the network
    _, seq_len, n_features = x_train.shape
    model = build_LSTM_model(n_features, seq_len)

    if os.path.exists(model_save_path):
        model.load_weights(model_save_path)

    else:
        # Specify checkpoint saving
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=model_save_path, verbose=True)

        # fit the network
        model.fit(x_train, y_train, epochs=20, batch_size=20, validation_split=0.2, verbose=1, callbacks = [cp_callback])

        model.save_weights(model_save_path)

    # training metrics
    scores = model.evaluate(x_train, y_train, verbose=1, batch_size=200)
    print('AUC-ROC, and accuracy: {}'.format(scores[1:]))

    return model


def model_extraction(**params):

    check_dir_exists(params["exp_path"])

    # Load data
    dataset_path = params["dataset_path"]
    x_train, _, x_test, y_test = data_loader_roomocc(dataset_path)
    y_test = np.squeeze(y_test)

    # load model
    model_path = params["model_save_path"]
    rnn_model = get_model(dataset_path, model_path)

    # Retrieve thresholded labels predicted by original model
    y_rnn_pred = pred_label_thresholded(rnn_model, x_test)

    # Compute extracted model
    extracted_model = ExtractedModel(rnn_model, x_train, **params)

    # Retrieve labels predicted by extracted model
    y_extr_pred = extracted_model.batch_predict(x_test, return_sequences=True)

    # Compute performance metrics
    metrics = evaluate_extracted_model(y_test, y_rnn_pred, y_extr_pred)

    # Save results
    feature_names = ["Temp.", "Hum.", "Light", "CO2"]
    save_model_results(extracted_model, params["t_model"], params["exp_path"], feature_names)

    return metrics


def main(args):

    # Retrieve experiment params
    params = {"window_size":        args.window_size,
              "max_trans_samples":  args.max_trans_samples,
              "max_state_samples":  args.max_state_samples,
              "layer_id":           args.layer_id,
              "n_states":           args.n_states,
              "t_model":            args.t_model,
              "rep_converter":      None,
              "cls_names":          ["empty", "occupied"],
              "exp_path":           args.exp_path,
              "model_save_path":    args.model_save_path,
              "dataset_path":       args.dataset_path}


    # Run model extraction experiments
    n_experiments = 10
    compute_average_metrics(n_experiments, model_extraction, **params)

    # Run window effect experiments
    results = compute_window_effect(n_experiments, model_extraction, **params)
    fname = os.path.join(params["exp_path"], "window_results.txt")
    save_window_effect_results(results, fname)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('window_size',          type=int, default=0,    help='Context window size to use during model extraction')
    parser.add_argument('max_trans_samples',    type=int, default=None, help='Maximum number of samples to use for transition model training')
    parser.add_argument('max_state_samples',    type=int, default=None, help='Maximum number of samples to use for concept (aka state) extraction')
    parser.add_argument('layer_id',             type=int, default=-2,   help='ID of the recurrent layer to use for the latent space')
    parser.add_argument('n_states',             type=int, default=2,    help='Number of concepts (aka states) to extract')
    parser.add_argument('t_model',              type=str, choices=['dt', 'mlp'], default='dt', help='Type of transition model to use')
    parser.add_argument('exp_path',             type=str, default = "./", help='Path for saving experiments')
    parser.add_argument('dataset_path',         type=str, default = "./", help='Path to the dataset')
    parser.add_argument('model_save_path',      type=str, default = "./", help='Path for saving/loading the RNN model')

    args = parser.parse_args()

    main(args)


