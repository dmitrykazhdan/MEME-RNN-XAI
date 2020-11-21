import numpy as np
import tensorflow as tf
import os
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score
import argparse

from IHMP.model_builder import build_LSTM_model
from model_extraction.model_extraction import ExtractedModel
from IHMP.representation_converters import MimicRepConverter
from model_extraction.utils import save_model_results, evaluate_extracted_model, \
    pred_label_thresholded, get_confident_points, get_balanced_dataset, compute_average_metrics, compute_window_effect, \
    save_window_effect_results, check_dir_exists

'''

For a description of how the values were computed:

1) Original paper: https://arxiv.org/pdf/1703.07771.pdf
2) Listing of possible values: https://github.com/YerevaNN/mimic3-benchmarks/blob/master/mimic3models/resources/channel_info.json

Overall, there is a binary vector indicating whether the variable was observed, or not, followed by the value of the variable

Total number of features:

(variable observed or not) + (continuous variables) + (one-hot encoded categorical variables) =
          17               +          12            + ( 2 + 8 + 12 + 13 + 12 )                =  76
          
Features are:

[0:1]   : Capillary refill rate
2       : Diastolic blood pressure 
3       : Fraction inspired oxygen
[4:11]  : Glascow coma scale eye opening
[12:23] : Glascow coma scale motor response
[24:36] : Glascow coma scale total
[37:48] : Glascow coma scale verbal response 
49      : Glucose
50      : Heart Rate
51      : Height
52      : Mean blood pressure
53      : Oxygen saturation
54      : Respiratory rate
55      : Systolic blood pressure
56      : Temperature
57      : Weight
58      : pH
[59:75] : binary, whether feature was observed or not          

'''


def data_loader_ihm(data_path):

    # Specify paths to data files
    x_train_name, y_train_name  = os.path.join(data_path, "X_train.npy"),   os.path.join(data_path, "y_train.npy")
    x_val_name, y_val_name      = os.path.join(data_path, "X_val.npy"),     os.path.join(data_path, "y_val.npy")
    x_test_name, y_test_name    = os.path.join(data_path, "X_test.npy"),    os.path.join(data_path, "y_test.npy")

    # Load data from data files
    x_train, y_train = np.load(x_train_name), np.load(y_train_name)
    x_val, y_val = np.load(x_val_name), np.load(y_val_name)
    x_test, y_test = np.load(x_test_name), np.load(y_test_name)

    # Balance out the datasets via downsampling
    x_train, y_train    = get_balanced_dataset(x_train, y_train)
    x_val, y_val        = get_balanced_dataset(x_val, y_val)
    x_test, y_test      = get_balanced_dataset(x_test, y_test)

    return (x_train, y_train, x_val, y_val, x_test, y_test)



def get_model(dataset_path, model_save_path):

    x_train, y_train, x_val, y_val, x_test, y_test = data_loader_ihm(dataset_path)

    # build the network
    _, seq_len, n_features = x_train.shape
    model = build_LSTM_model(n_features, seq_len)

    if os.path.exists(model_save_path):
        model.load_weights(model_save_path)

    else:
        # Specify checkpoint saving
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=model_save_path, verbose=True)

        # fit the network
        model.fit(x_train, y_train, epochs=10, batch_size=100, validation_data=(x_val, y_val), verbose=1, callbacks = [cp_callback])

        model.save_weights(model_save_path)


    # training metrics
    scores = model.evaluate(x_train, y_train, verbose=1, batch_size=200)
    print('AUC-ROC, and accuracy: {}'.format(scores[1:]))

    pred = np.squeeze(model.predict_classes(x_test))

    f1 = f1_score(y_test, pred)
    pr = precision_score(y_test, pred)
    rec = recall_score(y_test, pred)

    ppred = np.squeeze(model.predict(x_test))
    roc_auc = roc_auc_score(y_test, ppred)
    acc = accuracy_score(y_test, pred)

    print("Pr:  ",      pr)
    print("Rec: ",      rec)
    print("F1:  ",      f1)
    print("Accuracy: ", acc)
    print("Roc-AUC: ",  roc_auc)

    return model



def mimic3(**params):

    check_dir_exists(params["exp_path"])
    dataset_path = params["dataset_path"]
    checkpoint_path = params["model_save_path"]

    # load trained model
    rnn_model = get_model(dataset_path, checkpoint_path)

    # Load original dataset
    x_train, y_train, x_val, y_val, x_test, y_test = data_loader_ihm(dataset_path)

    # Filter out high-confident points
    x_train, y_train    = get_confident_points(rnn_model, x_train, y_train)
    x_val, y_val        = get_confident_points(rnn_model, x_val, y_val)
    x_test, y_test      = get_confident_points(rnn_model, x_test, y_test)

    # Combine train and val sets
    x_train = np.vstack([x_train, x_val])
    y_train = np.concatenate([y_train, y_val])

    # Balance out the datasets
    x_train, y_train    = get_balanced_dataset(x_train, y_train)
    x_test, y_test      = get_balanced_dataset(x_test, y_test)

    # Retrieve labels predicted by original model
    y_rnn_pred = pred_label_thresholded(rnn_model, x_test)

    # Compute extracted model
    extracted_model = ExtractedModel(rnn_model, x_train, **params)

    # Retrieve labels predicted by extracted model
    y_extr_pred = extracted_model.batch_predict(x_test)

    # Compute performance metrics
    metrics = evaluate_extracted_model(y_test, y_rnn_pred, y_extr_pred)

    # Save results
    feature_names = params["rep_converter"].get_feature_names()
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
              "rep_converter":      MimicRepConverter(),
              "cls_names":          ["healthy", "sick"],
              "exp_path":           args.exp_path,
              "model_save_path":    args.model_save_path,
              "dataset_path":       args.dataset_path}

    # Run model extraction experiments
    n_experiments = 10
    compute_average_metrics(n_experiments, mimic3, **params)

    # Run window effect experiments
    results = compute_window_effect(n_experiments, mimic3, **params)
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