import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


#------------------------------------------------------General Utils----------------------------------------------------


def evaluate_extracted_model(y_true, y_rnn_pred, y_extr_pred, verbose=True):
    '''
    :param y_true:      Ground truth values
    :param y_rnn_pred:  RNN predictions
    :param y_extr_pred: Extracted model predictions
    :param verbose:     Whether to print the values to console or not
    :return:            Returns a dictionary of metrics, including f1, prediction, recall, accuracy, and fidelity scores
    '''

    # If there is a time dimension, need to flatten
    y_true = y_true.flatten()
    y_rnn_pred = y_rnn_pred.flatten()
    y_extr_pred = y_extr_pred.flatten()

    metrics = {}

    metrics["fidelity"] = (y_rnn_pred == y_extr_pred).sum() / y_rnn_pred.shape[0] * 100
    metrics["accuracy"] = (y_true == y_extr_pred).sum() / y_true.shape[0] * 100
    metrics["orig_acc"] = (y_true == y_rnn_pred).sum() / y_true.shape[0] * 100
    metrics["f1"]       = f1_score(y_rnn_pred, y_extr_pred)
    metrics["pr"]       = precision_score(y_rnn_pred, y_extr_pred)
    metrics["rec"]      = recall_score(y_rnn_pred, y_extr_pred)

    if verbose:
        print("True labels:                  ", y_true)
        print("Predicted by original model:  ", y_rnn_pred)
        print("Predicted by extracted model: ", y_extr_pred)
        print("Extracted model fidelity:     ", metrics["fidelity"])
        print("Extracted model accuracy:     ", metrics["accuracy"])
        print("Original model accuracy:      ", metrics["orig_acc"])
        print("Pr:  ", metrics["pr"])
        print("Rec: ", metrics["rec"])
        print("F1:  ", metrics["f1"])

    return metrics



def pred_label_thresholded(rnn_model, x_data, threshold=0.5):
    '''
    Predict sigmoid values of x_data from rnn_model, and threshold them based on the threshold value
    '''

    y_rnn_pred = np.squeeze(rnn_model(x_data).numpy())
    y_rnn_pred = (y_rnn_pred>threshold).astype(int)
    return y_rnn_pred


def try_to_retrieve(key, **dict):
    '''
    Safe dictionary retrieval
    '''

    if key not in dict:
        raise ValueError("Expecting key: ", str(key))
    else:
        return dict[key]


#------------------------------------------------------Results Saving---------------------------------------------------


def compute_average_metrics(n_experiments, exp_runner, **params):
    '''
    Returns averaged metrics as a dictionary, with every metric name
    mapping to a list consisting of [avg, std] values for that metric

    :param n_experiments: Number of re-runs
    :param exp_runner:    Experiment runner file
    :param params:        Experiment parameters dictionary
    :return:              Dictionary (metric_key --> [avg_metric_val, std_metric_val]), averaged over n_experiment re-runs
    '''

    # Compute all metrics for all experiments
    all_metrics = {}
    for i in range(n_experiments):
        metrics = exp_runner(**params)

        for key in metrics.keys():
            if key not in all_metrics: all_metrics[key] = []
            all_metrics[key].append(metrics[key])
    print("")

    # Aggregate metrics per experiment
    agg_metrics = {}
    for key in all_metrics.keys():
        avg = np.round(np.mean(all_metrics[key]), decimals=2)
        std = np.round(np.std(all_metrics[key]),  decimals=2)
        agg_metrics[key] = [avg, std]
        print(key + " " + str(avg) + " +/- " + str(std))

    return agg_metrics


def compute_window_effect(n_experiments, exp_runner, window_max=5, **params):
    '''
    Experiment with varying window effect
    :param n_experiments: Number of experiment re-runs per window value
    :param exp_runner:    Experiment runner file
    :param window_max:    Maximum value of window size to use
    :param params:        Experiment parameters
    :return:
    '''

    # Compute average metrics for each value of the window size parameter
    results = []
    for w in range(0, window_max):
        params["window_size"] = w
        agg_metrics = compute_average_metrics(n_experiments, exp_runner, **params)
        agg_metrics['w'] = [w]
        results.append(agg_metrics)
    print(results)

    return results

def save_window_effect_results(agg_metrics, fname):
    '''
    :param agg_metrics: A list of dictionaries, with every dictionary consisting of
                        key:val, and val is a space-separated list of values
    :return:
    '''

    with open(fname, 'w') as f:
        for next_metrics in agg_metrics:
            line = ""
            for key in next_metrics.keys():
                vals = next_metrics[key]
                line += str(key) + ":" + " ".join([str(i) for i in vals]) + ","

            line = line[:-1]                # Omit the last comma
            line += "\n"

            f.write(line)


def check_dir_exists(exp_path):
    # Ensure experiment dir exists
    import os
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)


def save_state_info(exp_path, extracted_model):
    '''
    Save the cluster names of the extracted model, as well as the cluster purities
    '''
    statefile_fname = exp_path + "state_information.txt"

    with open(statefile_fname, 'w') as f:
        state_inf = str(extracted_model.cluster_cls_model.cls_dist)
        f.write(state_inf)
        
        state_names = str(extracted_model.get_state_names())
        f.write(state_names)


def save_feature_importance_data(extracted_model, t_model, exp_path, feature_names):

    fnb_name = exp_path + "feature_importance.txt"

    with open(fnb_name, 'w') as f:
        for i in range(extracted_model.n_states):

            if t_model == "dnn":
                fnbs = extracted_model.transition_models[i].f_importances["importances_mean"]
            elif t_model == "dt":
                fnbs = extracted_model.transition_models[i].clf.feature_importances_
            else:
                raise ValueError("Unrecognised model type")

            fnbs = np.abs(fnbs)
            fnb_inds = list(np.argsort(-fnbs))

            for fnb_ind in fnb_inds:
                line = feature_names[fnb_ind] + ":" + str(fnbs[fnb_ind]) + ","
                f.write(line)
            f.write("\n")


def save_model_results(extracted_model, t_model, exp_path, feature_names):

    # Save information on extracted model clusters
    cls_names = extracted_model.get_state_names()
    save_state_info(exp_path, extracted_model)

    # Save information on feature importances
    save_feature_importance_data(extracted_model, t_model, exp_path, feature_names)

    # For decision trees, also save their plots
    if t_model == "dt":
        from sklearn.tree import plot_tree
        import matplotlib.pyplot as plt

        for i in range(extracted_model.n_states):
            dt = extracted_model.transition_models[i].clf
            dt_name = exp_path + "DT_" + str(cls_names[i]) + ".png"

            fig, ax = plt.subplots(figsize=(15, 15))  # whatever size you want
            plot_tree(dt,
                      ax=ax,
                      feature_names=feature_names,
                      filled=True,
                      rounded=True,
                      proportion=True,
                      precision=2,
                      class_names=extracted_model.transition_models[i].cls_names,
                      impurity=False)
            plt.savefig(dt_name)





#------------------------------------------------------Data Loading-----------------------------------------------------

def get_balanced_dataset(x, y):
    '''
    Balances the samples in x, such that the number of samples of cls 0 is equal
    to the number of samples of cls 1
    :param x: data samples
    :param y: data labels
    :return: balanced samples and labels
    '''

    # Indices of classes
    cls_0 = np.where(y == 0)[0]
    cls_1 = np.where(y == 1)[0]

    # Number of each class
    n_zeros = cls_0.shape[0]
    n_ones = cls_1.shape[0]

    if n_zeros > n_ones:
        bigger, smaller = cls_0, cls_1
    else:
        bigger, smaller = cls_1, cls_0

    # Need to extract subset of bigger class
    n_samples = smaller.shape[0]
    inds = np.arange(bigger.shape[0])
    np.random.shuffle(inds)
    inds = inds[:n_samples]
    bigger = bigger[inds]

    x0, y0 = x[bigger], y[bigger]
    x1, y1 = x[smaller], y[smaller]

    x_balanced, y_balanced = np.vstack([x0, x1]), np.concatenate([y0, y1])

    # Shuffle the dataset before returning
    inds = np.arange(x_balanced.shape[0])
    np.random.shuffle(inds)
    x_balanced, y_balanced = x_balanced[inds], y_balanced[inds]

    return x_balanced, y_balanced



def get_subset(x_data, y_data, ratio):
    '''
    Get random subset of a dataset
    '''

    n_samples = int(x_data.shape[0] * ratio)
    inds = np.arange(x_data.shape[0])
    np.random.shuffle(inds)
    inds = inds[:n_samples]
    x_data, y_data = x_data[inds], y_data[inds]

    return x_data, y_data



def get_confident_points(model, x_data, y_data, threshold=0.4999):
    '''
    Retrieve points within a certain threshold of sigmoid confidence
    Note: only works for binary classification tasks
    '''

    # Retrieve sigmoid outputs of the model
    confs = np.squeeze(model(x_data).numpy())

    # Compute indices of outputs such that (output < threshold) OR (output > 1 - threshold)
    confident_inds = np.where(np.logical_or((confs < threshold),  (confs > (1.0 - threshold))))[0]

    x_conf, y_conf = x_data[confident_inds], y_data[confident_inds]

    return x_conf, y_conf


