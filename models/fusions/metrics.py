import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score

def evaluate_new(df):
    auprc = average_precision_score(df['y_truth'], df['y_pred'])
    auroc = roc_auc_score(df['y_truth'], df['y_pred'])
    accuracy = accuracy_score(df['y_truth'], (df['y_pred']>0.5).astype(int))
    return auprc, auroc, accuracy

def bootstraping_eval(df, num_iter):
    auprc_list = []
    auroc_list = []
    accuracy_list = []
    for _ in range(num_iter):
        sample = df.sample(frac=1, replace=True)
        auprc, auroc, accuracy = evaluate_new(sample)
        auprc_list.append(auprc)
        auroc_list.append(auroc)
        accuracy_list.append(accuracy)
    return auprc_list, auroc_list, accuracy_list

def computing_confidence_intervals(list_, true_value):
    delta = (true_value - list_)
    delta = list(np.sort(delta))
    delta_lower = np.percentile(delta, 97.5)
    delta_upper = np.percentile(delta, 2.5)

    upper = true_value - delta_upper
    lower = true_value - delta_lower
    return upper, lower

def get_model_performance(df):
    test_auprc, test_auroc, test_accuracy = evaluate_new(df)
    auprc_list, auroc_list, accuracy_list = bootstraping_eval(df, num_iter=1000)
    upper_auprc, lower_auprc = computing_confidence_intervals(auprc_list, test_auprc)
    upper_auroc, lower_auroc = computing_confidence_intervals(auroc_list, test_auroc)
    upper_accuracy, lower_accuracy = computing_confidence_intervals(accuracy_list, test_accuracy)
    return (test_auprc, upper_auprc, lower_auprc), (test_auroc, upper_auroc, lower_auroc), (test_accuracy, upper_accuracy, lower_accuracy)