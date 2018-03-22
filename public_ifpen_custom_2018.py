import numpy as np
import sys
import csv

def compute_q2(y_true, y_pred):

    if y_true.shape[0] != y_pred.shape[0]:
        sys.exit("Error : number of elements in prediction not equal to number of elements in test")
    elif y_true.shape[1] != y_pred.shape[1]:
        sys.exit("Error : number of features in prediction not equal to number of features in test")

    y_mean = compute_ymean(y_true)

    score_data = compute_score(y_true, y_pred, 10, 0.01)

    score_mean = compute_score(y_true, y_mean, 10, 0.01)

    return 1 - sum(score_data)/sum(score_mean)


def compute_ymean(y_true):
    row_y_mean = np.mean(y_true, 0)
    return np.tile(row_y_mean, (np.size(y_true, 0), 1))


def compute_score(y_true, y_pred, scaling_sor, scaling_bins):
    score_material = (y_true[:, 0] - y_pred[:, 0])**2

    sor_mean_true = np.mean(y_true[:, 1])

    score_sor = ((y_true[:, 1] - y_pred[:, 1])/sor_mean_true)**2
    row_bin_mean_true = np.mean(y_true[:, 2:14], 1)
    row_bin_mean_pred = np.mean(y_pred[:, 2:14], 1)
    bin_mean_true = np.mean(row_bin_mean_true)
    score_bins = ((row_bin_mean_true-row_bin_mean_pred)/bin_mean_true)**2

    return score_material + scaling_sor * score_sor + scaling_bins * score_bins

if __name__ == '__main__':

    y_true = np.empty([112, 14])

    row_number = 0
    with open(sys.argv[1], 'r') as csvfile:
        true_file = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in true_file:
            for key in row:
                key2 = key.split(';')
                if row_number > 0:
                    line_array = np.array(key2)
                    if line_array[1] == 'Benth':
                        y_true[row_number - 1, 0] = 0
                    elif line_array[1] == 'Clash':
                        y_true[row_number - 1, 0] = 1
                    else:
                        sys.exit("Wrong material type")
                    for j in range(1, 14):
                        y_true[row_number-1, j] = line_array[j+1]
            row_number += 1

    y_pred = np.empty([112, 14])

    row_number = 0
    with open(sys.argv[2], 'r') as csvfile:
        pred_file = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in pred_file:
            for key in row:
                key2 = key.split(';')
                if row_number > 0:
                    line_array = np.array(key2)
                    if line_array[1] == 'Benth':
                        y_pred[row_number - 1, 0] = 0
                    elif line_array[1] == 'Clash':
                        y_pred[row_number - 1, 0] = 1
                    else:
                        sys.exit("Wrong material type")
                    for j in range(1, 14):
                        y_pred[row_number-1, j] = line_array[j+1]
            row_number += 1
    score = compute_q2(y_true, y_pred)
    print(score)
