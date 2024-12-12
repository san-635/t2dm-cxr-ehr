from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import platform
import pickle
import json
import os

class Discretizer:
    def __init__(self, timestep=0.25, impute_strategy='zero', store_masks=True, start_time='zero', config_path='vilt/ehr_ecg_utils/discretizer_config.json'):
        # READ JSON CONFIG. FILE #
        with open(config_path) as f:
            config_json = json.load(f)
            self._ts_var = config_json['ts_variables']
            self._non_ts_var = config_json['non_ts_variables']
            self._normal_values = config_json['normal_values']
            self._ts_var_to_id = dict(zip(self._ts_var, range(len(self._ts_var))))
            # i.e., self._ts_var_to_id = {'Diastolic blood pressure': 0, 'Heart Rate': 1, ...}
        
        self._non_ts_var_to_id = {}
        id = len(self._ts_var)
        for non_ts_var in self._non_ts_var:
            self._non_ts_var_to_id[non_ts_var] = id
            id += 1
        # i.e., self._non_ts_var_to_id = {'gender': 8, 'age': 9, 'family_history': 10}

        # INITIALISATIONS #
        self._header = ["Hours"] + self._ts_var
        self._timestep = timestep
        self._store_masks = store_masks
        self._impute_strategy = impute_strategy
        self._start_time = start_time
        
        # STATS #
        self._done_count = 0
        self._empty_bins_sum = 0
        self._unused_data_sum = 0

    def transform(self, ts_rows, non_ts_data, max_rows=None):
        header = self._header
        assert header[0] == "Hours"
        eps = 1e-6

        # NUMBER OF EHR FEATURES (COLS) #
        N_var = len(self._ts_var) + len(self._non_ts_var)

        # ENSURE THAT ROWS ARE CHRONOLOGICALLY SORTED #
        ts = [float(row[0]) for row in ts_rows]
        for i in range(len(ts) - 1):
            assert ts[i] < ts[i+1] + eps

        # NUMBER OF BINS (ROWS) #
        if self._start_time == 'relative':
            first_time = ts[0]
        elif self._start_time == 'zero':
            first_time = 0
        else:
            raise ValueError('Invalid start_time')
        max_hours = max(ts) - first_time

        N_bins = int(max_hours / self._timestep + 1.0 - eps)    # + 1 to ensure that the last bin is not missed by int()
        if max_rows is not None:
            # if the number of bins is greater than max_text_len-1, it is capped and max_hours is recalculated
            if N_bins > max_rows:
                N_bins = max_rows
                max_hours = N_bins * self._timestep - eps
            # else, N_bins is unchanged (can be smaller than max_text_len-1; padded later with 0s)

        # DISCRETIZED DATA AND CORRESPONDING MASK #
        data = np.zeros(shape=(N_bins, N_var), dtype=float)
        mask = np.zeros(shape=(N_bins, N_var), dtype=int)
        original_value = [["" for j in range(N_var)] for i in range(N_bins)]      # for imputation

        # STATS #
        total_data = 0
        unused_data = 0

        # DISCRETIZATION OF TIMESERIES FEATURE COLS ACROSS ALL BINS #
        for row in ts_rows:
            # DETERMINE CORRESPONDING BIN FOR THIS ROW #
            t = float(row[0]) - first_time
            if t > max_hours + eps:
                continue                                # ignore if corresponding bin is larger than max_text_len-1
            bin_id = int(t / self._timestep - eps)
            if bin_id < 0 or bin_id >= N_bins:
                continue                                # ignore if corresponding bin is outside the range of bins

            # WRITE/OVERWRITE VALUES TO ALL TIMESERIES FEATURE COLS FOR THIS BIN #
            for j in range(1, len(row)):
                if row[j] == "":
                    continue                            # ignore if col is empty at this time
                if j < len(header):
                    variable = header[j]
                    value = row[j]
                else:
                    continue                            # ignore if col is static feature
                variable_id = self._ts_var_to_id[variable]

                total_data += 1
                if mask[bin_id][variable_id] == 1:
                    unused_data += 1                    # if bin for this col already has a value, update unused_data since current value will now be overwritten

                data[bin_id, variable_id] = float(value)
                mask[bin_id][variable_id] = 1

                original_value[bin_id][variable_id] = value

        # IMPUTATION OF EMPTY TIMESERIES FEATURE COLS ACROSS ALL BINS #
        if self._impute_strategy not in ['zero', 'normal_value', 'mean', 'previous', 'next']:
            raise ValueError("Invalid impute_strategy")

        if self._impute_strategy in ['normal_value', 'previous']:
            prev_values = [[] for i in range(len(self._ts_var))]
            for bin_id in range(N_bins):
                for variable in self._ts_var:
                    variable_id = self._ts_var_to_id[variable]
                    if mask[bin_id][variable_id] == 1:
                        prev_values[variable_id].append(original_value[bin_id][variable_id])
                        continue

                    if self._impute_strategy == 'normal_value':
                        imputed_value = self._normal_values[variable]       # impute with normal value
                    if self._impute_strategy == 'previous':
                        if len(prev_values[variable_id]) == 0:
                            imputed_value = self._normal_values[variable]   # impute with normal value if no previous value
                        else:
                            imputed_value = prev_values[variable_id][-1]    # impute with immediately previous value
                    data[bin_id, variable_id] = float(imputed_value)

        if self._impute_strategy == 'next':
            prev_values = [[] for i in range(len(self._ts_var))]
            for bin_id in range(N_bins-1, -1, -1):   # iterate over all bins in reverse order
                for variable in self._ts_var:
                    variable_id = self._ts_var_to_id[variable]
                    if mask[bin_id][variable_id] == 1:
                        prev_values[variable_id].append(original_value[bin_id][variable_id])
                        continue

                    if len(prev_values[variable_id]) == 0:
                        imputed_value = self._normal_values[variable]       # impute with normal value if no next value
                    else:
                        imputed_value = prev_values[variable_id][-1]        # impute with immediately next value
                    data[bin_id, variable_id] = float(imputed_value)

        if self._impute_strategy == 'mean':
            load_file_path = os.path.join(os.path.dirname(__file__), 'ehr_normalizer__{}_{}h_zero'.format(max_rows, self._timestep))
            with open(load_file_path, "rb") as load_file:
                if platform.python_version()[0] == '2':
                    dict = pickle.load(load_file)
                else:
                    dict = pickle.load(load_file, encoding='latin1')
                col_means = dict['means']
            for bin_id in range(N_bins):
                for variable in self._ts_var:
                    variable_id = self._ts_var_to_id[variable]
                    if mask[bin_id][variable_id] == 1:
                        continue
                    data[bin_id, variable_id] = col_means[variable_id]      # impute with mean value

        # STATS - CAN BE PRINTED #
        empty_bins = np.sum([1 - min(1, np.sum(mask[i, :])) for i in range(N_bins)])
        self._done_count += 1
        self._empty_bins_sum += empty_bins / (N_bins + eps)
        self._unused_data_sum += unused_data / (total_data + eps)

        # CONCATENATE STATIC FEATURE COLS #
        for non_ts, value in non_ts_data.items():
            non_ts_id = self._non_ts_var_to_id[non_ts]
            data[:, non_ts_id] = value
            mask[:, non_ts_id] = 1

        # indicates if model should ignore a certain bin since it has no useful data apart from the static features
        mask_1d = np.any(mask[:, :len(self._ts_var)], axis=1).astype(int)

        if self._store_masks:
            return (data, mask_1d)          # return discretized ts and static data and its mask
        else:
            return data                     # return discretized ts and static data

class Normalizer:
    def __init__(self):
        self._means = None
        self._stds = None
        
        self._sum_col = None
        self._sum_sq_col = None
        self._count = 0

    # calculate the running totals of count, sum and sum of squares of the cols over all samples in training set
    def _feed_data(self, data):
        data = np.array(data)
        self._count += data.shape[0]                     # running count of num. of rows in each sample's data
        if self._sum_col is None:
            self._sum_col = np.sum(data, axis=0)
            self._sum_sq_col = np.sum(data**2, axis=0)
        else:
            self._sum_col += np.sum(data, axis=0)        # running sum of the cols of sample's data
            self._sum_sq_col += np.sum(data**2, axis=0)  # running sum of squares of the cols of sample's data

    # calculate the means and std devs of cols considering all samples in training set and save them to a .pkl file
    def _save_params(self, save_file_path):
        eps = 1e-7
        with open(save_file_path, "wb") as save_file:
            N = self._count
            self._means = self._sum_col / N
            self._stds = np.sqrt(1.0/(N - 1) * (self._sum_sq_col - 2.0 * self._sum_col * self._means + N * self._means**2))
            self._stds[self._stds < eps] = eps           # to avoid division by 0
            pickle.dump(obj = {'means': self._means, 'stds': self._stds}, file=save_file, protocol=2)

    # load the means and std devs from pickle file for normalisation of any dataset's samples
    def load_params(self, load_file_path):
        with open(load_file_path, "rb") as load_file:
            if platform.python_version()[0] == '2':
                dict = pickle.load(load_file)
            else:
                dict = pickle.load(load_file, encoding='latin1')
            self._means = dict['means']
            self._stds = dict['stds']

    # normalise a sample's 2D array data column-wise using the loaded means and std devs
    def transform(self, data):
        ret = 1.0 * data
        for col in range(data.shape[1]):
            ret[:, col] = (data[:, col] - self._means[col]) / self._stds[col]
        return ret