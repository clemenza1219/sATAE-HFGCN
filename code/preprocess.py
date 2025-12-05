import os
import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
from matplotlib.colors import ListedColormap
from statsmodels.stats.multitest import multipletests
import seaborn as sns
from scipy.signal import butter, lfilter, resample, filtfilt


#names = [' ']
#full_name = '  '
regions = ['OC','PHG','HIP', 'AMY']
base_dir = f'/home/phd-yan.huachao/Project/Xiehedata/CCEP_data/'

#LinLF_regions =  [(0, 14), (14, 28), (28, 40), (40, 52),
#                  (52, 66), (66, 78), (78, 90),
#                   (90, 106), (106, 122), (122, 130), (130, 138)]


def Bipolar_lead(raw, regions):
    diff_regions = []

    for start, end in regions:
        # print(start, end)
        region_data = raw[start:end, ]
        # print(region_data.shape)
        diff_region = region_data[:-1, :] - region_data[1:, :]  #nmeige

        print('*'*80)
        diff_regions.append(diff_region)
        # print(diff_array.shape)
    diff_array = np.concatenate(diff_regions, axis=0)
    # print(diff_array.shape)
    # bi_mne  = mne.io.RawArray(diff_array, info=raw_data.info)
    # print(bi_mne.info)
    return diff_array

def delete_rows(raw_data, rows_to_delete):

    modified_matrix = np.copy(raw_data)
    if rows_to_delete:
        modified_matrix = np.delete(modified_matrix, rows_to_delete, axis=0)
    return modified_matrix

def downsample_signal(filtered_signal, original_sampling_rate, target_sampling_rate):
    downsample_factor = int(original_sampling_rate / target_sampling_rate)
    downsampled_signal = resample(filtered_signal.T, len(filtered_signal.T) // downsample_factor)
    # print(len(filtered_signal))
    return downsampled_signal.T


def plot_fdr_corrected_p_value_matrix(matrix1, matrix2, alpha=0.05):
    # Ensure both matrices have the same shape
    assert matrix1.shape == matrix2.shape, "Matrices must have the same shape."

    # Initialize a matrix to store p-values
    p_value_matrix = np.zeros((matrix1.shape[0], matrix1.shape[0]))

    # Perform paired t-test for each pair of rows
    p_values = []
    for i in range(matrix1.shape[0]):
        for j in range(i, matrix2.shape[0]):
            _, p_value = ttest_rel(matrix1[i], matrix2[j])
            p_value_matrix[i, j] = p_value
            p_value_matrix[j, i] = p_value  # Ensure symmetry
            p_values.append(p_value)

    # Apply FDR correction
    _, corrected_p_values, _, _ = multipletests(p_values, alpha=alpha, method='fdr_bh')
    corrected_p_value_matrix = np.zeros_like(p_value_matrix)
    k = 0
    for i in range(matrix1.shape[0]):
        for j in range(i, matrix2.shape[0]):
            corrected_p_value_matrix[i, j] = corrected_p_values[k]
            corrected_p_value_matrix[j, i] = corrected_p_values[k]
            k += 1

    # Create a mask for corrected p-values greater than 0.05
    corrected_p_value_matrix[corrected_p_value_matrix > 0.05] = 1


    return corrected_p_value_matrix


def plot_correlation_matrix(matrix):
    # Compute the correlation matrix
    corr_matrix = np.corrcoef(matrix)
    corr_matrix[corr_matrix < 0.8] = 0
    return corr_matrix

down_rate = int(5000)
for name in names:
    for region in regions:
        save_all_baseline = []
        save_all_ccep_1s = []
        stim_dir = f'{base_dir}/CCEP_2regions/{name}/{region}'
        file_list = os.listdir(stim_dir)
        filtered_files = [file for file in file_list if 'stim' in file]
        for stim_file  in sorted(filtered_files):
            raw = mne.io.read_raw_edf(os.path.join(stim_dir, stim_file), preload=True)
            channels_to_drop = ['ECG+', 'ECG-']
            if all(channel in raw.ch_names for channel in channels_to_drop):
                raw.drop_channels(channels_to_drop)
            print(raw.info)
            print(raw.ch_names)
            raw_data, times = raw[:, :]
            print(raw_data.shape)
            bi_data = Bipolar_lead(raw_data, LinLF_regions)
            print(bi_data.shape)

            down_signal = downsample_signal(bi_data, original_sampling_rate=10000,
                                            target_sampling_rate=down_rate)

            rows_to_delete = None
            modified_data = delete_rows(down_signal, rows_to_delete)
            print(modified_data.shape)

            print('full data:',  modified_data.shape)
            baseline_data = modified_data[:, : down_rate]
            print('baseline:',  baseline_data.shape)
            ccep_data = modified_data[:, down_rate:]
            print('ccep:', ccep_data.shape)
            baseline_1s = baseline_data[:, :down_rate]
            ccep_1s = ccep_data[:, : down_rate]
            print(baseline_1s.shape)
            print(ccep_1s.shape)
            fdr_matrix = plot_fdr_corrected_p_value_matrix(baseline_1s, ccep_1s)
            # print(mask1)
            correlation_matrix1 = plot_correlation_matrix(ccep_data)
            correlation_matrix2 = plot_correlation_matrix(correlation_matrix1)
            save_all_ccep_1s.append(correlation_matrix2)

        array_save_all_ccep_1s = np.stack(save_all_ccep_1s, axis=0)
        # print(array_save_all_ccep_1s.shape)
        mean_ccep_1s = np.mean(array_save_all_ccep_1s, axis=0)
        print(f'adj shape', mean_ccep_1s.shape)

        np.save(f'./onset_wake_sleep/{full_name}/{full_name}_{region}_adj_0.3', mean_ccep_1s)
        print(f'adj save to:./onset_wake_sleep/{full_name}/{full_name}_{region}_adj_0.3.npy')











