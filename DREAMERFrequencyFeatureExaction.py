import scipy.io as scio
import numpy as np
import os

from tqdm import tqdm

from scipy.integrate import simps
from tqdm import tqdm
from scipy.signal import butter, sosfilt
from scipy import signal

data_path = 'G:/Dataset/DREAMER'

# 第5个维度 0-22 代表不同的受试者
# 第8个维度 2代表EEG 3代表ECG
# 第9个和第10个维度固定为0
# 第11维度 1代表baseline 2代表stimuli
# 第12维度 0-17 代表不同的视频片段
# 第13维度 固定为0

subject_index = 0  # 0-22
EEG_ECG_index = 2  # 2-EEG, 3-ECG
baseline_stimuli_index = 0  # 0-baseline, 1-stimuli
trial_index = 0  # 0-17

# a = scio.loadmat(data_path)['DREAMER'][0][0][0][0][subject_index][0][0][EEG_ECG_index][0][0][
# baseline_stimuli_index][trial_index][0]

data_dict = scio.loadmat(data_path)['DREAMER'][0][0][0][0]
window_size = 128  # 滑动窗的长度


def butter_bandpass(lowcut, highcut, fs, order=21):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    return sos


def butter_bandpass_filter(data, lowcut, highcut, fs, order=21):
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sosfilt(sos, data, axis=0)
    return y


for subject_index in range(23):
    print("受试者{}".format(subject_index + 1))

    de_feature = np.zeros((1, 5, 14, 1))
    psd_feature = np.zeros((1, 5, 14, 1))

    for trial_index in range(18):
        print(trial_index)
        baseline_stimuli_index = 0  # 基线数据读取
        # (7808,14)
        baseline = data_dict[subject_index][0][0][EEG_ECG_index][0][0][baseline_stimuli_index][trial_index][0]
        baseline = baseline.T  # (14,7808) (channel*length)

        baseline_segment_length = baseline.shape[1] // window_size  # 61
        segment_sum = np.zeros((baseline.shape[0], window_size))
        for i in range(baseline_segment_length):
            segment_sum += baseline[:, window_size * i: window_size * (i + 1)]
        baseline_signal = segment_sum / baseline_segment_length

        baseline_stimuli_index = 1  # 实验刺激数据读取
        # (data_point, 14)
        origin_data = data_dict[subject_index][0][0][EEG_ECG_index][0][0][baseline_stimuli_index][trial_index][0]
        origin_data = origin_data.T  # (14,data_point) (channel*length)
        data_segment_length = origin_data.shape[1] // window_size

        # 滤波
        band_set = [0.1, 4, 8, 14, 30, 50]
        # trial * segment * band * time * ch
        segment_band_time_ch = np.zeros((data_segment_length, 5, 128, 14))

        # psd特征
        psd_feature_trial = np.zeros((data_segment_length, 5, 14, 1))
        # de特征
        de_feature_trial = np.zeros((data_segment_length, 5, 14, 1))

        for train_index in range(segment_band_time_ch.shape[0]):  # data_segment_length
            sample_data = origin_data[:,
                          window_size * train_index: window_size * (train_index + 1)] - baseline_signal  # (14, 128)
            time_ch = sample_data.T
            for band in range(segment_band_time_ch.shape[1]):  # frequency band
                segment_band_time_ch[train_index, band, :, :] = butter_bandpass_filter(time_ch,
                                                                                       lowcut=band_set[
                                                                                           band],
                                                                                       highcut=
                                                                                       band_set[
                                                                                           band + 1],
                                                                                       fs=128,
                                                                                       order=21)

                # 求psd特征
                freqs, psd = signal.periodogram(segment_band_time_ch[train_index, band, :, :].T,
                                                fs=128,
                                                window='hann', nfft=512,
                                                scaling='density',
                                                detrend='constant')
                # 求积分，求psd_all，顺便求de
                for channel in range(segment_band_time_ch.shape[3]):
                    # psd_all = np.zeros(len(band_set) - 1)
                    low, high = band_set[band], band_set[band + 1]
                    idx_min = np.argmax(freqs > low) - 1
                    idx_max = np.argmax(freqs > high) - 1
                    idx = np.zeros(dtype=bool, shape=freqs.shape)
                    idx[idx_min:idx_max] = True
                    psd_all = simps(psd[channel, :][idx], freqs[idx])
                    psd_feature_trial[train_index, band, channel, :] = psd_all

                    # 求de
                    de_feature_trial[train_index, band, channel, :] = 0.5 * np.log2(2 * np.pi * np.exp(1) *
                                                                                    np.var(
                                                                                        segment_band_time_ch[
                                                                                        train_index, band,
                                                                                        :,
                                                                                        channel].T))

        psd_feature = np.vstack([psd_feature, psd_feature_trial])
        de_feature = np.vstack([de_feature, de_feature_trial])

    de_feature = de_feature[1:, :, :, :] #
    psd_feature = psd_feature[1:, :, :, :]
    np.save('E:/资料/code/code_self/DEAP_ML_V5/data/EEG/DREAMER/features/de/' + 's{}.npy'.format(subject_index + 1), de_feature)
    np.save('E:/资料/code/code_self/DEAP_ML_V5/data/EEG/DREAMER/features/psd/' + 's{}.npy'.format(subject_index + 1), psd_feature)
