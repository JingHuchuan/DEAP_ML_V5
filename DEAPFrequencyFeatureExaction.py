import pickle
import numpy as np
import random
import os

from scipy.integrate import simps
from tqdm import tqdm
from scipy.signal import butter, sosfilt
from scipy import signal
import matplotlib.pyplot as plt

if not os.path.exists('./data/'):
    os.mkdir('./data/')

data_path = 'G:/Dataset/DEAP/data_preprocessed_python/'
subjects = os.listdir(data_path)

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


for i, subject in enumerate(tqdm(subjects)):
    print("受试者{}".format(i+1))
    data = list()
    label = list()

    subject_path = data_path + subject
    with open(subject_path, 'rb') as f:
        subject_config = pickle.load(f, encoding='latin1')
        f.close()

    # 去除基线的数据
    subject_data = subject_config['data'][:, 0:32, :]  # (40, 32, 8064)
    base_signal = (subject_data[:, :, 0:128] + subject_data[:, :, 128:256] +
                   subject_data[:, :, 256:384]) / 3

    # 后面60s的数据
    subject_data = subject_data[:, :, 384:8064]
    for t in range(0, 60):
        subject_data[:, :, t * 128:(t + 1) * 128] = subject_data[:, :, t * 128:(t + 1) * 128] - base_signal

    # 滤波
    band_set = [4, 8, 14, 30, 50]
    # trial * segment * band * time * ch
    trial_segment_band_time_ch = np.zeros((40, 60, 4, 128, 32))

    # psd特征
    psd_feature = np.zeros((40, 60, 4, 32, 1))
    # de特征
    de_feature = np.zeros((40, 60, 4, 32, 1))

    for trial in range(trial_segment_band_time_ch.shape[0]):  # 40
        print("实验{}".format(trial+1))
        for train_index in range(trial_segment_band_time_ch.shape[1]):  # 60
            sample_data = subject_data[trial, :, (train_index * 128):((train_index + 1) * 128)]  # (32, 128)
            time_ch = sample_data.T
            for band in range(trial_segment_band_time_ch.shape[2]):  # frequency band
                trial_segment_band_time_ch[trial, train_index, band, :, :] = butter_bandpass_filter(time_ch,
                                                                                                    lowcut=band_set[
                                                                                                        band],
                                                                                                    highcut=band_set[
                                                                                                        band + 1],
                                                                                                    fs=128,
                                                                                                    order=21)

                # 求psd特征
                freqs, psd = signal.periodogram(trial_segment_band_time_ch[trial, train_index, band, :, :].T, fs=128,
                                                window='hann', nfft=512,
                                                scaling='density',
                                                detrend='constant')
                # 求积分，求psd_all，顺便求de
                for channel in range(trial_segment_band_time_ch.shape[4]):
                    # psd_all = np.zeros(len(band_set) - 1)
                    low, high = band_set[band], band_set[band + 1]
                    idx_min = np.argmax(freqs > low) - 1
                    idx_max = np.argmax(freqs > high) - 1
                    idx = np.zeros(dtype=bool, shape=freqs.shape)
                    idx[idx_min:idx_max] = True
                    psd_all = simps(psd[channel, :][idx], freqs[idx])
                    psd_feature[trial, train_index, band, channel, :] = psd_all

                    # 求de
                    de_feature[trial, train_index, band, channel, :] = 0.5 * np.log2(2 * np.pi * np.exp(1) *
                                                                                     np.var(trial_segment_band_time_ch[
                                                                                            trial, train_index, band, :,
                                                                                            channel].T))

    feature_psd_array = psd_feature.reshape(40 * 60, 4, 32, 1)
    feature_de_array = de_feature.reshape(40 * 60, 4, 32, 1)

    # np.save('E:/资料/code/code_self/DEAP_ML_V5/data/features/psd/' + 's{}.npy'.format(i + 1), feature_psd_array)
    # np.save('E:/资料/code/code_self/DEAP_ML_V5/data/features/de/' + 's{}.npy'.format(i + 1), feature_de_array)
