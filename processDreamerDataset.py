import scipy.io as scio
import numpy as np
import os

from tqdm import tqdm

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

for subject_index in range(23):
    print(subject_index)
    data = list()
    label = list()

    subject_label = list()
    valence_label = data_dict[subject_index][0][0][4]  # valence_label  (18, )
    valence_label = valence_label.reshape(valence_label.shape[0], )
    subject_label.append(valence_label)
    arousal_label = data_dict[subject_index][0][0][5]  # arousal_label
    arousal_label = arousal_label.reshape(arousal_label.shape[0], )
    subject_label.append(arousal_label)
    dominance_label = data_dict[subject_index][0][0][6]  # dominance_label
    dominance_label = dominance_label.reshape(dominance_label.shape[0], )
    subject_label.append(dominance_label)
    subject_label = np.array(subject_label).T  # (18, 3)

    for trial_index in range(18):
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

        for j in range(data_segment_length):
            # sample_data = origin_data[:, window_size * j: window_size * (j + 1)] - baseline_signal
            sample_data = origin_data[:, window_size * j: window_size * (j + 1)]
            data.append(sample_data)

            label.append((subject_label[trial_index] >= 3).astype(int))

    data = np.array(data, dtype=np.float64)
    label = np.array(label, dtype=np.int32)

    np.save('./data/EEG/DREAMER/eachSub/withBaseline/data/' + 's{}.npy'.format(subject_index+1), data)
    np.save('./data/EEG/DREAMER/eachSub/withBaseline/label/' + 's{}.npy'.format(subject_index+1), label)


