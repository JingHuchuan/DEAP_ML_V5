import math
import pickle
import numpy as np
import random
import os
from tqdm import tqdm

if not os.path.exists('./data/'):
    os.mkdir('./data/')

data_path = 'G:/Dataset/DEAP/data_preprocessed_python/'
subjects = os.listdir(data_path)

for i, subject in enumerate(tqdm(subjects)):
    data = list()
    label = list()

    subject_path = data_path + subject
    with open(subject_path, 'rb') as f:
        subject_config = pickle.load(f, encoding='latin1')
        f.close()

    # EEG信号的处理
    # # 去除基线的数据
    # subject_data = subject_config['data'][:, 0:32, :]  # (40, 32, 8064)
    # base_signal = (subject_data[:, :, 0:128] + subject_data[:, :, 128:256] +
    #                subject_data[:, :, 256:384]) / 3
    #
    # # 后面60s的数据
    # subject_data = subject_data[:, :, 384:8064]
    # for t in range(0, 60):
    #     subject_data[:, :, t * 128:(t + 1) * 128] = subject_data[:, :, t * 128:(t + 1) * 128] - base_signal
    #
    # subject_label = subject_config['labels']  # (40, 4)
    #
    # for j in range(subject_data.shape[0]):
    #     for train_index in range(60):  # 60s
    #         # 按窗将数据划分
    #         sample_data = subject_data[j, :, (train_index * 128):((train_index + 1) * 128)]  # (32, 128)
    #
    #         data.append(sample_data)         # (32, 128)
    #         label.append((subject_label[j] > 5.).astype(int))  # (5)
    #
    # data = np.array(data, dtype=np.float64)
    # label = np.array(label, dtype=np.int32)
    #
    # np.save('./record/allSub/data/' + 's{}.npy'.format(i+1), data)
    # np.save('./record/allSub/label/' + 's{}.npy'.format(i+1), label)

    # EOG信号的处理
    subject_data = subject_config['data'][:, 32:34, :]  # (40, 2, 8064)
    # base_signal = (subject_data[:, :, 0:128] + subject_data[:, :, 128:256] +
    #                subject_data[:, :, 256:384]) / 3

    subject_data = subject_data[:, :, 384:8064]
    # for t in range(0, 60):
    #     subject_data[:, :, t * 128:(t + 1) * 128] = subject_data[:, :, t * 128:(t + 1) * 128] - base_signal

    for j in range(subject_data.shape[0]):
        for train_index in range(60):  # 60s
            # 按窗将数据划分
            sample_data = subject_data[j, :, (train_index * 128):((train_index + 1) * 128)]  # (32, 128)

            data.append(sample_data)         # (32, 128)

    data = np.array(data, dtype=np.float64)

    np.save('E:/资料/code/code_self/DEAP_ML_V5/data/EOG/DEAP/' + 's{}.npy'.format(i+1), data)