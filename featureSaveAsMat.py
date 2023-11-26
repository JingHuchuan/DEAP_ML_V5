import scipy.io as io
import numpy as np

feature_path = 'E:/资料/code/code_self/DEAP_ML_V5/data/EEG/DEAP/features/mean/'
label_path = 'E:/资料/code/code_self/DEAP_ML_V5/data/EEG/DEAP/eachSub/label/'
save_path = 'E:/资料/code/code_self/DEAP_ML_V5/data/mat/mean/'

for i in range(32):
    feature = np.load(feature_path + 's{}.npy'.format(i+1))
    label = np.load(label_path + 's{}.npy'.format(i+1))
    io.savemat(save_path + 's{}-feature.mat'.format(i+1), {'feature': feature})
    io.savemat(save_path + 's{}-label.mat'.format(i+1), {'label': label})


