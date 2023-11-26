import numpy as np

fatherPath = "E:/资料/code/code_self/DEAP_ML_V5/data/EEG/DEAP/eachSub/data/"
for i in range(32):
    print(i)

    data = np.load(fatherPath + 's{}.npy'.format(i + 1))
    feature_mean = np.mean(data, axis=2)
    feature_std = np.std(data, axis=2)
    feature_one_diff = np.diff(data, axis=2)
    feature_one_diff = np.mean(np.abs(feature_one_diff), axis=2)  # 一阶差分绝对值的平均值
    slection_order = [4, 3, 0, 25, 7, 2, 21, 20, 16, 29, 11, 1, 17, 6, 19, 8, 26, 9, 30, 27, 15, 31, 24, 22, 13, 28,
                      23, 18, 5, 10, 12, 14]
    feature_one_diff_selection = feature_one_diff[:, slection_order]

    # np.save('E:/资料/code/code_self/DEAP_ML_V5/data/EEG/DEAP/features/mean/' + 's{}.npy'.format(i + 1), feature_mean)
    # np.save('E:/资料/code/code_self/DEAP_ML_V5/data/EEG/DEAP/features/std/' + 's{}.npy'.format(i + 1), feature_std)
    # np.save('E:/资料/code/code_self/DEAP_ML_V5/data/EEG/DEAP/features/diff/' + 's{}.npy'.format(i + 1),
    #         feature_one_diff)

    np.save('E:/资料/code/code_self/DEAP_ML_V5/data/EEG/DEAP/features/selection/' + 's{}.npy'.format(i + 1),
            feature_one_diff_selection)
