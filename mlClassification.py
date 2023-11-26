import numpy as np
import os
import scipy.io as scio
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from scipy.signal import savgol_filter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.svm import SVC

import time

if __name__ == '__main__':
    # fatherPath = "E:/资料/code/code_self/DEAP_ML_V5/data/EEG/DREAMER/"
    fatherPath = "E:/资料/code/code_self/DEAP_ML_V5/data/EEG/DEAP/"

    meanPath = fatherPath + "features/mean/"
    stdPath = fatherPath + "features/std/"
    diffPath = fatherPath + "features/diff/"
    dePath = fatherPath + "features/de/"
    psdPath = fatherPath + "features/psd/"
    labelPath = fatherPath + "eachSub/label/"

    result_10_fold = []

    time_record_each_fold = np.zeros((10, 32))
    acc_record_each_fold = np.zeros((10, 32))
    # time_record_each_fold = np.zeros((32, 10, 32))
    # acc_record_each_fold = np.zeros((32, 10, 32))

    # # c代表32个通道的不断变化
    # for c in range(32):
    #     print("运行到通道还剩"+str(32-c))
    #     for cnt in range(10):
    #         print("第{}次实验".format(cnt + 1))
    #
    #         accuracySum = 0
    #         for i in range(32):
    #             print("[===============================训练到第" + str(i + 1) + "个受试者===============================]")
    #             dataStruct = []
    #             # featureMean = np.load(meanPath + "s{}.npy".format(i+1))
    #             featureStd = np.load(stdPath + "s{}.npy".format(i + 1))
    #             # featureDiff = np.load(diffPath + "s{}.npy".format(i + 1))
    #             # featureDiff = featureDiff[:, :32-c]
    #             # if featureDiff.shape[1] == 1:
    #             #     featureDiff = featureDiff.reshape((featureDiff.shape[0], 1))
    #
    #             # featureDe = np.load(dePath + "s{}.npy".format(i + 1))
    #             # featurePsd = np.load(psdPath + "s{}.npy".format(i+1))
    #             label = np.load(labelPath + "s{}.npy".format(i + 1))[:, 0]
    #
    #             bandList = ["theta", "alpha", "beta", "gamma"]
    #
    #             # print("使用的特征为gamma频段的psd特征")
    #             # featureDeSubBand = featurePsd[:, 3, :, 0]
    #             # featureDeSubBand = featurePsd[:, 3, :, 0]
    #
    #             featureDeSubBand = featureStd
    #
    #             slection_order = [4, 3, 0, 25, 7, 2, 21, 20, 16, 29, 11, 1, 17, 6, 19, 8, 26, 9, 30, 27, 15, 31, 24, 22,
    #                               13, 28,
    #                               23, 18, 5, 10, 12, 14]
    #
    #             featureDeSubBandSelection = featureDeSubBand[:, slection_order]
    #             featureDeSubBandSelection = featureDeSubBandSelection[:, :32-c]
    #             if featureDeSubBandSelection.shape[1] == 1:
    #                 featureDeSubBandSelection = featureDeSubBandSelection.reshape((featureDeSubBandSelection.shape[0], 1))
    #
    #             # featureDeSubBand = np.array(featureDeSubBand)
    #
    #             # 训练集和验证集划分
    #             x_train, x_test, y_train, y_test = train_test_split(featureDeSubBandSelection, label, test_size=0.2, shuffle=True)
    #
    #             time_start = time.perf_counter()  # 记录开始时间
    #             # clf = SVC(kernel='rbf', random_state=0)
    #             clf = RandomForestClassifier(random_state=0)
    #             clf.fit(x_train, y_train.ravel())
    #             y_predict = clf.predict(x_test)
    #             time_end = time.perf_counter()  # 记录结束时间
    #
    #             time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
    #             print('运行时间为：' + str(time_sum))
    #
    #             time_record_each_fold[c, cnt, i] = time_sum  # 记录时间
    #
    #             accuracy = accuracy_score(y_test, y_predict)
    #             accuracySum += accuracy
    #             print("受试者" + str(i + 1) + "的分类精度为：%.4f" % accuracy)
    #
    #             acc_record_each_fold[c, cnt, i] = accuracy
    #         print("[==============================第" + str(i + 1) + "个受试者训练结束==============================]")
    #         print("[==============================所有受试者训练结束！==============================]")
    #
    #         # 打印平均分类精度
    #         print("[==========================所有受试者的平均精度为：%.4f==========================]" % np.mean(accuracySum / 32))
    #         # result = np.array(result)
    #         # # np.save("DE-Gamma.npy", result)
    #         result_10_fold.append(np.mean(accuracySum / 32))
    #     print(result_10_fold)
    #
    # np.save(stdPath + 'record/selection/timeRecord.npy', time_record_each_fold)
    # np.save(stdPath + 'record/selection/accRecord.npy', acc_record_each_fold)

    for cnt in range(10):
        print("第{}次实验".format(cnt + 1))

        accuracySum = 0
        for i in range(23):
            print("[===============================训练到第" + str(i + 1) + "个受试者===============================]")
            dataStruct = []
            featureMean = np.load(meanPath + "s{}.npy".format(i+1))
            # featureStd = np.load(stdPath + "s{}.npy".format(i + 1))
            # featureDiff = np.load(diffPath + "s{}.npy".format(i + 1))
            # featureDiff = featureDiff[:, :32-c]
            # if featureDiff.shape[1] == 1:
            #     featureDiff = featureDiff.reshape((featureDiff.shape[0], 1))

            # featureDe = np.load(dePath + "s{}.npy".format(i + 1))
            # featurePsd = np.load(psdPath + "s{}.npy".format(i+1))
            label = np.load(labelPath + "s{}.npy".format(i + 1))[:, 0]

            bandList = ["theta", "alpha", "beta", "gamma"]

            # print("使用的特征为gamma频段的psd特征")
            # featureDeSubBand = featureDe[:, 4, :, 0]
            # featureDeSubBand = featurePsd[:, 4, :, 0]

            # featureDeSubBand = featureDiff

            # 训练集和验证集划分
            x_train, x_test, y_train, y_test = train_test_split(featureMean, label, test_size=0.2, shuffle=True)

            time_start = time.perf_counter()  # 记录开始时间
            # clf = SVC(kernel='rbf', random_state=0)
            clf = RandomForestClassifier(random_state=0)
            clf.fit(x_train, y_train.ravel())
            y_predict = clf.predict(x_test)
            time_end = time.perf_counter()  # 记录结束时间

            time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
            print('运行时间为：' + str(time_sum))

            time_record_each_fold[cnt, i] = time_sum  # 记录时间

            accuracy = accuracy_score(y_test, y_predict)
            accuracySum += accuracy
            print("受试者" + str(i + 1) + "的分类精度为：%.4f" % accuracy)

            acc_record_each_fold[cnt, i] = accuracy
        print("[==============================第" + str(i + 1) + "个受试者训练结束==============================]")
        print("[==============================所有受试者训练结束！==============================]")

        # 打印平均分类精度
        print("[==========================所有受试者的平均精度为：%.4f==========================]" % np.mean(accuracySum / 23))
        # result = np.array(result)
        # # np.save("DE-Gamma.npy", result)
        result_10_fold.append(np.mean(accuracySum / 23))
    print(result_10_fold)

    np.save(dePath + 'record/gamma/timeRecord.npy', time_record_each_fold)
    np.save(dePath + 'record/gamma/accRecord.npy', acc_record_each_fold)
