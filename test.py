import seaborn as sns  # 导入seaborn绘图库
import matplotlib.pyplot as plt
import numpy as np

def plot_confusion_matrix(matrix, normalFlag):
    '''
    martrix, 混淆矩阵
    normalFlag，是否归一化
    '''
    print(matrix)  # 打印出来看看
    if normalFlag:
        matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
        matrix = np.around(matrix, decimals=2)
    print(matrix)  # 打印出来看看
    f, ax = plt.subplots(figsize=(7, 5))
    sns.set(font_scale=2)
    if matrix[0, 0] < 1:
        sns.heatmap(matrix, annot=True, cmap="Blues", ax=ax, vmin=0, vmax=1)  # 如果值小于1，为归一化的画热力图保留两位小数
    else:
        sns.heatmap(matrix, annot=True, cmap="OrRd", ax=ax, fmt="d")  # 否则保留整数

    ax.set_title('Dominance on DREAMER', fontsize=30)  # 标题
    ax.set_xticklabels(['Negative', 'Positive'], fontsize=30)
    ax.set_yticklabels(['Negative', 'Positive'], fontsize=30, rotation=-90)

    plt.savefig('1.svg', bbox_inches='tight', dpi=400)
    plt.show()

matrix = [[118, 5], [4, 115]]
matrix = np.array(matrix)
plot_confusion_matrix(matrix, True)