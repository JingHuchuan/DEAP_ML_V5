import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

# path = "D:\sh\deap\deap_data\Valence.csv"
# df = pd.read_csv(path)
# label = df["label"]
# data = np.array(df['Valence']).reshape(-1,1)

path = "D:\sh\deap\deap_data\label_four.csv"
df = pd.read_csv(path)
label = df["label"]
data = np.array(label).reshape(-1, 1)
param = [{'kernel': ['rbf'], 'gamma': [0.1, 1], 'C': [1, 10], }]
# param = [{'kernel': ['rbf'],"gamma":[0.001,0.01,0.1,1,10,100],
#             "C":[0.001,0.01,0.1,1,10,100]}]

# print(np.array(data).shape)
# print(np.array(label).shape)
x_train, x_test, y_train, y_test = train_test_split(data, label, train_size=0.7, random_state=0)
clf = SVC()
grid_search = GridSearchCV(estimator=SVC(), param_grid=param, cv=5, n_jobs=5)
grid_search.fit(x_train, y_train)
print(f"最好的参数为：{grid_search.best_params_}")
print(f"训练集的最佳为score为：{format((grid_search.best_score_ * 100), '.2f')}%")

# 进行十折交叉验证
scores = cross_val_score(grid_search, x_test, y_test, cv=10, scoring='f1_macro')
mean_acc = format((np.mean(scores) * 100), '.2f')
std = format(np.std(scores), '.4f')
var = format(np.var(scores), '.4f')

print(f"平均准确率为：{mean_acc}%,标准差为:{std},方差为:{var}")
