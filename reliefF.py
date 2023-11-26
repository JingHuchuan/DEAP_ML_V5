from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from ReliefF import ReliefF

digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target)

fs = ReliefF(n_neighbors=100, n_features_to_keep=10)
X_train = fs.fit_transform(X_train, y_train)
X_test_subset = fs.transform(X_test)
print(X_test_subset)
# print(X_test.shape, X_test_subset.shape)
