from sklearn.datasets import load_svmlight_file
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

# Load datasets
print('Loading datasets...')
X_train, y_train = load_svmlight_file("a9a.txt")

X_test, y_test = load_svmlight_file("a9atest.txt")

print(f'x train: {X_train.shape}, y train: {y_train.shape}')
print(f'x test: {X_test.shape}, y test: {y_test.shape}')

# cross validation 3-fold
C_values = [0.01, 0.05, 0.1, 0.5, 1]
results_linear = {}

for C in C_values:
    model = SVC(kernel="linear", C=C)
    scores = cross_val_score(model, X_train, y_train, cv=3)
    results_linear[C] = scores.mean()

print("Linear Kernel Results:", results_linear)