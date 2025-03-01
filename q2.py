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

gamma_values = [0.01, 0.05, 0.1, 0.5, 1]
results_rbf = {}

for C in C_values:
    print(f'Train with C={C}')
    for gamma in gamma_values:
        print(f'Train with gamma={gamma}')
        model = SVC(kernel="rbf", C=C, gamma=gamma)
        scores = cross_val_score(model, X_train, y_train, cv=3)
        results_rbf[(C, gamma)] = scores.mean()
        print(f'Done with gamma={gamma}')
    print(f'Done with C={C}')

print("RBF Kernel Results:", results_rbf)
