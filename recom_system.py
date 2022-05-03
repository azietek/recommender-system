import argparse
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.metrics import mean_squared_error
from scipy import sparse
from sklearn.linear_model import SGDRegressor
from sklearn import preprocessing
import warnings

warnings.filterwarnings("ignore")


def ParseArguments():
    parser = argparse.ArgumentParser(description="Recommendation System s315221")
    parser.add_argument('--train', default="train_ratings.csv", required=False,
                        help='file with csv file containing training data (default: %(default)s)')
    parser.add_argument('--test', default="test_ratings.csv", required=False,
                        help='file with csv file containing testing data (default: %(default)s)')
    parser.add_argument('--alg', default="SVD1", required=False,
                        help='algorithm that you want to use (default: %(default)s)')
    parser.add_argument('--result', default="", required=False,
                        help='file where a final RMSE will be saved (default: %(default)s)')

    args = parser.parse_args()

    return args.train, args.test, args.alg, args.result


training_file, testing_file, alg, output_file = ParseArguments()

# wczytanie tabel z plików
training = pd.read_csv(training_file)
testing = pd.read_csv(testing_file)

# tworzenie tabeli na wzór ratings.csv
ratings = training.append(testing, ignore_index=True)
# matrix_shape = ratings.pivot_table(columns='movieId', index='userId', values='rating').reset_index().shape

# rozmiar oczekiwanych macierzy
n_users = len(set(ratings['userId']))
n_items = len(set(ratings['movieId']))
matrix_shape = (n_users, n_items)

# konwersja kolumn z tabeli na kolejne inty
dict_vals = list(set(ratings['movieId']))
zip_iterator = zip(dict_vals, [i for i in range(len(dict_vals))])
dictionary = dict(zip_iterator)

train_index = training[['userId', 'movieId']].values
train_rating = training['rating'].values

test_index = testing[['userId', 'movieId']].values
test_rating = testing['rating'].values

def convertmatrix(X, y, shape):
    row = X[:, 0]
    col = []
    for movie_id in X[:, 1]:
        col.append(dictionary.get(movie_id))
    data = y
    matrix_sparse = sparse.csr_matrix((data, (row, col)), shape=(shape[0] + 1, shape[1] + 1))
    R = matrix_sparse.todense()
    R = R[1:, :]
    R = np.asarray(R)
    R[R == 0] = np.nan
    return R


# oczekiwane macierze
Z = convertmatrix(train_index, train_rating, matrix_shape)
T = convertmatrix(test_index, test_rating, matrix_shape)


def check_filling(T, kind):
    Z = T.copy()
    if kind == 'zeros':
        Z[np.isnan(Z)] = 0
    elif kind == 'film_mean':
        col_mean = np.nan_to_num(np.nanmean(Z, axis=0))
        indices = np.where(np.isnan(Z))
        Z[indices] = np.take(col_mean, indices[1])
    elif kind == 'user_mean':
        row_mean = np.nan_to_num(np.nanmean(Z, axis=1))
        indices = np.where(np.isnan(Z))
        Z[indices] = np.take(row_mean, indices[0])
    return Z


def SVD1(M, r=13, filling='user_mean'):
    A = check_filling(M, filling)

    svd = TruncatedSVD(n_components=r, random_state=42)
    svd.fit(A)
    Sigma2 = np.diag(svd.singular_values_)
    VT = svd.components_

    W = svd.transform(A) / svd.singular_values_
    H = np.dot(Sigma2, VT)

    A_approximated = np.dot(W, H)

    # pozostawienie oryginalnych wartości
    for (x, y), value in np.ndenumerate(M):
        if not np.isnan(M[x, y]):
            A_approximated[x, y] = M[x, y]

    return A_approximated


def SVD2(M, r=10, filling='user_mean'):
    A_approximated = SVD1(M, r, filling=filling)

    svd = TruncatedSVD(n_components=r, random_state=42)
    svd.fit(A_approximated)
    Sigma2 = np.diag(svd.singular_values_)
    VT = svd.components_

    W = svd.transform(A_approximated) / svd.singular_values_
    H = np.dot(Sigma2, VT)

    A_SVD2 = np.dot(W, H)

    for (x, y), value in np.ndenumerate(M):
        if not np.isnan(M[x, y]):
            A_SVD2[x, y] = M[x, y]

    return A_SVD2


def NMF_(M, r=2, filling='user_mean'):
    Z = check_filling(M, filling)

    model = NMF(n_components=r, init='random', random_state=0)

    W = model.fit_transform(Z)
    H = model.components_

    Z_approximated = np.dot(W, H)

    for (x, y), value in np.ndenumerate(M):
        if not np.isnan(M[x, y]):
            Z_approximated[x, y] = M[x, y]

    return Z_approximated


def my_rmse(Z, T):
    res = 0
    i = 0
    for (x, y), value in np.ndenumerate(T):
        if not np.isnan(T[x, y]):
            res += mean_squared_error([Z[x, y]], [T[x, y]])
            i += 1
    return np.sqrt(res / i)


def SGD(train_index, test_index, train_rating, test_rating, max_iters=30, pen='l1'):
    X_train, X_test, y_train, y_test = train_index, test_index, train_rating, test_rating

    scaler = preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = pd.DataFrame(data=X_train)
    X_train['rating'] = list(y_train)
    X_test = pd.DataFrame(data=X_test)
    X_test['rating'] = list(y_test)

    clf = SGDRegressor(max_iter=max_iters, penalty=pen)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return y_pred


def sgd_rmse(Z, T):
    T_vals = []
    i = 0
    for (x, y), value in np.ndenumerate(T):
        if not np.isnan(T[x, y]):
            T_vals.append(T[x, y])
            i += 1
    return np.sqrt(mean_squared_error(T_vals, Z))


if alg == 'NMF':
    result = my_rmse(NMF_(Z), T)
elif alg == 'SVD1':
    result = my_rmse(SVD1(Z), T)
elif alg == 'SVD2':
    result = my_rmse(SVD2(Z), T)
elif alg == 'SGD':
    result = sgd_rmse(SGD(train_index, test_index, train_rating, test_rating), T)
else:
    raise TypeError("Available names: SVD1, SVD2, NMF, SGD")

if output_file == "":
    print(alg, "algorithm, RMSE value:\n", result)
else:
    print("Saving RMSE value to", output_file)
    with open(output_file, "w") as file:
        print(result, file=file)

# python3 recom_system_315221.py

