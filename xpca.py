import numpy as np
import sklearn.decomposition

def estimate_eigenvalues(A, X_hat, return_extra=False):
    n = np.shape(A)[0]
    R = np.eye(n) - X_hat.T @ X_hat
    S = X_hat.T @ A @ X_hat
    lmbda = np.empty(n)
    for i in range(n): lmbda[i] = S[i, i] / (1. - R[i, i])
    if return_extra:
        return {
            'lambda': lmbda,
            'R': R,
            'S': S,
        }
    else:
        return lmbda

def ogita_aishima_step(A, X_hat):
    eigenvalues = estimate_eigenvalues(A=A, X_hat=X_hat, return_extra=True)
    R = eigenvalues['R']
    S = eigenvalues['S']
    lmbda = eigenvalues['lambda']
    D = np.diag(lmbda)
    delta = 2. * (np.linalg.norm(S - D, ord=2) + np.linalg.norm(A, ord=2) * np.linalg.norm(R, ord=2))
    n = np.shape(A)[0]
    E = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            if np.abs(lmbda[i] - lmbda[j]) > delta:
                E[i, j] = (S[i, j] + lmbda[j] * R[i, j]) / (lmbda[j] - lmbda[i])
            else:
                E[i, j] = .5 * R[i, j]
    return X_hat + X_hat @ E

def ogita_aishima(
        A, X_hat,
        tol=1e-6,
        max_iter_count=None,
        store_history=False,
        sort_by_eigenvalues=False,
        return_extra=False):
    X_hats = [X_hat]
    epsilons = []
    iter_count = 0
    while True:
        iter_count += 1
        X_hat = ogita_aishima_step(A=A, X_hat=X_hats[-1])
        X_hats.append(X_hat)
        if max_iter_count is not None and iter_count == max_iter_count: break
        epsilons.append(np.linalg.norm(X_hats[-1] - X_hats[-2], ord=2))
        if epsilons[-1] < tol: break
        if not store_history:
            X_hats = X_hats[-1:]
            epsilons = epsilons[-1:]
    X_hat_new = X_hats[-1]
    lmbda = None
    if sort_by_eigenvalues:
        lmbda = estimate_eigenvalues(A=A, X_hat=X_hats[-1], return_extra=False)
        sorted_indices = np.argsort(lmbda)[::-1]
        lmbda = lmbda[sorted_indices]
        X_hat_new = X_hat_new[:,sorted_indices]
        for i in range(len(X_hats)):
            X_hats[i] = X_hats[i][:,sorted_indices]
    if return_extra:
        result = {
            'result': X_hat_new,
            'lmbda': lmbda,
            'iter_count': iter_count,
        }
        if store_history:
            result['X_hats'] = X_hats
            result['epsilons'] = epsilons
        else:
            result['epsilon'] = epsilons[-1]
        if lmbda is not None: result['lambda'] = lmbda
    else:
        result = X_hat_new
    return result

def sorted_eig(A):
    eigenvalues, eigenvectors = np.linalg.eig(A)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:,sorted_indices]
    return eigenvalues, eigenvectors

class IPCA(object):
    def __init__(self, tol=1e-6, max_iter_count=None):
        self.pca = sklearn.decomposition.PCA()
        self.components_ = None
        self.__tol = tol
        self.__max_iter_count = max_iter_count

    def fit(self, X):
        if self.components_ is None:
            self.pca.fit(X)
            self.components_ = self.pca.components_
        else:
            X_raw = X.values if hasattr(X, 'values') else X
            Q = np.cov(X_raw, rowvar=False)
            self.components_ = ogita_aishima(
                    A=Q,
                    X_hat=self.components_.T,
                    tol=self.__tol,
                    max_iter_count=self.__max_iter_count
                ).T

    def clear(self):
        self.components_ = None

    def transform(self, X):
        X_raw = X.values if hasattr(X, 'values') else X
        x_mean = np.mean(X_raw, axis=0)
        X_centred = X - x_mean
        return X_centred @ self.components_.T

class EWMCov(object):
    def __init__(self, alpha):
        self.alpha = alpha
        self.__dim = None
        self.__mean = None
        self.__cov = None

    @property
    def dim(self):
        return self.__dim

    @property
    def mean(self):
        return self.__mean

    @property
    def cov(self):
        return self.__cov

    def add(self, x):
        x = np.reshape(x, (-1, 1))
        if self.__mean is None:
            self.__mean = x
            self.__dim = np.size(x)
            self.__cov = np.zeros((self.__dim, self.__dim))
        else:
            self.__mean = (1. - self.alpha) * x + self.alpha * self.__mean
            x_centred = x - self.__mean
            self.__cov = (1. - self.alpha) * x_centred @ x_centred.T + self.alpha * self.__cov

class EWMPCA(object):
    def __init__(self, alpha, W_initial=None, tol=1e-6, max_iter_count=None):
        self.__ewmcov = EWMCov(alpha)
        self.__W = W_initial
        self.__tol = tol
        self.__max_iter_count = max_iter_count

    def add(self, x):
        x = np.reshape(x, (-1, 1))
        self.__ewmcov.add(x)
        self.__W = ogita_aishima(
                A=self.__ewmcov.cov,
                X_hat=self.__W,
                tol=self.__tol,
                max_iter_count=self.__max_iter_count,
                sort_by_eigenvalues=True
            )
        x_centered = x - self.__ewmcov.mean
        return x_centered.reshape((1, -1)) @ self.__W

    def add_all(self, xs, verbose=True):
        if self.__W is None:
            # TODO Here we are peeking into the future
            prime_size = min(100, len(xs))
            sample_cov = np.cov(xs[:prime_size,:], rowvar=False)
            self.__W = sorted_eig(sample_cov)[1]
        zs = []
        for i, x in enumerate(xs):
            if verbose:
                if (i + 1) % 1000 == 0: print(f'Processing data point {i + 1}')
            zs.append(self.add(x))
        return np.vstack(zs)
