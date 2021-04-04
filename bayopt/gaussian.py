import numpy as np


class gaussian:

    def __init__(self, func, theta_mean, covTheta, covNoise):
        self.theta_mean = theta_mean
        self.covTheta = covTheta
        self.covNoise = covNoise
        self.posterior_mean = None
        self.posterior_cov = None
        self.matrix_G = None
        self.Data_tr = None
        self.out_tr = None
        self.func = func

    def update_cov_noise(self, COV):
        self.covNoise = COV

    def update_function(self, new_func):
        self.func = new_func

    def fit(self, X, y):
        self.covNoise = self.covNoise + 1e-4 * np.eye(self.covNoise.shape[0], dtype=complex)
        self.covTheta = self.covTheta + 1e-4 * np.eye(self.covTheta.shape[0], dtype=complex)

        self.Data_tr = self.func(X)  # save training data
        self.out_tr = y  # save output   training
        self.calc_matrix_G()  # calc auxiliar matrix
        self.posterior_cov_matrix()  # posteriori Covariance matrix
        self.posterior_mean_vector()  # posteriori Covariance matrix

    def model_order(self):
        return self.theta_mean.size

    def calc_matrix_G(self):
        invTheta = np.linalg.pinv(self.covTheta)
        invCovNoise = np.linalg.pinv(self.covNoise)
        self.matrix_G = invTheta + self.Data_tr.conj().T @ invCovNoise @ self.Data_tr

    def posterior_mean_vector(self):
        self.posterior_mean = self.theta_mean + self.posterior_cov @ self.Data_tr.conj().T @ np.linalg.pinv(
            self.covNoise) @ (self.out_tr - self.Data_tr @ self.theta_mean)

    def posterior_cov_matrix(self):
        self.posterior_cov = np.linalg.pinv(self.matrix_G)

    def predict(self, x):
        X = self.func(x)
        theta_mean = X @ self.theta_mean
        sigma = self.covNoise[-1, -1] + X @ self.posterior_cov @ X.conj().T

        return theta_mean, sigma


def eMM_BSL(B, Y, a_gamma, b_gamma, c_noise, d_noise, n_max=50):
    # multiple measurements - bayesian sparse optimization

    row, col = Y.shape

    # N -> number of samples per vector measurement
    # NM -> number of vector measurements

    K = B.shape[1]
    # K -> dimension of the sparse domain

    a = np.random.randn(K)
    g = np.zeros(K)
    M = np.ones((K,col), dtype=complex)

    beta = 1
    error = 10
    n_iter =0

    while (error >= 1e-5) & (n_iter < n_max):
        M_old = M
        A = np.diag(a)

        SIGMA = np.linalg.pinv(A + beta * B.conj().T @ B)
        M = beta * SIGMA @ B.conj().T @ Y

        for kk in range(K):
            a[kk] = (1 + 2 * a_gamma) / (np.abs(M[kk,kk]) ** 2 + np.abs(SIGMA[kk, kk]) + 2 * b_gamma)
            g[kk] = 1 - a[kk] * np.abs(SIGMA[kk, kk])

        beta = (row - np.sum(g) + 2 * c_noise) / (
                            np.linalg.norm(Y - B @ M,'fro') ** 2 + 2 * d_noise)
        error = np.linalg.norm(Y - B @ M,'fro')
        n_iter = n_iter + 1

    return M, 1 / np.mean(beta), error


def EM_linearRegression(B, y, a_gamma, b_gamma, c_noise, d_noise, n_max=100):
    N = y.size
    K = int(B.size / N)
    a = np.random.randn(K)
    g = np.zeros(K)
    mu = np.ones(K, dtype=complex)

    beta = 1
    error = 10
    n_iter = 0

    while (error >= 1e-5) & (n_iter < n_max):
        mu_old = mu
        A = np.diag(a)
        SIGMA = np.linalg.pinv(A + beta * B.conj().T @ B)
        mu = beta * SIGMA @ B.conj().T @ y
        for kk in range(K):
            a[kk] = (1 + 2 * a_gamma) / (np.abs(mu[kk]) ** 2 + np.real(SIGMA[kk, kk]) + 2 * b_gamma)
            g[kk] = 1 - a[kk] * np.real(SIGMA[kk, kk])

        beta = (N - np.linalg.norm(g) + 2 * c_noise) / (np.linalg.norm(y - B @ mu) ** 2 + 2 * d_noise)
        error = np.linalg.norm(mu_old - mu) ** 2
        n_iter = n_iter + 1

    return mu, 1 / beta
