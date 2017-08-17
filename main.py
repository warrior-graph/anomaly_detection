import numpy as np
from sys import argv
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


def main():
    m = 1
    data = []
    kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0))
    with open('/home/marques/Downloads/s1.txt/s1.txt') as file:
        for line in file:
            data += list(map(int, line.split()))

    data = [data[i:i+900] for i in range(0, len(data), 900)]

    data = [sum(x) for x in data]

    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)

    for i in range(2880 - m * 24):
        window = data[i:i + m * 24]
        T = np.arange(i, i + len(window), dtype=np.int64)[:, np.newaxis]
        y = np.atleast_2d(window).reshape(-1, 1)
        gp.fit(T, y)
        for horizon in range(1, 4):
            t = np.arange(i + len(window) + 1, i + len(window) + 1 + 96 * horizon, dtype=np.int64).reshape(-1, 1)
            y_pred = gp.predict(t)
            print(y_pred)
        break
if __name__ == '__main__':
    main()



