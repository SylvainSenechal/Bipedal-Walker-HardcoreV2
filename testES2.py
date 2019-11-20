import multiprocessing as mp
import random
import string
import timeit
import platform
from matplotlib import pyplot as plt

import numpy as np

def parzen_estimation(x_samples, point_x, h):
    """
    Implementation of a hypercube kernel for Parzen-window estimation.

    Keyword arguments:
        x_sample:training sample, 'd x 1'-dimensional numpy array
        x: point x for density estimation, 'd x 1'-dimensional numpy array
        h: window width

    Returns the predicted pdf as float.

    """
    k_n = 0
    for row in x_samples:
        x_i = (point_x - row[:,np.newaxis]) / (h)
        for row in x_i:
            if np.abs(row) > (1/2):
                break
        else: # "completion-else"*
            k_n += 1
    return (k_n / len(x_samples)) / (h**point_x.shape[1])

def serial(samples, x, widths):
    return [parzen_estimation(samples, x, w) for w in widths]

def multiprocess(processes, samples, x, widths):
    pool = mp.Pool(processes=processes)
    results = [pool.apply_async(parzen_estimation, args=(samples, x, w)) for w in widths]
    results = [p.get() for p in results]
    results.sort() # to sort the results by input window width
    return results

widths = np.linspace(1.0, 1.2, 100)
mu_vec = np.array([0,0])
cov_mat = np.array([[1,0],[0,1]])
n = 10000

x_2Dgauss = np.random.multivariate_normal(mu_vec, cov_mat, n)
widths = np.arange(0.1, 1.3, 0.1)
point_x = np.array([[0],[0]])
results = []

results = multiprocess(4, x_2Dgauss, point_x, widths)

# for r in results:
#     print('h = %s, p(x) = %s' %(r[0], r[1]))
benchmarks = []

benchmarks.append(timeit.Timer('serial(x_2Dgauss, point_x, widths)',
            'from __main__ import serial, x_2Dgauss, point_x, widths').timeit(number=1))

benchmarks.append(timeit.Timer('multiprocess(2, x_2Dgauss, point_x, widths)',
            'from __main__ import multiprocess, x_2Dgauss, point_x, widths').timeit(number=1))

benchmarks.append(timeit.Timer('multiprocess(3, x_2Dgauss, point_x, widths)',
            'from __main__ import multiprocess, x_2Dgauss, point_x, widths').timeit(number=1))

benchmarks.append(timeit.Timer('multiprocess(4, x_2Dgauss, point_x, widths)',
            'from __main__ import multiprocess, x_2Dgauss, point_x, widths').timeit(number=1))

benchmarks.append(timeit.Timer('multiprocess(6, x_2Dgauss, point_x, widths)',
            'from __main__ import multiprocess, x_2Dgauss, point_x, widths').timeit(number=1))
benchmarks.append(timeit.Timer('multiprocess(12, x_2Dgauss, point_x, widths)',
            'from __main__ import multiprocess, x_2Dgauss, point_x, widths').timeit(number=1))
benchmarks.append(timeit.Timer('multiprocess(32, x_2Dgauss, point_x, widths)',
            'from __main__ import multiprocess, x_2Dgauss, point_x, widths').timeit(number=1))

def print_sysinfo():

    print('\nPython version  :', platform.python_version())
    print('compiler        :', platform.python_compiler())

    print('\nsystem     :', platform.system())
    print('release    :', platform.release())
    print('machine    :', platform.machine())
    print('processor  :', platform.processor())
    print('CPU count  :', mp.cpu_count())
    print('interpreter:', platform.architecture()[0])
    print('\n\n')

def plot_results():
    bar_labels = ['serial', '2', '3', '4', '6']

    fig = plt.figure(figsize=(10,8))

    # plot bars
    y_pos = np.arange(len(benchmarks))
    plt.yticks(y_pos, bar_labels, fontsize=16)
    bars = plt.barh(y_pos, benchmarks,
             align='center', alpha=0.4, color='g')

    # annotation and labels

    for ba,be in zip(bars, benchmarks):
        plt.text(ba.get_width() + 2, ba.get_y() + ba.get_height()/2,
                '{0:.2%}'.format(benchmarks[0]/be),
                ha='center', va='bottom', fontsize=12)

    plt.xlabel('time in seconds for n=%s' %n, fontsize=14)
    plt.ylabel('number of processes', fontsize=14)
    t = plt.title('Serial vs. Multiprocessing via Parzen-window estimation', fontsize=18)
    plt.ylim([-1,len(benchmarks)+0.5])
    plt.xlim([0,max(benchmarks)*1.1])
    plt.vlines(benchmarks[0], -1, len(benchmarks)+0.5, linestyles='dashed')
    plt.grid()

    plt.show()
plot_results()
print_sysinfo()
# import numpy as np
# from time import time
#
# from multiprocessing import Pool
# # print("Number of processors: ", mp.cpu_count())
#
#
# pool = Pool()
#
# np.random.RandomState(100)
# arr = np.random.randint(0, 10, size=[2000000, 5])
# data1 = arr.tolist()
# data1[:5]
#
# arr = np.random.randint(0, 10, size=[2000000, 5])
# data2 = arr.tolist()
# data2[:5]
# print('oui')
#
# def howmany_within_range(row, minimum, maximum):
#     """Returns how many numbers lie within `maximum` and `minimum` in a given `row`"""
#     count = 0
#     for n in row:
#         if minimum <= n <= maximum:
#             count = count + 1
#     return count
#
# def f1(data):
#     results = []
#     for row in data:
#         results.append(howmany_within_range(row, minimum=4, maximum=8))
#     return results[:10]
#
#
# result1 = f1(data1)
# result2 = f1(data2)
# print(result1)
# print(result2)
#
# # res = pool.map(f1, [data1, data2])
# result1 = pool.apply_async(f1, [data1])    # evaluate "solve1(A)" asynchronously
# result2 = pool.apply_async(f1, [data2])
# answer1 = result1.get(timeout=10)
# answer2 = result2.get(timeout=10)
# print(answer1)
# print(answer2)



# hl_size = 3
#
# a = np.random.randn(24, hl_size)
# print(a)
# a= a / np.sqrt(24)
# print('oui')
# print(a)
# b = np.random.randn(hl_size, 4) / np.sqrt(hl_size)
#
# print(np.sqrt(24))










# solution = np.array([0.5, 0.1, -0.3])
#
#
# def fnnn(w): return -np.sum((w - solution)**2)
#
#
#
# npop = 50      # population size
# sigma = 0.1    # noise standard deviation
# alpha = 0.001  # learning rate
#
# w = np.random.randn(3) # initial guess
#
#
#
# for i in range(3000):
#
#   N = np.random.randn(npop, 3)
#   R = np.zeros(npop)
#
#   for j in range(npop):
#
#     w_try = w + sigma*N[j]
#     R[j] = fnnn(w_try)
#
#   A = (R - np.mean(R)) / np.std(R)
#
#   # print("#############")
#   # print(np.dot(N.T, A))
#   # print(A)
#   # print(N.T)
#   w = w + alpha/(npop*sigma) * np.dot(N.T, A)
#   # print(w)
#
# print(w)
