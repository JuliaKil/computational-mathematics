import numpy as np
import math
from timeit import default_timer as timer


def swaplines(A, i, j):
    operations = 0
    if(i != j):
        C = np.copy(A)
        operations += 2 * len(A)
        A[i], A[j] = C[j], C[i]
    return A, operations


def swapcolumns(A, i, j):
    operations = 0
    if(i != j):
        C = np.copy(A)
        operations += 2 * len(A)
        A[::, i], A[::, j] = C[::, j], C[::, i]
    return A, operations


def LUdecomposition(A):
    operations = 0
    U = np.copy(A)
    L = np.zeros((len(A), len(A)))
    P = np.eye(len(A))
    Q = np.eye(len(A))
    for k in range(len(A) - 1):
        u = abs(U)
        if u[k:, k:].sum() < pow(10, -14):
            break
        i, j = np.where(u[k:, k:] == u[k:, k:].max())
        i[0] += k
        j[0] += k
        U, operationsUl = swaplines(U, k, i[0])
        U, operationsUc = swapcolumns(U, k, j[0])
        L, operationsLl = swaplines(L, k, i[0])
        L, operationsLc = swapcolumns(L, k, j[0])
        P, o = swaplines(P, k, i[0])
        Q, o = swapcolumns(Q, k, j[0])
        operations += operationsUl + operationsUc + operationsLl + operationsLc
        M = np.eye(len(A))
        for l in range(k+1, len(A)):
            L[l][k] = U[l][k] / U[k][k]
            M[l][k] = (-1)*L[l][k]
            operations += 2
        U = np.dot(M, U)
        operations += pow(len(A), 3) + (len(A)-1)*(pow(len(A), 2))
    np.fill_diagonal(L, 1)
    operations += len(A)
    return L, U, P, Q, operations


def solvedown(A, b):
    x = np.zeros(len(A))
    x[0] = b[0]
    operations = 1
    for k in range(1, len(A)):
        x[k] = b[k] - np.dot(A[k, :k ],x[:k].T)
        operations += 2*k
    return x, operations


def solveup(A, b):
    x = np.zeros(len(A))
    x[len(A)-1] = b[len(A)-1]/A[len(A)-1, len(A)-1]
    operations = 2
    for k in range(len(A)-2, -1, -1):
        x[k] = (b[k] - np.dot(A[k, k+1:],x[k+1:].T))/A[k][k]
        operations = 2*k + 1
    return x, operations


def LUsolve(L, U, P, Q, b):
    y, operations_down = solvedown(L, np.dot(P, b.T))
    z, operations_up = solveup(U, y)
    x = np.dot(Q, z)
    operations = operations_down + operations_up + pow(len(U), 2) + len(U)*(len(U)-1)
    return x, operations


def F(x):
    F = np.mat([
        math.cos(x[1] * x[0]) - math.exp(-3 * x[2]) + x[3] * x[4] ** 2 - x[5] - math.sinh(
            2 * x[7]) * x[8] + 2 * x[9] + 2.000433974165385440,
        math.sin(x[1] * x[0]) + x[2] * x[8] * x[6] - math.exp(-x[9] + x[5]) + 3 * x[4] ** 2 - x[5] * (x[7] + 1) + 10.886272036407019994,
        x[0] - x[1] + x[2] - x[3] + x[4] - x[5] + x[6] - x[7] + x[8] - x[9] - 3.1361904761904761904,
        2 * math.cos(-x[8] + x[3]) + x[4] / (x[2] + x[0]) - math.sin(x[1] ** 2) + math.cos(
            x[6] * x[9]) ** 2 - x[7] - 0.1707472705022304757,
        math.sin(x[4]) + 2 * x[7] * (x[2] + x[0]) - math.exp(-x[6] * (-x[9] + x[5])) + 2 * math.cos(x[1]) - 1.0 / (
                    -x[8] + x[3]) - 0.3685896273101277862,
        math.exp(x[0] - x[3] - x[8]) + x[4] ** 2 / x[7] + math.cos(3 * x[9] * x[1]) / 2 - x[5] * x[2] + 2.0491086016771875115,
        x[1] ** 3 * x[6] - math.sin(x[9] / x[4] + x[7]) + (x[0] - x[5]) * math.cos(x[3]) + x[2] - 0.7380430076202798014,
        x[4] * (x[0] - 2 * x[5]) ** 2 - 2 * math.sin(-x[8] + x[2]) + 0.15e1 * x[3] - math.exp(
            x[1] * x[6] + x[9]) + 3.5668321989693809040,
        7 / x[5] + math.exp(x[4] + x[3]) - 2 * x[1] * x[7] * x[9] * x[6] + 3 * x[8] - 3 * x[0] - 8.4394734508383257499,
        x[9] * x[0] + x[8] * x[1] - x[7] * x[2] + math.sin(x[3] + x[4] + x[5]) * x[6] - 0.78238095238095238096])
    return F


def J(x):
    J = np.mat([[-x[1] * math.sin(x[1] * x[0]), -x[0] * math.sin(x[1] * x[0]), 3 * math.exp(-3 * x[2]), x[4] ** 2, 2 * x[3] * x[4],
                    -1, 0, -2 * math.cosh(2 * x[7]) * x[8], -math.sinh(2 * x[7]), 2],
                   [x[1] * math.cos(x[1] * x[0]), x[0] * math.cos(x[1] * x[0]), x[8] * x[6], 0, 6 * x[4],
                    -math.exp(-x[9] + x[5]) - x[7] - 1, x[2] * x[8], -x[5], x[2] * x[6], math.exp(-x[9] + x[5])],
                   [1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
                   [-x[4] / (x[2] + x[0]) ** 2, -2 * x[1] * math.cos(x[1] ** 2), -x[4] / (x[2] + x[0]) ** 2, -2 * math.sin(-x[8] + x[3]),
                    1.0 / (x[2] + x[0]), 0, -2 * math.cos(x[6] * x[9]) * x[9] * math.sin(x[6] * x[9]), -1,
                    2 * math.sin(-x[8] + x[3]), -2 * math.cos(x[6] * x[9]) * x[6] * math.sin(x[6] * x[9])],
                   [2 * x[7], -2 * math.sin(x[1]), 2 * x[7], 1.0 / (-x[8] + x[3]) ** 2, math.cos(x[4]),
                    x[6] * math.exp(-x[6] * (-x[9] + x[5])), -(x[9] - x[5]) * math.exp(-x[6] * (-x[9] + x[5])), 2 * x[2] + 2 * x[0],
                    -1.0 / (-x[8] + x[3]) ** 2, -x[6] * math.exp(-x[6] * (-x[9] + x[5]))],
                   [math.exp(x[0] - x[3] - x[8]), -1.5 * x[9] * math.sin(3 * x[9] * x[1]), -x[5], -math.exp(x[0] - x[3] - x[8]),
                    2 * x[4] / x[7], -x[2], 0, -x[4] ** 2 / x[7] ** 2, -math.exp(x[0] - x[3] - x[8]),
                    -1.5 * x[1] * math.sin(3 * x[9] * x[1])],
                   [math.cos(x[3]), 3 * x[1] ** 2 * x[6], 1, -(x[0] - x[5]) * math.sin(x[3]),
                    x[9] / x[4] ** 2 * math.cos(x[9] / x[4] + x[7]),
                    -math.cos(x[3]), x[1] ** 3, -math.cos(x[9] / x[4] + x[7]), 0, -1.0 / x[4] * math.cos(x[9] / x[4] + x[7])],
                   [2 * x[4] * (x[0] - 2 * x[5]), -x[6] * math.exp(x[1] * x[6] + x[9]), -2 * math.cos(-x[8] + x[2]), 1.5,
                    (x[0] - 2 * x[5]) ** 2, -4 * x[4] * (x[0] - 2 * x[5]), -x[1] * math.exp(x[1] * x[6] + x[9]), 0,
                    2 * math.cos(-x[8] + x[2]),
                    -math.exp(x[1] * x[6] + x[9])],
                   [-3, -2 * x[7] * x[9] * x[6], 0, math.exp(x[4] + x[3]), math.exp(x[4] + x[3]),
                    -7.0 / x[5] ** 2, -2 * x[1] * x[7] * x[9], -2 * x[1] * x[9] * x[6], 3, -2 * x[1] * x[7] * x[6]],
                   [x[9], x[8], -x[7], math.cos(x[3] + x[4] + x[5]) * x[6], math.cos(x[3] + x[4] + x[5]) * x[6],
                    math.cos(x[3] + x[4] + x[5]) * x[6], math.sin(x[3] + x[4] + x[5]), -x[2], x[1], x[0]]])
    return J


def Newton(x):
    iterations = 0
    operations = 0
    while True:
        L, U, P, Q, operations_dec = LUdecomposition(J(x))
        amendments, operations_solve = LUsolve(L, U, P, Q, -F(x))
        xNew = x + amendments
        iterations += 1
        operations += operations_dec + operations_solve
        if max(abs(amendments)) < e:
            break
        x = xNew
    return xNew, iterations, operations


def Newton_mod(x):
    iterations = 0
    J0 = J(x)
    L, U, P, Q, operations_dec = LUdecomposition(J0)
    operations = operations_dec
    while True:
        amendments, operations_solve = LUsolve(L, U, P, Q, -F(x))
        xNew = x + amendments
        iterations += 1
        operations += operations_solve
        if max(abs(amendments)) < e:
            break
        x = xNew
    return xNew, iterations, operations


def Newton_modk(x, k):
    iterations = 0
    operations = 0
    while True:
        if iterations < k:
            Jk = J(x)
            L, U, P, Q, operations_dec = LUdecomposition(Jk)
            operations += operations_dec
        amendments, operations_solve = LUsolve(L, U, P, Q, -F(x))
        xNew = x + amendments
        iterations += 1
        operations += operations_solve
        if max(abs(amendments)) < e:
            break
        x = xNew
    return xNew, iterations, operations


def Newton_gibrid(x, k):
    iterations = 0
    operations = 0
    while True:
        if iterations % k == 0:
            Jk = J(x)
            L, U, P, Q, operations_dec = LUdecomposition(Jk)
            operations += operations_dec
        amendments, operations_solve = LUsolve(L, U, P, Q, -F(x))
        xNew = x + amendments
        iterations += 1
        operations += operations_solve
        if max(abs(amendments)) < e:
            break
        x = xNew
    return xNew, iterations, operations


x0 = [0.5, 0.5, 1.5, -1.0, -0.5, 1.5, 0.5, -0.5, 1.5, -1.5]
e = pow(10, -4)

start = timer()
x, iterations, operations = Newton(x0)
end = timer()
print('Метод Ньютона:')
print('Решение:', x)
print('Кол-во итераций:', iterations)
print('Кол-во операций:', operations)
print('Время выполнения:', end - start)
print()

start = timer()
x, iterations, operations = Newton_mod(x0)
end = timer()
print('Модифицированный метод Ньютона:')
print('Решение:', x)
print('Кол-во итераций:', iterations)
print('Кол-во операций:', operations)
print('Время выполнения:', end - start)
print()

start = timer()
x, iterations, operations = Newton_modk(x0, 3)
end = timer()
print('C переходом к модифицированному:')
print('Решение:', x)
print('Кол-во итераций:', iterations)
print('Кол-во операций:', operations)
print('Время выполнения:', end - start)
print()

start = timer()
x, iterations, operations = Newton_gibrid(x0, 5)
end = timer()
print('Гибридный метод Ньютона:')
print('Решение:', x)
print('Кол-во итераций:', iterations)
print('Кол-во операций:', operations)
print('Время выполнения:', end - start, )
print()

x0new = [0.5, 0.5, 1.5, -1.0, -0.2, 1.5, 0.5, -0.5, 1.5, -1.5]
start = timer()
x, iterations, operations = Newton(x0new)
end = timer()
print('x5 = -0.2:')
print('Решение:', x)
print('Кол-во итераций:', iterations)
print('Кол-во операций:', operations)
print('Время выполнения:', end - start)
