import numpy as np
import math


def swaplines(A, i, j):
    if(i != j):
        C = np.copy(A)
        A[i], A[j] = C[j], C[i]
    return A


def swapcolumns(A, i, j):
    if(i != j):
        C = np.copy(A)
        A[::, i], A[::, j] = C[::, j], C[::, i]
    return A


def LUdecomposition(A):
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
        U = swaplines(U, k, i[0])
        U = swapcolumns(U, k, j[0])
        L = swaplines(L, k, i[0])
        L = swapcolumns(L, k, j[0])
        P = swaplines(P, k, i[0])
        Q = swapcolumns(Q, k, j[0])
        M = np.eye(len(A))
        for l in range(k+1, len(A)):
            L[l][k] = U[l][k] / U[k][k]
            M[l][k] = (-1)*L[l][k]
        U = np.dot(M, U)
    L = L + np.eye(len(A))
    return L, U, P, Q


def solvedown(A, b):
    x = np.zeros(len(A))
    x[0] = b[0]
    for k in range(1, len(A)):
        x[k] = b[k] - np.dot(A[k, :k ],x[:k].T)
    return x


def solveup(A, b):
    x = np.zeros(len(A))
    x[len(A)-1] = b[len(A)-1]/A[len(A)-1, len(A)-1]
    for k in range(len(A)-2, -1, -1):
        x[k] = (b[k] - np.dot(A[k, k+1:],x[k+1:].T))/A[k][k]
    return x


def LUsolve(A, b):
    L, U, P, Q = LUdecomposition(A)
    y = solvedown(L, np.dot(P, b))
    z = solveup(U, y)
    x = np.dot(Q, z)
    return x


def inverse(A):
    e = np.eye(len(A))
    a = np.zeros((len(A), len(A)))
    for i in range(len(A)):
        a[i] = LUsolve(A, e[i]).T
    return a.T


def cond(A):
    a = np.zeros(len(A))
    for i in range(len(A)):
        a[i] = sum(abs(A.T[i]))
    x = np.max(a)
    return x


def rank(A):
    L, U, P, Q = LUdecomposition(A)
    r = len(A)
    u = abs(U)
    while u[r - 1].sum() < pow(10, -14):
        r -= 1
    return r


def compSolve(A, b):
    L, U, P, Q = LUdecomposition(A)
    r = rank(A)
    rExtended = 0
    y = solvedown(L, np.dot(P, b))
    mExtended = np.column_stack((U, y))
    while abs(mExtended[rExtended]).sum() > pow(10, -14):
        rExtended += 1
        if rExtended == len(A):
            break
    if r == rExtended:
        print('Система совместна')
        z = np.zeros(len(A))
        z[r - 1] = y[r - 1] / U[r - 1, r - 1]
        for k in range(r - 2, -1, -1):
            z[k] = (y[k] - np.dot(U[k, k + 1:], z[k + 1:].T)) / U[k][k]
        x = np.dot(Q, z)
        return x
    else:
        return 'Система несовместна'


def QRdecomposition(A):
    Q = np.zeros((len(A), len(A)))
    R = np.zeros((len(A), len(A)))
    a = np.copy(A)
    for k in range(len(A)):
        R[k, k] = math.sqrt(sum(pow(a[::, k], 2)))
        Q[::, k] = a[::, k]/R[k, k]
        for j in range(k+1, len(A)):
            R[k, j] = np.dot(Q[::, k].T, a[::, j])
            a[::, j] = a[::, j] - Q[::, k]*R[k, j]
    return Q, R


def QRsolve(A, b):
    Q, R = QRdecomposition(A)
    x = solveup(R, np.dot(Q.T, b))
    return x


def diagonalPrevalence(n):
    M = np.array(np.random.randint(-20, 20, size=(n, n)), float)
    np.fill_diagonal(M, 0)
    for i in range(n):
        M[i, i] = np.random.randint(1, 20) + sum(abs(M[i]))
    return M


def aprior(normB, d, e):
    normd = max(abs(d))
    return math.ceil((np.log(e) + np.log(1 - normB) - np.log(normd)) / np.log(normB))


def Yakoby(A, b):
    D = np.diag(np.diag(A))
    B = np.eye(len(A)) - np.dot(np.linalg.inv(D), A)
    d = np.dot(np.linalg.inv(D), b)
    normB = max(abs(B[i]).sum() for i in range(len(A)))
    e = pow(10, -9)*(1-normB)/normB
    x1 = d
    x2 = np.dot(B, x1) + d
    aposterior = 1
    while max(abs(x2-x1)) > e:
        x1 = x2
        x2 = np.dot(B, x2) + d
        aposterior += 1
    return x2, aprior(normB, d, e), aposterior


def Zeidel(A, b):
    D = np.diag(np.diag(A))
    U = np.zeros((len(A), len(A)))
    L = np.zeros((len(A), len(A)))
    for i in range(len(A)-1):
        U[i, i+1::] = A[i, i+1::]
        L[i+1, :i+1] = A[i+1, :i+1]
    B = (-1)*np.dot(np.linalg.inv(L+D), U)
    d = np.dot(np.linalg.inv(L+D), b)
    normB = max(abs(B[i]).sum() for i in range(len(A)))
    e = pow(10, -9) * (1 - normB) / normB
    x1 = d
    x2 = np.dot(B, x) + d
    aposterior = 1
    while max(abs(x2-x1)) > e:
        x1 = x2
        x2 = np.dot(B, x2) + d
        aposterior += 1
    return x2, aprior(normB, d, e), aposterior


n = 4
A = np.array(np.random.randint(-20, 20, size=(n, n)), float)
b = np.array(np.random.randint(-20, 20, n), float)
print('A:')
print(A)
print('b:')
print(b)
L, U, P, Q = LUdecomposition(A)
print('L:')
print(L)
print('U:')
print(U)
print('P:')
print(P)
print('Q:')
print(Q)
print('LU:')
print(np.dot(L, U))
print('PAQ:')
print(np.dot(np.dot(P, A), Q))
print('Определитель:', U.diagonal().prod())
x = LUsolve(A, b)
print('x:', x)
print('Ax - b:', np.dot(A, x.T)-b)
print('A^-1:')
print(inverse(A))
print('A*A^-1:')
print(np.dot(A, inverse(A)))
print('A*A^-1:')
print(np.dot(inverse(A), A))
print('Число обусловленности:',cond(A)*cond(inverse(A)))
C = np.array([[1, 3, 5],
              [1, -2, 3],
              [2, 11, 12]], float)
f = np.array([1, 2, 4], float)
print('A:')
print(C)
print('b:')
print(f)
print('rank A:', rank(C))
print(compSolve(C, f))
D = np.array([[1, 2, 3, 4],
              [2, 4, 6, 8],
              [3, 5, 9, 7],
              [1, 5, 3, 4]], float)
e = np.array([1, 2, 4, 1], float)
print('A:')
print(D)
print('b:')
print(e)
print('rank A:', rank(D))
print(compSolve(D, e))
Q, R = QRdecomposition(A)
print('Q:')
print(Q)
print('R:')
print(R)
print('QR:')
print(np.dot(Q, R))
y = QRsolve(A, b)
print('x:', y)
print('Ax - b:', np.dot(A, y.T)-b)
print('x(LU) - x(QR):', x-y)
DP = diagonalPrevalence(n)
g = np.array(np.random.randint(-20, 20, n), float)
print('A:')
print(DP)
print('b:', g)
print('Метод Якоби')
xYa, apriorYa, aposteriorYa = Yakoby(DP, g)
print('x:', xYa)
print('Ax - b:', np.dot(DP, xYa) - g)
print('Априорная оценка:', apriorYa)
print('Апостериорная оценка:', aposteriorYa)
print('Метод Зейделя')
xZ, apriorZ, aposteriorZ = Zeidel(DP, g)
print('x:', xZ)
print('Ax - b:', np.dot(DP, xZ) - g)
print('Априорная оценка:', apriorZ)
print('Апостериорная оценка:', aposteriorZ)
