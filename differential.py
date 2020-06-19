import numpy as np
import math
import matplotlib.pyplot as plt
# Вариант 6


def exact(x):
    y = np.empty(4)
    y[0] = math.exp(math.sin(x ** 2))
    y[1] = math.exp(B * math.sin(x ** 2))
    y[2] = C * math.sin(x ** 2) + A
    y[3] = math.cos(x ** 2)
    return y


def f(x, y):
    dy = np.empty(4)
    dy[0] = 2*x*pow(y[1], 1/B)*y[3]
    dy[1] = 2*B*x*math.exp(B/C*(y[2] - A))*y[3]
    dy[2] = 2*C*x*y[3]
    dy[3] = -2*x*math.log(y[0])
    return dy


def YAMRK(h, y0, x0, xn):
    k = round((xn - x0) / h)
    x = x0
    y = y0
    a21 = c2
    b2 = 1/(2*c2)
    b1 = 1 - b2
    for i in range(int(k)):
        K1 = f(x, y)
        K2 = f(x + c2*h, y + a21*h*K1)
        y = y + h*(b1*K1 + b2*K2)
        x += h
    return y


def opponent(h, y0, x0, xn):
    k = round((xn - x0) / h)
    y = y0
    x = x0
    for i in range(int(k)):
        K1 = f(x, y)
        K2 = f(x + h/3, y + h/3 * K1)
        K3 = f(x + 2*h/3, y - h/3 * K1 + h * K2)
        K4 = f(x + h, y + h*K1 - h * K2 + h * K3)
        y = y + h*(K1/8 + 3*K2/8 + 3*K3/8 + K4/8)
        x += h
    return y


def plot2k():
    h, normRK, normop = np.empty(7), np.empty(7), np.empty(7)
    for i in range(7):
        h[i] = 0.01 / pow(2, i)
        normRK[i] = np.linalg.norm(YAMRK(h[i], y0, x0, xn) - exact(xn))
        normop[i] = np.linalg.norm(opponent(h[i], y0, x0, xn) - exact(xn))
    plt.subplot(2, 1, 1)
    plt.title('Зависимость нормы точной полной погрешности от длины шага')
    plt.ylabel('ЯМРК')
    plt.xscale('log')
    plt.yscale('log')
    plt.plot(h, h ** 2)
    plt.plot(h, normRK)
    plt.subplot(2, 1, 2)
    plt.ylabel('Метод-оппонент')
    plt.xlabel('h')
    plt.xscale('log')
    plt.yscale('log')
    plt.plot(h, h ** 4)
    plt.plot(h, normop)
    plt.show()


def hopt(tol, method, p):
    R2 = np.linalg.norm((method(h/2, y0, x0, xn) - method(h, y0, x0, xn))/(pow(2, p) - 1))
    hopt = 0.95 * h/2 * pow(tol/R2, 1/p)
    return (xn-x0)/math.ceil((xn-x0)/hopt)


def plothopt(tol, method, p):
    x = x0
    y = y0
    h = hopt(tol, method, p)
    k = round((xn - x0) / h)
    X = np.empty(0)
    norm = np.empty(0)
    for i in range(int(k)):
        y = method(h, y, x, x + h)
        x += h
        X = np.append(X, x)
        norm = np.append(norm, np.linalg.norm(y - exact(x)))
    plt.title('Зависимость нормы точной полной погрешности от x при решении с hopt')
    plt.ylabel('norm')
    plt.xlabel('x')
    plt.plot(X, norm)
    plt.show()


def hfirst(rtol, p):
    tol = rtol * np.linalg.norm(y0) + atol
    delta = pow(1 / max(x0, xn), p + 1) + pow(np.linalg.norm(f(x0, y0)), p + 1)
    h1 = pow(tol / delta, 1 / (p + 1))
    u1 = y0 + h1 * f(x0 + h1 / 2, y0 + h1 / 2 * f(x0, y0))
    delta = pow(1 / max(x0, xn), p + 1) + pow(np.linalg.norm(f(x0 + h1, u1)), p + 1)
    h1new = pow(tol / delta, 1 / (p + 1))
    h = min(h1, h1new)
    return h


def auto(rtol, method, p):
    x = x0
    y = y0
    h = hfirst(rtol, p)
    X, Y1, Y2, Y3, Y4 = np.empty(0), np.empty(0), np.empty(0), np.empty(0), np.empty(0)
    norm = np.empty(0)
    H = np.empty(0)
    n = 0 # кол-во обращений к правой части
    while x < xn:
        H = np.append(H, h)
        tol = rtol * np.linalg.norm(y) + atol
        y1 = method(h, y, x, x + h)
        y2 = method(h/2, y, x, x + h)
        n += p*2*3
        r = np.linalg.norm((y2 - y1)/(1 - pow(2, -p)))
        if r > tol * pow(2, p):
            h = h/2
        elif tol < r:
            x += h
            h = h/2
            y = y2
        elif tol/pow(2, p+1) <= r:
            x += h
            y = y1
        else:
            x += h
            h = 2*h
            y = y1
        if x < xn < x + h:
            h = xn - x
        X = np.append(X, x)
        Y1 = np.append(Y1, y[0])
        Y2 = np.append(Y2, y[1])
        Y3 = np.append(Y3, y[2])
        Y4 = np.append(Y4, y[3])
        norm = np.append(norm, np.linalg.norm(y - exact(x)))
    return X, Y1, Y2, Y3, Y4, H, norm, n


def plotauto(method, p):
    X, Y1, Y2, Y3, Y4, H, norm, n = auto(10**(-6), method, p)
    plt.title('Решение')
    plt.plot(X, Y1, X, Y2, X, Y3, X, Y4)
    plt.show()
    plt.title('Зависимость длины шага от x')
    plt.plot(X, H)
    plt.ylabel('h')
    plt.xlabel('x')
    plt.show()
    plt.title('Зависимость нормы точной полной погрешности от x')
    plt.ylabel('norm')
    plt.xlabel('x')
    plt.plot(X, norm)
    plt.show()


def plotrtol():
    R = np.array([10**(-4), 10**(-5), 10**(-6), 10**(-7), 10**(-8)])
    NRK, Nop = np.empty(5), np.empty(5)
    for i in range(5):
        X, Y1, Y2, Y3, Y4, H, norm, nRK = auto(R[i], YAMRK, 2)
        X, Y1, Y2, Y3, Y4, H, norm, nop = auto(R[i], opponent, 4)
        NRK[i], Nop[i] = nRK, nop
    plt.subplot(2, 1, 1)
    plt.title('Зависимость числа обращений к правой части от rtol')
    plt.ylabel('ЯМРК')
    plt.xscale('log')
    plt.yscale('log')
    plt.plot(R, NRK)
    plt.subplot(2, 1, 2)
    plt.ylabel('Метод-оппонент')
    plt.xlabel('rtol')
    plt.xscale('log')
    plt.yscale('log')
    plt.plot(R, Nop)
    plt.show()


c2 = 0.3
A, B, C = -1, -3, 3
x0, xn = 0, 5
y0 = np.array([1, 1, A, 1])
h = 0.000625
print('Точное решение задачи:', exact(xn))
print('ЯMРК:', YAMRK(h, y0, x0, xn))
print('Метод-оппонент:', opponent(h, y0, x0, xn))
plot2k()
print('Решение с hopt')
tol = pow(10, -5)
print('ЯМРК...')
plothopt(tol, YAMRK, 2)
print('Метод-оппонент...')
plothopt(tol, opponent, 4)
print('Автоматический выбор шага')
atol = pow(10, -12)
print('ЯМРК...')
plotauto(YAMRK, 2)
print('Метод-оппонент...')
plotauto(opponent, 4)
print('Число обращений к правой части...')
plotrtol()
