import numpy as np
import math

e = pow(10, -4)


def f(x):
    return math.tan(0.5*x + 0.2) - pow(x, 2)


def deltaF(x):
    return 1/(2*pow(math.cos(0.5*x + 0.2), 2)) - 2*x


xOld = -0.3
while True:
    g = -f(xOld)/deltaF(xOld)
    xNew = xOld + g
    if abs(g) < e:
        break
    xOld = xNew
print('Решение уравнения:', xNew)


def f1(x, y):
    return math.sin(y + 0.5) - x - 1


def f2(x, y):
    return y + math.cos(x - 2)


def f1x():
    return -1


def f2x(x):
    return -math.sin(x-2)


def f1y(y):
    return math.cos(y+0.5)


def f2y():
    return 1


xOld, yOld = -0.2, 0.5
while True:
    J = np.array([[f1x(), f1y(yOld)],
                  [f2x(xOld), f2y()]])
    F = np.array([-f1(xOld, yOld), -f2(xOld, yOld)])
    g, h = np.linalg.solve(J, F)
    xNew, yNew = xOld + g, yOld + h
    if max(abs(g), abs(h)) < e:
        break
    xOld, yOld = xNew, yNew
print('Решение системы:', xNew, yNew)
