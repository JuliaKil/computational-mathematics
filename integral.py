import math
import numpy as np


def f(x):
    return 4*math.cos(0.5*x)*math.exp(-5*x/4) + 2*math.sin(4.5*x)*math.exp(x/8) + 2


def M0(x):
    return -6*pow(2.2 - x, 1/6)


def M1(x):
    return -6*pow(2.2 - x, 1/6)*(5*x + 66)/35


def M2(x):
    return -6*pow(2.2 - x, 1/6)*(175*pow(x, 2) + 660*x + 8712)/2275


def M3(x):
    return -6*pow(2.2 - x, 1/6)*(11375*pow(x, 3) + 34650*pow(x, 2) + 130680*x + 1724976)/216125


def M4(x):
    return -6*pow(2.2 - x, 1/6)*(1080625*pow(x, 4) + 3003000*pow(x, 3) + 9147600*pow(x, 2) + 34499520*x + 455393664)/27015625


def M5(x):
    return -6*pow(2.2 - x, 1/6)*(27015625*pow(x, 5) + 71321250*pow(x, 4) + 198198000*pow(x, 3) + 603741600*pow(x, 2) + 2276968320*x + 30055981824)/837484375


def IKF(a, b):
    x1, x3 = a, b
    x2 = (a+b)/2
    m0, m1, m2 = M0(x3) - M0(x1), M1(x3) - M1(x1), M2(x3) - M2(x1)
    A1, A2, A3 = np.linalg.solve([[1, 1, 1], [x1, x2, x3], [pow(x1, 2), pow(x2, 2), pow(x3, 2)]], [m0, m1, m2])
    return A1 * f(x1) + A2 * f(x2) + A3 * f(x3)


def SKF(a, b, L, e, formulatype, kopt = 1):
    k = 1*kopt
    while True:
        s = 0
        h = (b-a) / k
        for i in range(int(k)):
            s += formulatype(a + h * i, a + h * (i + 1))
        if k == 1*kopt:
            S = np.array([s])
            H = np.array([h])
        else:
            S = np.append(S, s)
            H = np.append(H, h)
        m = 3
        # примерное значение, если недостаточно сеток для расчета
        r = len(H) - 1
        print('Шаг №', r + 1)
        print("Значение:", S[r])
        if len(H) >= 3:
            m = -math.log((S[r] - S[r-1]) / (S[r-1] - S[r-2])) / math.log(L)
            print('Скорость сходимости:', m)
        H2 = np.empty([r + 1, r + 1])
        for i in range(r):
            H2[::, i] = pow(H, i + m)
        H2[::, r] = -1
        C = np.linalg.solve(H2, -S)
        if r == 0:
            # если недостаточно сеток для расчета погрешности
            error = S[r] - exact_integral
        else:
            error = S[r] - C[r]
            print('J(f):', C[r])
            print('Погрешность методом Ричардсона:', error)
        if error < e:
            break
        k *= L
    return s, S


def opt(a, b, L, e, S):
    m = -math.log((S[2] - S[1]) / (S[1] - S[0])) / math.log(L)
    h2 = (b-a)/pow(L, 2)
    Rh2 = (S[2] - S[1])/(pow(L, m)-1)
    hopt = 0.95*h2*pow(e/abs(Rh2), 1/m)
    return math.ceil((b-a)/hopt)


def KFG(a, b):
    m0, m1, m2, m3, m4, m5 = M0(b) - M0(a), M1(b) - M1(a), M2(b) - M2(a), M3(b) - M3(a), M4(b) - M4(a), M5(b) - M5(a)
    b = np.array([m3, m4, m5])
    A = np. array([[m0, m1, m2],
                   [m1, m2, m3],
                   [m2, m3, m4]])
    a = np.empty(len(b)+1)
    a[0] = 1
    a[:0:-1] = np.linalg.solve(A, -b)
    x = np.roots(a)
    A1, A2, A3 = np.linalg.solve([[1, 1, 1], [x[0], x[1], x[2]], [pow(x[0], 2), pow(x[1], 2), pow(x[2], 2)]], [m0, m1, m2])
    return A1 * f(x[0]) + A2 * f(x[1]) + A3 * f(x[2])


a, b = 1.3, 2.2
integralIKF = IKF(a, b)
print("Интерполяционная квадратурная формула:", integralIKF)

Mn = 240
abspxwx = 0.06569040766966083
exact_integral = 10.83954510946909397740794566485262705081
metod_eval = Mn/math.factorial(3)*abspxwx
exact_eval = integralIKF - exact_integral
print("Оценка методической погрешности:", metod_eval)
print("Точная погрешность:", exact_eval)

L = 2
e = pow(10, -6)
print("Составная квадратурная формула")
integralSKF, S = SKF(a, b, L, e, IKF)

print("Расчет с оптимального шага")
kopt = opt(a, b, L, e, S)
integralSKF, S = SKF(a, b, L, e, IKF, kopt)

print("КФ Гаусса:", KFG(a, b))
print("Вариант Гаусса")
integralSKF, S = SKF(a, b, L, e, KFG)









