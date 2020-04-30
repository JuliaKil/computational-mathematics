from math import sin, sqrt


def mysin(x):
    k = 1
    a = x
    res = a
    while abs(a) > 0.000001/(2*1.2):
        a *= ((-1)*x**2)/((2*k)*(2*k+1))
        res += a
        k += 1
    return res


def mysqrt(x):
    xn1, xn = 1, 0
    while abs(xn1-xn) > 0.000001/(2*1.44):
        xn = xn1
        xn1 = (xn + x/xn)/2
    return xn1


x = 0.1
print('x', 'exact', 'appr', 'df', sep='             ')
for i in range(11):
    exactf = sin(4.5*x + 0.6)/sqrt(1 + x + 12*x ** 2)
    apprf = mysin(4.5 * x + 0.6) / mysqrt(1 + x + 12 * x ** 2)
    print('{0:.2f}'.format(x), exactf, apprf, exactf - apprf, sep=' ')
    x += 0.01
