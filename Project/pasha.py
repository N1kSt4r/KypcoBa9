from math import inf, log

def f(x, u):
    return (u/x)*log(u/x)

y = [1]
h = 0.1
x = 1
for i in range(1, 4):
    phi_1 = h*f(x + (i-1)*h, y[i-1])
    phi_2 = h*f(x + (i-1)*h + h/2, y[i-1] + phi_1/2)
    phi_3 = h*f(x + (i-1)*h + h/2, y[i-1] + phi_2/2)
    phi_4 = h*f(x + i*h, y[i-1] + phi_3)
    y.append(y[i-1] + (phi_1 + 2*phi_2 + 2*phi_3 + phi_4)/6)
for i in range(3, 10):
    y.append(y[i] + (h/24)*(55*f(x + h*(i-1), y[i-1]) - 59*f(x + h*(i-2), y[i-2]) +
                     37*f(x + h*(i-3), y[i-3]) - 9*f(x + h*(i-4), y[i-4])))
for ind, u in enumerate(y):
    print(f'u({round(x + ind*h, 2)}) = {y[ind]}')