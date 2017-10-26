#import sympy
import numpy as np

n = 1
h = 1e-2
T = 10.0


#u = lambda t: 10*np.sin(t*2*np.pi)**2
#u = lambda t: t**2
#u = np.vectorize(u)
#kappa = 1 / epsilon
#b = lambda t, x: -kappa*(x-u(t))

# sympy.diff('10*4*x**2*(x**2-a)', 'x') => 80*x**3 + 80*x*(-a + x**2)
u0 = 5
u1 = 10
u = lambda t: u0 + (u1-u0)*0.5*(1+np.cos(t*2*np.pi))
#u = lambda t: u0
u = np.vectorize(u)

epsilon = 1
h = h * epsilon
r0 = 2

def b(t, x):
    deltaE = u(t)
    k = deltaE / (r0**4)
    a = 2 * r0**2
    return -k*(4*x**3 - 2*a*x)/epsilon

def btrunc(t, x):
    return np.max((-1/h, np.min((1/h, b(t, x)))))


sigma = np.sqrt(2 / epsilon)*mat.eye(n)

x0 = np.mat([r0]).T
y0 = np.mat([-r0]).T
t0 = 0

N = int(np.ceil(T / h))+1
S = 10

X = np.zeros((S, N))
Y = np.zeros((S, N))
B = np.zeros((S, N))
Bhat = np.zeros((S, N))
t = np.zeros((S, N))

for i in range(0, S):
    (X[i, :], Y[i, :], B[i, :], Bhat[i, :], t[i, :]) = sde_coupled(x0, y0, t0, btrunc, sigma, N, h)

with Xdmf3File('data/nonauto', 'w') as f:
    f.add_scalar('X', X)
    f.add_scalar('Y', Y)
    f.add_scalar('B', B)
    f.add_scalar('Bhat', Bhat)
    f.add_scalar('u', u(t))
