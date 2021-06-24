from casadi import *

x = SX.sym('x'); y = SX.sym('y')
qp = {'x':vertcat(x,y), 'f':x**2+y**2, 'g':x+y-10}
S = qpsol('S', 'qpoases', qp)
print(S)
r = S(lbg=0)
x_opt = r['x']
print('x_opt: ', x_opt)