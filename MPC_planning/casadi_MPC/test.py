import casadi as ca

x = ca.SX.sym("x", 2, 5)
print(ca.sumsqr(x[0, :]))
l = ca.SX.sym("l", 3, 5)
v = ca.vertcat(x, l)
print(v.size())
t = ca.reshape(v.T, -1, 1)
print(t.size()[0])
print(ca.fmin(0, 1))
