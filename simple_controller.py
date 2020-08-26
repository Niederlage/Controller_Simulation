# import control, slycot
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import threading
import time
from queue import Queue

Kp = 1.5
Ki = 12.0
Kd = 1.0

class Control_Simulation():
    def __init__(self, x0, u0, T, dt):
        self.J = 0.0023
        self.r0 = 0.0088
        self.c0= 0.7035
        self.x0 = x0
        self.u0 = u0
        self.Tm = 0.0125
        self.T = T
        self.dt = dt
        self.step = int(T / dt)
        self.A = np.array([[0,1.,0],[0,0,1.],[0,-self.c0/self.J,-self.r0/self.J]])
        self.B = np.array([0,0,1.])[:,None]
        self.C = np.array([self.c0/self.J,0,0])
        # print(self.B)
        p0 = [-2.,-2.,-2.]
        # k = control.acker(self.A,self.B,p0)

        pass

    def get_xdot(self, xk, uk):
        Ax = np.dot(self.A, xk).reshape((3,1))
        uk0 = uk
        if np.size(uk) ==1:
            uk0 = uk[0]
        Bu = np.dot(self.B,uk0)
        return Ax + Bu

    def feedforward(self, k):
        B_right = self.B.reshape(np.size(self.B),1)
        k_right = k.reshape(1, np.size(k))
        zwischen = np.dot(B_right, k_right) - self.A
        invzw = np.linalg.inv(zwischen)
        czwischen = np.dot(self.C,invzw)
        i = np.dot(invzw, zwischen)
        invS = np.dot(czwischen, B_right)
        pass
        if np.size(invS) == 1:
            return invS
        else:
            return np.linalg.inv(invS)

    def system_simulation(self):
        x_len = len(self.x0)
        u_len = 1
        # state_flow = np.arange((x_len + u_len) * int(self.T / self.dt)).reshape((x_len + u_len),int(self.T / self.dt))
        state_flow = np.zeros((x_len + u_len, self.step))
        xk = self.x0
        uk = self.u0
        ysoll = self.soll_trajectory()
        # print(state_flow)
        # print('\n')
        # print(ysoll)
        for time in np.arange(self.step):
            state_flow[:x_len, time] = xk.flatten()
            state_flow[x_len:, time] = uk

            uk = self.lqr_controller(xk, ysoll[time])
            x_dot = self.get_xdot(xk, uk)
            # noise = random.random()
            # if noise > 1e-3:
            #     noise = 1e-3

            pass
            xk += x_dot * self.dt #+ noise

        return state_flow

    def pid_controller(self, ysoll, ymess, ey_vor):
        eyk = ysoll - ymess
        int_ey = ey_vor + eyk
        diff_ey = eyk - ey_vor
        ur = Kp * eyk + Ki * int_ey + Kd * diff_ey
        if abs(ur) > 5:
            ur = 5
        return ur

    def lqr_controller(self, state, ysoll):
        k = np.array([-64.2123327747466,- 15.0104487417976,- 3.16227766016849])
        S = self.feedforward(k)
        ur = -np.dot(k, state) + np.dot(S, ysoll)
        return ur

    def soll_trajectory(self):
        ysoll = np.zeros((1, self.step)).flatten()
        for step in range(self.step):
            ysoll[step] = 0.2

        return ysoll

def demo_simulation(x0, u0, T, dt):
    pid_simu = Control_Simulation(x0, u0, T, dt)
    y = pid_simu.system_simulation()
    timeline = np.arange(0, T, dt)
    ysoll = pid_simu.soll_trajectory()

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.plot(timeline, y[0, :], 'r:', label='x1')
    ax1.plot(timeline, y[1, :], 'g:', label='x2')
    ax1.plot(timeline, y[2, :], 'b', label='ymess')
    ax1.plot(timeline, y[3, :], 'c:', label='x4')
    ax1.plot(timeline, ysoll, label='ysoll')
    legend = ax1.legend(loc='lower right', shadow=True)
    # legend.get_frame().set_facecolor('C0')
    ax1.set_xlabel('t/s')
    ax1.set_ylabel('x1')
    ax1.grid(True)
    plt.show()
xinit = np.array([0.,0.,0.])[:,None]
uinit = 0.
dt0 =0.01
Tinit = 10

demo_simulation(xinit, uinit, Tinit, dt0)
