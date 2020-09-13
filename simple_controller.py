# import control, slycot
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import threading
import time
from queue import Queue
import control


class Control_Simulation():
    def __init__(self, x0, u0, T, dt):
        self.J = 0.023
        self.r0 = 0.0088
        self.c0 = 0.7035
        self.x0 = x0
        self.u0 = u0
        self.Tm = 0.0125
        self.T = T
        self.dt = dt
        self.step = int(T / dt)
        self.Ld = 0.097e-3  # H
        self.Lq = 0.097e-3
        self.L_mx = np.array([[self.Ld, 0], [0, self.Lq]])
        self.L_dmx = np.array([[0, - self.Lq], [self.Ld, 0]])
        self.R0 = 0.194  # Ohm
        self.I = 0.  # A
        self.U = 24.  # V
        self.p = 7
        self.K1 = 1 / 24.48
        self.K2 = 0.3
        self.Psi_PM = 0.02
        self.T_last = 0.2
        self.ids = 0.
        self.iqs = 5.
        self.ns = 200.
        omega0 = self.ns * 2 * np.pi / 60
        self.A = np.array([[self.R0 / self.Ld, -omega0 * self.Lq / self.Ld, -self.Lq * self.iqs],
                           [omega0 * self.Ld / self.Lq, self.R0 / self.Lq, - self.Psi_PM / self.Lq],
                           [0, self.K2, 0.]])

        self.B = np.array([[1 / self.Ld, 0.], [0., 1 / self.Lq], [0., 0.]])
        self.Q = np.diag([1., 1., 1000.])
        self.R = np.diag([1., 1.])
        # self.C = np.array([self.c0 / self.J, 0, 0])
        # k = control.acker(self.A,self.B,p0)
        self.Kp, _, __ = control.lqr(self.A, self.B, self.Q, self.R)
        print(1)

    def motor_model(self, u_vec2, i_vec2, omega):
        Li = self.L_dmx.dot(i_vec2)
        temp = u_vec2 - i_vec2 * self.R0 - (Li + np.array([0, self.Psi_PM])) * omega
        Linv = np.linalg.inv(self.L_mx)
        di = Linv.dot(temp.reshape(-1))
        domega = (self.K2 * i_vec2[1]-self.T_last) / self.J
        i_vec2 += di * self.dt
        omega += domega * self.dt

        return i_vec2, omega

    def feedforward(self, k):
        B_right = self.B.reshape(np.size(self.B), 1)
        k_right = k.reshape(1, np.size(k))
        zwischen = np.dot(B_right, k_right) - self.A
        invzw = np.linalg.inv(zwischen)
        czwischen = np.dot(self.C, invzw)
        i = np.dot(invzw, zwischen)
        invS = np.dot(czwischen, B_right)
        pass
        if np.size(invS) == 1:
            return invS
        else:
            return np.linalg.inv(invS)

    def system_simulation(self):
        # state_flow = np.arange((x_len + u_len) * int(self.T / self.dt)).reshape((x_len + u_len),int(self.T / self.dt))
        state_flow = np.zeros((self.step, 3))
        x_soll_traj = self.soll_trajectory()
        xk = self.x0
        x_error = np.zeros((3,))
        for time in np.arange(self.step):
            uk = self.exakt_linear_controller(xk, x_soll_traj[time])
            state_flow[time, :2], state_flow[time, 2] = self.motor_model(uk, xk[:2], xk[2])
            if time > int(0.5 * self.step):
                self.T_last = 0.

        return state_flow

    def pid_controller(self, x_mess, x_soll, e_vor):
        e_x = x_soll - x_mess
        int_e = e_vor + e_x
        ur = self.Kp.dot(e_x)
        # if np.linalg.norm(ur) > 5:
        #     ur = 5
        return ur, int_e

    def exakt_linear_controller(self, x_mess, x_soll):
        u1 = self.Ld * ( x_mess[1] * x_mess[2] - 5 * x_mess[0]) + self.R0 * x_mess[0]
        u2 = self.J * self.Ld / self.K2 * (0 - 2 * x_mess[2] - 3 *self.K2 / self.J* x_mess[1]) - self.Ld * x_mess[1] * x_mess[2] + self.Psi_PM * x_mess[2] - self.R0*x_mess[1]
        return np.array([u1, u2])

    def lqr_controller(self, state, ysoll):
        k = np.array([-64.2123327747466, - 15.0104487417976, - 3.16227766016849])
        # S = self.feedforward(k)+ np.dot(S, ysoll)
        ur = -np.dot(k, state)
        return ur

    def soll_trajectory(self):
        ysoll = np.zeros((self.step, 3))
        for step in range(self.step):
            ysoll[step, 0] = 0.
            ysoll[step, 1] = 1.
            ysoll[step, 2] = 2
        return ysoll

    def demo_simulation(self):

        y = self.system_simulation()
        ysoll = self.soll_trajectory()
        timeline = np.linspace(0, self.T, self.step)
        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        ax1.plot(timeline, y[:, 0], 'r:', label='id')
        ax1.plot(timeline, y[:, 1], 'g:', label='iq')
        ax1.plot(timeline, y[:, 2], 'b', label='n')
        # ax1.plot(timeline, ysoll, label='ysoll')

        legend = ax1.legend(loc='lower right', shadow=True)
        # legend.get_frame().set_facecolor('C0')
        ax1.set_xlabel('t/s')
        ax1.set_ylabel('Amplitude')
        ax1.grid(True)

        ax2 = fig.add_subplot(122)
        ax2.plot(y[0, :], y[3, :], 'r', label='phase space')
        legend = ax1.legend(loc='lower right', shadow=True)
        # legend.get_frame().set_facecolor('C0')
        ax2.set_xlabel('x1')
        ax2.set_ylabel('x2')
        ax2.grid(True)
        plt.show()


if __name__ == '__main__':
    xinit = np.array([0.1, 0.5, 1.])
    uinit = 0.
    dt0 = 0.01
    Tinit = 5
    control_simu = Control_Simulation(xinit, uinit, Tinit, dt0)
    control_simu.demo_simulation()
