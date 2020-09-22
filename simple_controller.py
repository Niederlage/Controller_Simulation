# import control, slycot
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import time
import control


class Control_Simulation():
    def __init__(self, x0, u0, T, dt):
        self.J = 0.053
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
        self.T_last = 5.
        # self.ids = 0.
        # self.iqs = 5.
        # self.ns = 200.
        # omega0 = self.ns * 2 * np.pi / 60
        # self.A = np.array([[self.R0 / self.Ld, -omega0 * self.Lq / self.Ld, -self.Lq * self.iqs],
        #                    [omega0 * self.Ld / self.Lq, self.R0 / self.Lq, - self.Psi_PM / self.Lq],
        #                    [0, self.K2, 0.]])

        # self.B = np.array([[1 / self.Ld, 0.], [0., 1 / self.Lq], [0., 0.]])
        # self.Q = np.diag([1., 1., 1000.])
        # self.R = np.diag([1., 1.])
        # self.C = np.array([self.c0 / self.J, 0, 0])
        # k = control.acker(self.A,self.B,p0)
        # self.Kp, _, __ = control.lqr(self.A, self.B, self.Q, self.R)

    def motor_model(self, u_vec2, x_vec3):

        did = (u_vec2[0] - x_vec3[0] * self.R0 + self.Lq * x_vec3[2] * x_vec3[1]) / self.Ld
        diq = (u_vec2[1] - x_vec3[1] * self.R0 - self.Ld * x_vec3[2] * x_vec3[0] - self.Psi_PM * x_vec3[2]) / self.Lq
        domega = (self.K2 * x_vec3[1] - self.T_last) / self.J

        x_vec3[0] += did * self.dt
        x_vec3[1] += diq * self.dt
        x_vec3[2] += domega * self.dt

        return x_vec3

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
        state_flow = np.zeros((self.step, 3))
        x_soll_traj = self.soll_trajectory()
        xk = np.copy(self.x0)
        e_i = 0.
        u_traj = []
        for time in np.arange(self.step):
            uk, e_i = self.exakt_linear_controller(xk, x_soll_traj[time], e_i)
            xk = self.motor_model(uk, xk)
            u_traj.append(uk)
            state_flow[time] = xk
            if time > int(0.7 * self.step):
                self.T_last = 0.

        return state_flow, np.array(u_traj)

    def exakt_linear_controller(self, x_mess, x_soll, e_y):
        e_int = e_y + x_mess[2] - x_soll[2]
        u_bar = - 4000 * (x_mess[2] - x_soll[2]) - 682.8427 * self.K2 / self.J * (x_mess[1] - x_soll[1]) - 38.2842 * e_int
        u1 = self.Lq * x_mess[1] * x_mess[2] - 1000 * self.Ld * (x_mess[0] - x_soll[0]) + self.R0 * x_mess[0]
        u2 = self.J * self.Lq / self.K2 * u_bar + self.Ld * x_mess[1] * x_mess[2] + self.Psi_PM * x_mess[2] + self.R0 * \
             x_mess[1]
        if abs(u1) > 24.:
            u1 = np.sign(u1) * 24.
        if abs(u2) > 24.:
            u2 = np.sign(u2) * 24.

        return np.array([u1, u2]), e_int

    def pid_controller(self, x_mess, x_soll, e_vor):
        e_x = x_soll - x_mess
        int_e = e_vor + e_x
        ur = self.Kp.dot(e_x)
        # if np.linalg.norm(ur) > 5:
        #     ur = 5
        return ur, int_e

    def lqr_controller(self, state, ysoll):
        k = np.array([-64.2123327747466, - 15.0104487417976, - 3.16227766016849])
        # S = self.feedforward(k)+ np.dot(S, ysoll)
        ur = -np.dot(k, state)
        return ur

    def soll_trajectory(self):
        ysoll = np.zeros((self.step, 3))
        for step in range(self.step):
            ysoll[step, 0] = 0.
            ysoll[step, 1] = self.T_last / self.K2
            ysoll[step, 2] = 209.

        return ysoll

    def demo_simulation(self):

        y, u = self.system_simulation()
        ysoll = self.soll_trajectory()
        timeline = np.linspace(0, self.T, self.step)
        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        ax1.plot(timeline, y[:, 0], 'r', label='id')
        ax1.plot(timeline, y[:, 1], 'g', label='iq')
        ax1.plot(timeline, y[:, 2], 'b', label='n')

        ax1.plot(timeline, ysoll[:, 0], 'r:', label='id_soll')
        ax1.plot(timeline, ysoll[:, 1], 'g:', label='iq_soll')
        ax1.plot(timeline, ysoll[:, 2], 'b:', label='n_soll')
        # ax1.set_ylim(-5,5)
        legend = ax1.legend(loc='lower right', shadow=True)
        # legend.get_frame().set_facecolor('C0')
        ax1.set_xlabel('t/s')
        ax1.set_ylabel('Amplitude')
        ax1.grid(True)

        ax2 = fig.add_subplot(122)
        ax2.plot(timeline, u[:, 0], label='ud')
        ax2.plot(timeline, u[:, 1], label='uq')
        legend = ax2.legend(loc='lower right', shadow=True)
        # legend.get_frame().set_facecolor('C0')
        ax2.set_xlabel('t/s')
        ax2.set_ylabel('Amplitude')
        ax2.grid(True)
        plt.show()

if __name__ == '__main__':
    xinit = np.array([0., 0., 0.])
    uinit = 0.
    dt0 = 0.001
    Tinit = 10
    control_simu = Control_Simulation(xinit, uinit, Tinit, dt0)
    control_simu.demo_simulation()
