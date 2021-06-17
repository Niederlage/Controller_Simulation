"""

Path tracking simulation with LQR speed and steering control

author Atsushi Sakai (@Atsushi_twi)

"""
import math
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la

sys.path.append("../../PathPlanning/CubicSpline/")

try:
    import gears.cubic_spline_planner as csp
except ImportError:
    raise

show_animation = True


class State:

    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0, omega=0.):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.omega = omega


class LQR_Controller:
    def __init__(self):
        # LQR parameter
        self.lqr_Q = np.eye(5)
        self.lqr_R = np.eye(2)
        self.dt = 0.1  # time tick[s]
        self.L = 0.5  # Wheel base of the vehicle [m]
        self.max_v = 2.0
        self.max_omega = np.deg2rad(45.0)  # maximum steering angle[rad]
        self.max_acc = 1.

    def update(self, state, a_, delta_, use_RungeKutta=False):
        if delta_ >= self.max_omega:
            delta_ = self.max_omega
        if delta_ <= - self.max_omega:
            delta_ = - self.max_omega

        if a_ >= self.max_acc:
            a_ = self.max_acc
        if delta_ <= - self.max_acc:
            a_ = - self.max_acc

        if use_RungeKutta:

            k1_dx = state.v * math.cos(state.yaw)
            k1_dy = state.v * math.sin(state.yaw)
            k1_dyaw = state.omega
            k1_dv = a_
            k1_domega = delta_

            k2_dx = (state.v + 0.5 * self.dt * k1_dv) * math.cos(state.yaw + 0.5 * self.dt * k1_dyaw)
            k2_dy = (state.v + 0.5 * self.dt * k1_dv) * math.sin(state.yaw + 0.5 * self.dt * k1_dyaw)
            k2_dyaw = state.omega + 0.5 * self.dt * k1_domega
            k2_dv = a_
            k2_domega = delta_

            k3_dx = (state.v + 0.5 * self.dt * k2_dv) * math.cos(state.yaw + 0.5 * self.dt * k2_dyaw)
            k3_dy = (state.v + 0.5 * self.dt * k2_dv) * math.sin(state.yaw + 0.5 * self.dt * k2_dyaw)
            k3_dyaw = state.omega + 0.5 * self.dt * k2_domega
            k3_dv = a_
            k3_domega = delta_

            k4_dx = (state.v + self.dt * k3_dv) * math.cos(state.yaw + self.dt * k3_dyaw)
            k4_dy = (state.v + self.dt * k3_dv) * math.sin(state.yaw + self.dt * k3_dyaw)
            k4_dyaw = state.omega + self.dt * k3_domega
            k4_dv = a_
            k4_domega = delta_

            dx = self.dt * (k1_dx + 2 * k2_dx + 2 * k3_dx + k4_dx) / 6
            dy = self.dt * (k1_dy + 2 * k2_dy + 2 * k3_dy + k4_dy) / 6
            dyaw = self.dt * (k1_dyaw + 2 * k2_dyaw + 2 * k3_dyaw + k4_dyaw) / 6
            dv = self.dt * (k1_dv + 2 * k2_dv + 2 * k3_dv + k4_dv) / 6
            domega = self.dt * (k1_domega + 2 * k2_domega + 2 * k3_domega + k4_domega) / 6

            # state.yaw = state.yaw + state.v / self.L * math.tan(delta) * self.dt
            state.x += dx
            state.y += dy
            state.yaw = self.pi_2_pi(state.yaw + dyaw)
            state.v += dv
            state.omega += domega

        else:
            noise = np.random.random(5) * 0.1
            state.x = state.x + state.v * math.cos(state.yaw) * self.dt + noise[0] * 0
            state.y = state.y + state.v * math.sin(state.yaw) * self.dt + noise[1] * 0
            # state.yaw = state.yaw + state.v / self.L * math.tan(delta) * self.dt
            state.yaw = self.pi_2_pi(state.yaw + state.omega * self.dt) + noise[2] * 0.1
            state.v = state.v + a_ * self.dt + noise[3] * 0.5
            # state.omega = state.omega + delta_ * self.dt
            state.omega = delta_ + noise[4]

        return state

    def pi_2_pi(self, angle):
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def solve_dare(self, A, B, Q, R):
        """
        solve a discrete time_Algebraic Riccati equation (DARE)
        """
        x = Q
        x_next = Q
        max_iter = 150
        eps = 0.01

        for i in range(max_iter):
            x_next = A.T @ x @ A - A.T @ x @ B @ \
                     la.inv(R + B.T @ x @ B) @ B.T @ x @ A + Q
            if (abs(x_next - x)).max() < eps:
                break
            x = x_next

        return x_next

    def dlqr(self, A, B, Q, R):
        """Solve the discrete time lqr controller.
        x[k+1] = A x[k] + B u[k]
        cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
        # ref Bertsekas, p.151
        """

        # first, try to solve the ricatti equation
        P = self.solve_dare(A, B, Q, R)

        # compute the LQR gain
        K = la.inv(B.T @ P @ B + R) @ (B.T @ P @ A)

        eig_result = la.eig(A - B @ K)

        return K, P, eig_result[0]

    def lqr_speed_steering_control(self, state, cx, cy, cyaw, op_inputs, pe, pth_e):
        ind, e_dist = self.calc_nearest_index(state, cx, cy, cyaw)
        # k = ck[ind]
        if ind > len(op_inputs.T):
            ind = len(op_inputs.T)

        v_forward = op_inputs[0, ind]
        omega_forward = op_inputs[1, ind]
        a_forward = op_inputs[2, ind]
        omega_rate_forward = op_inputs[3, ind]

        v_e = state.v - v_forward
        th_e = self.pi_2_pi(state.yaw - cyaw[ind])
        om_e = self.pi_2_pi(state.omega - omega_forward)
        # A = [1.0, dt, 0.0, 0.0, 0.0
        #      0.0, 0.0, v, 0.0, 0.0]
        #      0.0, 0.0, 1.0, dt, 0.0]
        #      0.0, 0.0, 0.0, 0.0, 0.0]
        #      0.0, 0.0, 0.0, 0.0, 1.0]
        A = np.zeros((5, 5))
        A[0, 0] = 1.0
        A[0, 1] = self.dt
        A[1, 2] = state.v
        A[2, 2] = 1.0
        A[2, 3] = self.dt
        A[4, 4] = 1.0

        # B = [0.0, 0.0
        #     0.0, 0.0
        #     0.0, 0.0
        #     v/L, 0.0
        #     0.0, dt]

        B = np.zeros((5, 2))
        # B[3, 0] = v / self.L
        B[3, 0] = self.dt
        B[4, 1] = self.dt
        Q = self.lqr_Q
        R = self.lqr_R
        K, _, _ = self.dlqr(A, B, Q, R)

        # state vector
        # x = [e, dot_e, th_e, dot_th_e, delta_v]
        # e: lateral distance to the path
        # dot_e: derivative of e
        # th_e: angle difference to the path
        # dot_th_e: derivative of th_e
        # delta_v: difference between current speed and target speed
        x = np.zeros((5, 1))
        x[0, 0] = e_dist
        x[1, 0] = (e_dist - pe) / self.dt
        x[2, 0] = th_e
        x[3, 0] = (th_e - pth_e) / self.dt
        # x[3, 0] = om_e
        x[4, 0] = v_e

        # input vector
        # u = [delta, accel]
        # delta: steering angle
        # accel: acceleration
        ustar = -K @ x

        # calc steering input
        # ff = math.atan2(self.L * k, 1)  # feedforward steering angle
        # fb = self.pi_2_pi(ustar[0, 0])  # feedback steering angle
        # delta = ff + fb

        delta = (ustar[0, 0] + omega_rate_forward) * self.dt + omega_forward
        accel = ustar[1, 0] + a_forward

        return delta, accel, ind, e_dist, th_e,

    def calc_nearest_index(self, state, cx, cy, cyaw):
        dx = [state.x - icx for icx in cx]
        dy = [state.y - icy for icy in cy]

        d = [idx ** 2 + idy ** 2 for (idx, idy) in zip(dx, dy)]

        mind = min(d)

        ind = d.index(mind)

        mind = math.sqrt(mind)

        dxl = cx[ind] - state.x
        dyl = cy[ind] - state.y

        angle = self.pi_2_pi(cyaw[ind] - math.atan2(dyl, dxl))
        if angle < 0:
            mind *= -1

        return ind, mind

    def do_simulation(self, cx, cy, cyaw, op_inputs, goal):
        T = 500.0  # max simulation time
        goal_dis = 1.
        stop_speed = 0.05
        noise = 2 * np.random.random(3)
        # state = State(x=0.0, y=0.0, yaw=0.0, v=0.0, omega=0.)
        state = State(x=cx[0] + noise[0], y=cy[0] + noise[1], yaw=cyaw[0] + noise[2], v=0.0, omega=1.)

        time = 0.0
        x = [state.x]
        y = [state.y]
        yaw = [state.yaw]
        v = [state.v]
        omega = [state.omega]
        t = [0.0]
        elist = [0.]
        e_thlist = [0.]

        e, e_th = 0.0, 0.0
        target_ind = 0
        while T >= time:
            if target_ind > len(op_inputs.T) - 1:
                print("reach input profile end!")
                break
            dl, ai, target_ind, e, e_th = self.lqr_speed_steering_control(state, cx, cy, cyaw, op_inputs, e, e_th)
            state = self.update(state, ai, dl)

            if abs(state.v) <= stop_speed:
                target_ind += 1

            time = time + self.dt

            # check goal
            dx = state.x - goal[0]
            dy = state.y - goal[1]
            if math.hypot(dx, dy) <= goal_dis:
                print("Goal")
                break

            x.append(state.x)
            y.append(state.y)
            yaw.append(state.yaw)
            v.append(state.v)
            omega.append(state.omega)
            t.append(time)
            elist.append(e)
            e_thlist.append(e_th)

            if target_ind % 5 == 0 and show_animation:
                plt.cla()
                # for stopping simulation with the esc key.
                plt.gcf().canvas.mpl_connect('key_release_event',
                                             lambda event: [exit(0) if event.key == 'escape' else None])
                plt.plot(cx, cy, "-r", label="course")
                plt.plot(x, y, "ob", label="trajectory")
                plt.plot(cx[target_ind], cy[target_ind], "xg", label="target")
                plt.axis("equal")
                plt.grid(True)
                plt.title("v[m/s]:" + str(round(state.v, 2))
                          + ", omega[m/s]:" + str(round(state.omega, 3)) + ", target index:" + str(target_ind))
                plt.pause(0.001)

        plt.show()
        return t, x, y, yaw, v, omega, elist, e_thlist

    def calc_speed_profile(self, cyaw, target_speed):
        speed_profile = [target_speed] * len(cyaw)

        direction = 1.0

        # Set stop point
        for i in range(len(cyaw) - 1):
            dyaw = abs(cyaw[i + 1] - cyaw[i])
            switch = math.pi / 4.0 <= dyaw < math.pi / 2.0

            if switch:
                direction *= -1

            if direction != 1.0:
                speed_profile[i] = - target_speed
            else:
                speed_profile[i] = target_speed

            if switch:
                speed_profile[i] = 0.0

        # speed down
        for i in range(40):
            speed_profile[-i] = target_speed / (50 - i)
            if speed_profile[-i] <= 1.0 / 3.6:
                speed_profile[-i] = 1.0 / 3.6

        return speed_profile


def main():
    print("LQR steering control tracking start!!")
    # ax = [0.0, 6.0, 12.5, 10.0, 17.5, 20.0, 25.0]
    # ay = [0.0, -3.0, -5.0, 6.5, 3.0, 0.0, 0.0]
    # goal = [ax[-1], ay[-1]]
    # cx, cy, cyaw, ck, s = csp.calc_spline_course(
    #     ax, ay, ds=0.1)
    # target_speed = 10.0 / 3.6  # simulation parameter km/h -> m/s

    loads = np.load("../data/smoothed_traj_differ.npz")
    dt = loads["dt"]
    op_path = loads["traj"]
    op_input = loads["control"]
    ref_path = loads["refpath"]
    wx = ref_path[0, :]
    wy = ref_path[1, :]
    cx, cy, cyaw, ck, s = csp.calc_spline_course(wx, wy, ds=0.8 * 2 * dt)
    ref_yawlist = [np.rad2deg(cyaw[i]) for i in range(len(op_input[1, :]))]
    goal = [op_path[0, -1], op_path[1, -1]]
    # target_speed = 2.0  # simulation parameter km/h -> m/s
    lqr = LQR_Controller()

    # sp = lqr.calc_speed_profile(cyaw, target_speed)
    # t, x, y, yaw, v, omega = lqr.do_simulation(cx, cy, cyaw, op_input, goal)
    t, x, y, yaw, v, omega, elist, ethlist = lqr.do_simulation(cx, cy, cyaw, op_input, goal)

    if show_animation:  # pragma: no cover
        plt.close()
        f = plt.figure()
        ax = plt.subplot(111)
        ax.plot(wx, wy, "xb", label="waypoints")
        ax.plot(cx, cy, "-r", label="target course")
        ax.plot(x, y, "-g", label="tracking")
        ax.grid(True)
        ax.axis("equal")
        ax.set_xlabel("x[m]")
        ax.set_ylabel("y[m]")
        ax.legend()

        f2 = plt.figure()
        ax = plt.subplot(211)
        ax.plot(s[:len(op_input[1, :])], ref_yawlist, "-g", label="yaw")
        ax.plot(s[:len(op_path[2, :])], np.rad2deg(op_path[2, :]), color="orange", label="mpc yaw")
        ax.plot(s[:len(op_input[1, :])], np.rad2deg(op_input[1, :]), color="purple", label="mpc omega")
        ax.grid(True)
        ax.legend()
        ax.set_ylabel("ref yaw angle[deg]")
        ax = plt.subplot(212)
        ax.plot(s[:len(yaw)], np.rad2deg(yaw), "-r", label="lqr yaw")
        ax.plot(s[:len(omega)], np.rad2deg(omega), "-c", label="lqr omega")
        ax.plot(s[:len(ethlist)], np.rad2deg(ethlist), color="tab:purple", label="heading error")
        ax.grid(True)
        ax.legend()
        ax.set_xlabel("line length[m]")
        ax.set_ylabel("lqr yaw angle[deg]")

        f3 = plt.figure()
        ax = plt.subplot(211)
        ax.plot(s[:len(op_input[1, :])], op_input[0, :], "-g", label="mpc v")
        ax.legend()
        ax.grid(True)
        ax.set_ylabel("speed [m/s]")
        ax = plt.subplot(212)
        ax.plot(s[:len(v)], v, "-r", label="lqr v")
        ax.plot(s[:len(elist)], elist, color="tab:purple", label="lateral error")
        ax.grid(True)
        ax.legend()
        ax.set_xlabel("line length[m]")
        ax.set_ylabel("speed [m/s]")

        # plt.subplots(1)
        # plt.plot(s, ck, "-r", label="curvature")
        # plt.grid(True)
        # plt.legend()
        # plt.xlabel("line length[m]")
        # plt.ylabel("curvature [1/m]")

        plt.show()


if __name__ == '__main__':
    main()
