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

    def update(self, state, a, delta):
        if delta >= self.max_omega:
            delta = self.max_omega
        if delta <= - self.max_omega:
            delta = - self.max_omega

        state.x = state.x + state.v * math.cos(state.yaw) * self.dt
        state.y = state.y + state.v * math.sin(state.yaw) * self.dt
        # state.yaw = state.yaw + state.v / self.L * math.tan(delta) * self.dt
        state.yaw = self.pi_2_pi(state.yaw + delta * self.dt)
        state.v = state.v + a * self.dt
        state.omega = delta

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

    def lqr_speed_steering_control(self, state, cx, cy, cyaw, omega, pe, pth_e, sp):
        ind, e_dist = self.calc_nearest_index(state, cx, cy, cyaw)
        tv = sp[ind]
        # k = ck[ind]
        v = state.v
        omega = omega[ind]
        th_e = self.pi_2_pi(state.yaw - cyaw[ind])

        # A = [1.0, dt, 0.0, 0.0, 0.0
        #      0.0, 0.0, v, 0.0, 0.0]
        #      0.0, 0.0, 1.0, dt, 0.0]
        #      0.0, 0.0, 0.0, 0.0, 0.0]
        #      0.0, 0.0, 0.0, 0.0, 1.0]
        A = np.zeros((5, 5))
        A[0, 0] = 1.0
        A[0, 1] = self.dt
        A[1, 2] = v
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
        x[4, 0] = v - tv

        # input vector
        # u = [delta, accel]
        # delta: steering angle
        # accel: acceleration
        ustar = -K @ x

        # calc steering input
        # ff = math.atan2(self.L * k, 1)  # feedforward steering angle
        # fb = self.pi_2_pi(ustar[0, 0])  # feedback steering angle
        # delta = ff + fb

        ff = omega  # feedforward omega
        fb = self.pi_2_pi(ustar[0, 0]) * self.dt  # feedback steering angle
        delta = ff + fb
        # calc accel input
        accel = ustar[1, 0]

        return delta, ind, e_dist, th_e, accel

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

    def do_simulation(self, cx, cy, cyaw, v_profile, omega_profile, goal):
        T = 500.0  # max simulation time
        goal_dis = 0.8
        stop_speed = 0.01
        noise = 3 * np.random.random(3) * 0
        # state = State(x=0.0, y=0.0, yaw=0.0, v=0.0, omega=0.)
        state = State(x=cx[0] + noise[0], y=cy[0] + noise[1], yaw=cyaw[0] + noise[2], v=0.0, omega=0.)

        time = 0.0
        x = [state.x]
        y = [state.y]
        yaw = [state.yaw]
        v = [state.v]
        omega = [state.omega]
        t = [0.0]

        e, e_th = 0.0, 0.0

        while T >= time:
            dl, target_ind, e, e_th, ai = self.lqr_speed_steering_control(
                state, cx, cy, cyaw, omega_profile, e, e_th, v_profile)

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

            if target_ind % 10 == 0 and show_animation:
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
                plt.pause(0.0001)

        plt.show()
        return t, x, y, yaw, v, omega

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
    sp = op_input[0, :]
    om = op_input[1, :]
    t, x, y, yaw, v, omega = lqr.do_simulation(cx, cy, cyaw, sp, om, goal)

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
        ax.plot(s[:len(yaw)], np.rad2deg(yaw), "-r", label="lqr yaw")
        ax.plot(s[:len(op_path[2, :])], np.rad2deg(op_path[2, :]), color="orange", label="mpc yaw")
        ax.plot(s[:len(omega)], np.rad2deg(omega), "-c", label="lqr omega")
        ax.plot(s[:len(op_input[1, :])], np.rad2deg(op_input[1, :]), color="purple", label="mpc omega")
        ax.grid(True)
        ax.legend()
        ax.set_xlabel("line length[m]")
        ax.set_ylabel("ref yaw angle[deg]")

        ax = plt.subplot(212)
        ax.plot(s[:len(op_input[1, :])], op_input[0, :], "-g", label="mpc v")
        ax.plot(s[:len(v)], v, "-r", label="lqr v")

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
