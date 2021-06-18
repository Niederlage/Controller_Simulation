import time
import math
import numpy as np
import yaml
from mpc_motion_plot import UTurnMPC
from casadi_differ_reference_line import CasADi_MPC_differ
from gears.cubic_spline_planner import Spline2D
from controller.lqr_speed_steer_control import LQR_Controller
import matplotlib.pyplot as plt


class MPC_LQR_Cpntroller:

    def __init__(self):
        self.LF = 2.  # distance from rear to vehicle front end
        self.LB = 2.  # distance from rear to vehicle back end
        self.W = 2.

        self.dt = 0.1
        self.local_horizon = 2

        self.state_num = 5
        self.GOAL_DIS = 1.5  # goal distance
        self.STOP_SPEED = 0.5 / 3.6  # stop speed
        self.MAX_TIME = 500.0  # max simulation time

        self.v_max = 2.0
        self.max_omega = np.deg2rad(45.0)  # maximum steering angle[rad]
        self.ds = 0.8 * self.dt * self.v_max

        # iterative paramter
        self.MAX_ITER = 3  # Max iteration
        self.DU_TH = 0.1  # iteration finish param

        self.TARGET_SPEED = 10.0 / 3.6  # [m/s] target speed
        self.N_IND_SEARCH = 10  # Search index number
        self.lqr = LQR_Controller()
        self.show_animation = True

    class State:
        """
        vehicle state class
        """

        def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0, omega=0.):
            self.x = x
            self.y = y
            self.yaw = yaw
            self.v = v
            self.omega = omega
            self.predelta = None

    def expand_path(self, refpath, ds):
        x = refpath[0, :]
        y = refpath[1, :]
        sp = Spline2D(x, y)
        s = np.arange(0, sp.s[-1], ds)

        rx, ry, ryaw, rk = [], [], [], []

        for i_s in s:
            ix, iy = sp.calc_position(i_s)
            rx.append(ix)
            ry.append(iy)
            yaw_ = sp.calc_yaw(i_s)
            ryaw.append(yaw_)
            # yaw_last = yaw_
            # rk.append(sp.calc_curvature(i_s))
        return np.array([rx, ry, ryaw])

    def normalize_angle(self, yaw):
        return (yaw + np.pi) % (2 * np.pi) - np.pi

    def coordinate_transform(self, yaw, t, path, mode):
        rot = np.array([[np.cos(yaw), -np.sin(yaw)],
                        [np.sin(yaw), np.cos(yaw)]])
        trans = np.repeat(t[:, None], len(path.T), axis=1)

        if mode == "body to world":
            newpath = rot @ path[:2, :] + trans
            newyaw = path[2, :] + yaw
        else:
            newpath = rot.T @ (path[:2, :] - trans)
            newyaw = path[2, :] - yaw

        newyaw = [self.normalize_angle(i) for i in newyaw]

        return np.vstack((newpath, newyaw))

    def calc_nearest_index(self, state, global_path, pind):
        cx = global_path[0, :]
        cy = global_path[1, :]
        cyaw = global_path[2, :]

        dx = [state.x - icx for icx in cx[pind:(pind + self.N_IND_SEARCH)]]
        dy = [state.y - icy for icy in cy[pind:(pind + self.N_IND_SEARCH)]]

        d = [idx ** 2 + idy ** 2 for (idx, idy) in zip(dx, dy)]

        mind = min(d)

        ind = d.index(mind) + pind

        mind = np.sqrt(mind)

        dxl = cx[ind] - state.x
        dyl = cy[ind] - state.y

        angle = self.normalize_angle(cyaw[ind] - math.atan2(dyl, dxl))
        if angle < 0:
            mind *= -1

        return ind, mind

    def update_state(self, state, u_v, u_o):
        if u_o >= self.max_omega:
            u_o = self.max_omega
        if u_o <= - self.max_omega:
            u_o = - self.max_omega

        if u_v >= self.v_max:
            u_v = self.v_max
        if u_o <= - self.v_max:
            u_v = - self.v_max

        noise = np.random.random(5) * 0.1

        state.v = u_v
        state.omega = u_o
        state.x = state.x + state.v * np.cos(state.yaw) * self.dt + noise[0] * 0
        state.y = state.y + state.v * np.sin(state.yaw) * self.dt + noise[1] * 0
        state.yaw = self.normalize_angle(state.yaw + state.omega * self.dt) + noise[2] * 0.

        return state

    def local_planner(self, local_horizon, dt, x0, original_path):
        # x0_ext = np.repeat(x0[:, None], 3, axis=1)
        # x0_ = self.coordinate_transform(x0[2], x0[:2], x0_ext, mode="world to body")
        # x0_b = np.copy(x0)
        # x0_b[:3] = x0_[:3, 0]

        # ref_path = self.coordinate_transform(original_path[2, 0], original_path[:2, 0], original_path,
        #                                      mode="world to body")

        # states: (x ,y ,theta ,v , steer, a, steer_rate, jerk)
        cmpc = CasADi_MPC_differ()
        cmpc.dt0 = dt
        # ref_traj = self.expand_path(ref_path, self.ds)
        cmpc.horizon = int(local_horizon / dt)
        cmpc.v_end = cmpc.v_max
        cmpc.omega_end = cmpc.omega_max
        # print("mpc start:{}, {}".format(x0[0], x0[1]))
        cmpc.init_model_reference_line(x0, original_path)
        # cmpc.init_model_reference_line(x0_b, ref_path)
        op_traj, op_control = cmpc.get_result_reference_line()

        # print("ds:", dt * cmpc.v_max, " horizon after expasion:", len(ref_traj.T))
        # print("MPC total time:{:.3f}s".format(time.time() - start_time))
        # print("[b] smooth start:{}, {}".format(op_traj[0, 0], op_traj[1, 0]))
        # op_path = self.coordinate_transform(original_path[2, 0], original_path[:2, 0], op_traj,
        #                                     mode="body to world")
        # print("[w] smooth start:{}, {}".format(op_path[0, 0], op_path[1, 0]))
        # return op_path, op_control
        return op_traj, op_control

    def check_goal(self, state, goal, tind, nind):
        # check goal
        dx = state.x - goal[0]
        dy = state.y - goal[1]
        d = np.hypot(dx, dy)

        isgoal = (d <= self.GOAL_DIS)

        if abs(tind - nind) >= 5:
            isgoal = False

        isstop = (abs(state.v) <= self.STOP_SPEED)

        if isgoal and isstop:
            return True

        return False

    def calc_ref_trajectory(self, state, global_path, pind):
        window = int(self.v_max * self.local_horizon / self.dt) + 1
        xref = np.zeros((3, window))
        ncourse = len(global_path.T)

        ind, _ = self.calc_nearest_index(state, global_path, pind)

        if pind >= ind:
            ind = pind

        xref[0, 0] = global_path[0, ind]
        xref[1, 0] = global_path[1, ind]
        xref[2, 0] = global_path[2, ind]

        travel = 0.0

        for i in range(window):
            # travel += abs(state.v) * self.dt
            # travel += window
            # dind = int(round(travel / self.ds))
            dind = int(window)
            if (ind + dind) < ncourse:
                xref[0, i] = global_path[0, i + ind]
                xref[1, i] = global_path[1, i + ind]
                xref[2, i] = global_path[2, i + ind]

            else:
                xref[0, i] = global_path[0, ncourse - 1]
                xref[1, i] = global_path[1, ncourse - 1]
                xref[2, i] = global_path[2, ncourse - 1]

        # xref = self.expand_path(xref, self.ds)
        return xref, ind

    def controller(self, state, op_inputs, local_path, e, e_th):
        # if target_ind > len(op_inputs.T - self.ind_forward):
        #     print("reach input profile end!")
        #     break
        cx = local_path[0, :]
        cy = local_path[1, :]
        cyaw = local_path[2, :]
        u_v, u_o, target_ind, e, e_th = self.lqr.lqr_speed_steering_control(state, cx, cy, cyaw, op_inputs, e, e_th)

        state = self.update_state(state, u_v, u_o)

        if abs(state.v) <= self.STOP_SPEED:
            target_ind += 1

        return state, e, e_th

    def do_simulation(self, global_path, initial_state):
        """
        Simulation

        cx: course x position list
        cy: course y position list
        cy: course yaw position list
        ck: course curvature list
        sp: speed profile
        dl: course tick [m]

        """

        goal = global_path[:2, -1]

        state = initial_state

        time = 0.0
        x = [state.x]
        y = [state.y]
        yaw = [state.yaw]
        v = [state.v]
        omega = [state.omega]
        t = [0.0]
        e, e_th = 0.0, 0.0
        elist = [0.]
        ethlist = [0.]

        target_ind, _ = self.calc_nearest_index(state, global_path, 0)
        # cyaw = smooth_yaw(cyaw)

        while self.MAX_TIME >= time:

            xref, target_ind = self.calc_ref_trajectory(state, global_path, target_ind)

            x0 = np.array([state.x, state.y, state.yaw, state.v, state.yaw])  # current state
            # print("ref start:{}, {}".format(xref[0, 0], xref[1, 0]))
            print("ref goal:{}, {}".format(xref[0, -1], xref[1, -1]))
            smooth_path, smooth_control = self.local_planner(self.local_horizon, self.dt, x0, xref)
            print("smooth goal:{}, {}".format(smooth_path[0, -1], smooth_path[1, -1]))
            if smooth_control is not None:
                v_, omega_ = smooth_control[0, 1], smooth_control[1, 1]

            for k in range(len(smooth_path)):
                state, e, e_th = self.controller(state, smooth_control, smooth_path, e, e_th)
                # state = self.update_state(state, v_, omega_)
                time = time + self.dt

                x.append(state.x)
                y.append(state.y)
                yaw.append(state.yaw)
                v.append(state.v)
                omega.append(state.omega)
                elist.append(e)
                ethlist.append(e_th)
                t.append(time)

                if self.check_goal(state, goal, target_ind, len(global_path.T)):
                    print("Goal")
                    break

                if self.show_animation:  # pragma: no cover
                    plt.cla()
                    # for stopping simulation with the esc key.
                    plt.gcf().canvas.mpl_connect('key_release_event',
                                                 lambda event: [exit(0) if event.key == 'escape' else None])
                    if smooth_path is not None:
                        plt.plot(smooth_path[0, :], smooth_path[1, :], "xc", label="MPC smooth")

                    plt.plot(global_path[0, :], global_path[1, :], color="orange", label="course")
                    plt.plot(x, y, "xr", label="realtime traj")
                    plt.plot(xref[0, :], xref[1, :], color="purple", label="xref")
                    plt.plot(global_path[0, target_ind], global_path[1, target_ind], "xg", label="target")
                    self.plot_car(state)
                    plt.axis("equal")
                    plt.grid(True)
                    plt.title("Time[s]:{:.2f}, speed[m/s]:{:.2f}".format(time, state.v))
                    # f = plt.figure()
                    # self.plot_info(x, y, yaw, v, omega, elist, ethlist)
                    if k % 10 == 0:
                        plt.pause(0.001)

        real_traj = np.array([x, y, yaw, v, omega])
        return t, real_traj

    def plot_arrow(self, x, y, yaw, length=1.5, width=0.5):  # pragma: no cover
        plt.arrow(x, y, length * np.cos(yaw), length * np.sin(yaw),
                  head_length=width, head_width=width)
        plt.plot(x, y)

    def plot_car(self, state):

        outline = np.array([
            [self.LF, self.LF, -self.LB, -self.LB, self.LF],
            [self.W / 2, -self.W / 2, - self.W / 2, self.W / 2, self.W / 2]])

        Rot1 = np.array([[np.cos(state.yaw), -np.sin(state.yaw)],
                         [np.sin(state.yaw), np.cos(state.yaw)]])
        outline = Rot1 @ outline
        outline[0, :] += state.x
        outline[1, :] += state.y
        plt.plot(np.array(outline[0, :]).flatten(),
                 np.array(outline[1, :]).flatten(), "-", color='cadetblue')

        self.plot_arrow(state.x, state.y, state.yaw)

    def plot_info(self, x, y, yaw, v, omega, elist, ethlist):
        plt.cla()
        # s_lqr = np.linspace(0, s[len(op_path[0, :]) - 1], len(yaw))

        # f2 = plt.figure()
        # ax = plt.subplot(211)
        # ax.plot(ref_yawlist, "-g", label="yaw")
        # ax.plot(np.rad2deg(op_path[2, :]), color="orange", label="mpc yaw")
        # ax.plot(np.rad2deg(op_input[1, :]), color="purple", label="mpc omega")
        # ax.grid(True)
        # ax.legend()
        # ax.set_ylabel("ref yaw angle[deg]")
        ax = plt.subplot(211)
        ax.plot(np.rad2deg(yaw), "-r", label="lqr yaw")
        ax.plot(np.rad2deg(omega), "-c", label="lqr omega")
        ax.plot(np.rad2deg(ethlist), color="tab:purple", label="heading error")
        ax.grid(True)
        ax.legend()
        ax.set_xlabel("line length[m]")
        ax.set_ylabel("lqr yaw angle[deg]")

        # f = plt.figure()
        # ax = plt.subplot(111)
        # ax.plot(wx, wy, "xb", label="waypoints")
        # ax.plot(cx, cy, "-r", label="target course")
        # ax.plot(x, y, "-g", label="tracking")
        # ax.grid(True)
        # ax.axis("equal")
        # ax.set_xlabel("x[m]")
        # ax.set_ylabel("y[m]")
        # ax.legend()

        # f3 = plt.figure()
        # ax = plt.subplot(211)
        # # ax.plot(op_input[0, :], "-g", label="mpc v")
        # ax.legend()
        # ax.grid(True)
        # ax.set_ylabel("speed [m/s]")
        ax = plt.subplot(212)
        ax.plot(v, "-r", label="lqr v")
        ax.plot(elist, color="tab:purple", label="lateral error")
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


def get_raw_course(mpc, dl):
    ax = [0.0, 30.0, 26.0]
    ay = [0.0, 0.0, 20.0]
    control_points = np.array([ax, ay])

    return mpc.expand_path(control_points, ds=dl)


def main():
    address = "../../config_differ_smoother.yaml"

    load_file = False
    dl = 0.5
    dt = 0.1  # [s]
    local_horizon = 5  # [s]
    address = "../../config_differ_smoother.yaml"
    with open(address, 'r', encoding='utf-8') as f:
        param = yaml.load(f)
    ut = UTurnMPC()
    ut.set_parameters(param)
    mpclqr = MPC_LQR_Cpntroller()

    if not load_file:
        raw_course = get_raw_course(mpclqr, dl)
        global_path = mpclqr.expand_path(raw_course, 0.8 * dt * 2.)
        x0_ = global_path[0, 0]
        y0_ = global_path[1, 0]
        yaw0_ = global_path[2, 0]
        v0_ = 0.
        omega0_ = 0.

        state0 = mpclqr.State(x=x0_, y=y0_, yaw=yaw0_, v=v0_, omega=omega0_)

        t, real_traj = mpclqr.do_simulation(global_path, state0)

        # np.savez("../../data/smoothed_traj_differ", dt=dt, traj=op_path, control=op_input,
        #          refpath=raw_course)

        # ut.plot_results(dt, local_horizon, op_path, op_input, raw_course)
        # ut.show_plot()

    # else:
    #     loads = np.load("../../data/smoothed_traj_differ.npz")
    #     op_dt = loads["dt"]
    #     op_trajectories = loads["traj"]
    #     op_controls = loads["control"]
    #     ref_traj = loads["refpath"]
    #
    #     print("load file to check!")
    #     ut = UTurnMPC()
    #     with open(address, 'r', encoding='utf-8') as f:
    #         param = yaml.load(f)
    #
    #     ut.set_parameters(param)
    #     ut.reserve_footprint = True
    #     ut.plot_results(op_dt, op_trajectories, op_controls, ref_traj, four_states=True)
    #     ut.show_plot()


if __name__ == '__main__':
    main()
