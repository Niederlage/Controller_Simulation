import time
import math
import numpy as np
import yaml
from motion_plot.differ_motion_plot import UTurnMPC
from casadi_MPC.Modell_Differential.casadi_differ_reference_line import CasADi_MPC_differ
from gears.cubic_spline_planner import Spline2D
from controller.lqr_speed_steer_control import LQR_Controller
from controller.casadi_mpc_controller import CasADi_MPC_differ_Control
import matplotlib.pyplot as plt


class MPC_LQR_Controller:

    def __init__(self):
        self.LF = 2.  # distance from rear to vehicle front end
        self.LB = 2.  # distance from rear to vehicle back end
        self.W = 2.

        self.dt = 0.02
        self.local_horizon = 2
        self.window = int(self.local_horizon / self.dt)
        self.state_num = 5
        self.GOAL_DIS = 5.5  # goal distance
        self.STOP_SPEED = 0.5 / 3.6  # stop speed
        self.MAX_TIME = 500.0  # max simulation time

        self.max_v = 2.0
        self.max_omega = np.deg2rad(120.0)  # maximum steering angle[rad]
        self.max_acc = 2.0
        self.max_omega_rate = np.deg2rad(720.0)  # maximum steering angle[rad]
        self.ds = 0.5 * self.dt * self.max_v

        # iterative paramter
        self.MAX_ITER = 3  # Max iteration
        self.DU_TH = 0.1  # iteration finish param
        self.ref_v = [0.]
        self.ref_omega = [0.]
        self.ref_a = [0.]
        self.ref_omega_rate = [0.]
        self.fb_a = [[0., 0., 0., 0., 0.]]
        self.fb_o_rate = [[0., 0., 0., 0., 0.]]
        self.fb_a = [0.]
        self.fb_o_rate = [0.]

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
            ryaw.append(self.normalize_angle(yaw_))
            rk.append(sp.calc_curvature(i_s))

        return np.array([rx, ry, ryaw, rk])

    def normalize_angle(self, yaw):
        return (yaw + np.pi) % (2 * np.pi) - np.pi

    def coordinate_transform(self, yaw, t, path, mode):
        yaw = self.normalize_angle(yaw)
        yawlist = np.array([self.normalize_angle(i) for i in path[2, :]])

        rot = np.array([[np.cos(yaw), -np.sin(yaw)],
                        [np.sin(yaw), np.cos(yaw)]])
        trans = np.repeat(t[:, None], len(path.T), axis=1)

        if mode == "body to world":
            newpath = rot @ path[:2, :] + trans
            newyaw = yawlist + yaw
        else:
            newpath = rot.T @ (path[:2, :] - trans)
            newyaw = yawlist - yaw

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
        # if u_o >= self.max_omega:
        #     u_o = self.max_omega
        # if u_o <= - self.max_omega:
        #     u_o = - self.max_omega
        #

        if u_o >= self.max_omega_rate:
            u_o = self.max_omega_rate
        if u_o <= - self.max_omega_rate:
            u_o = - self.max_omega_rate

        if u_v >= self.max_acc:
            u_v = self.max_acc
        if u_o <= - self.max_acc:
            u_v = - self.max_acc

        noise = np.random.random(5) * 0.1

        state.x = state.x + state.v * np.cos(state.yaw) * self.dt + noise[1] * 0
        state.y = state.y + state.v * np.sin(state.yaw) * self.dt + noise[1] * 0
        state.yaw = self.normalize_angle(state.yaw + state.omega * self.dt) + noise[2] * 0.
        state.v = state.v + u_v * self.dt + noise[0] * 0
        state.omega = self.normalize_angle(state.omega + u_o * self.dt) + noise[0] * 0.
        if state.v >= self.max_v:
            state.v = self.max_v
        if state.v <= 0.:
            state.v = 0.

        return state

    def local_planner(self, x0, original_path, transformation=False):
        if transformation:
            x0_ext = np.repeat(x0[:, None], 3, axis=1)
            x0_ = self.coordinate_transform(x0[2], x0[:2], x0_ext, mode="world to body")
            x0_b = np.copy(x0)
            x0_b[:3] = x0_[:3, 0]

            ref_path = self.coordinate_transform(original_path[2, 0], original_path[:2, 0], original_path,
                                                 mode="world to body")
            x0_in = x0_b
            path_in = ref_path
        else:
            x0_in = x0
            path_in = original_path

        # states: (x ,y ,theta ,v , steer, a, steer_rate, jerk)
        cmpc = CasADi_MPC_differ()
        cmpc.dt0 = self.dt
        # ref_traj = self.expand_path(original_path, self.ds)
        cmpc.horizon = self.window
        cmpc.v_end = cmpc.v_max
        cmpc.omega_end = cmpc.omega_max
        # print("mpc start:{}, {}".format(x0[0], x0[1]))

        cmpc.init_model_reference_line(x0_in, path_in)
        op_traj, op_control = cmpc.get_result_reference_line()

        # print("ds:", dt * cmpc.v_max, " horizon after expasion:", len(ref_traj.T))
        # print("MPC total time:{:.3f}s".format(time.time() - start_time))
        if transformation:
            print("[b] smooth start:{}, {}".format(op_traj[0, 0], op_traj[1, 0]))
            op_path = self.coordinate_transform(original_path[2, 0], original_path[:2, 0], op_traj,
                                                mode="body to world")
            print("[w] smooth start:{}, {}".format(op_path[0, 0], op_path[1, 0]))
            return np.vstack([op_path, op_control])
        else:
            return np.vstack([op_traj, op_control])

    def linear_error_mpc_control(self, index, xref, x0):
        mpc = CasADi_MPC_differ_Control()
        ex0 = x0 - xref[:5, index]
        mpc.dt0 = self.dt
        mpc.horizon = 5
        mpc.init_model_controller_conic(index, ex0, xref)
        # mpc.init_model_controller_qpsol(index, ex0, xref)
        op_traj = mpc.get_mpc_result()

        oa = op_traj[0, :]
        od = op_traj[1, :]

        return oa, od

    def check_goal(self, state, goal, tind, nind):
        # check goal
        dx = state.x - goal[0]
        dy = state.y - goal[1]
        d = np.hypot(dx, dy)

        isgoal = (d <= self.GOAL_DIS)

        if abs(tind - nind) >= 10:
            isgoal = False

        isstop = (abs(state.v) <= 0.05)

        if isgoal and isstop:
            return True

        return False

    def calc_ref_trajectory(self, state, global_path, pind):
        window = self.window + 1
        xref = np.zeros((4, window))
        ncourse = len(global_path.T)

        ind, _ = self.calc_nearest_index(state, global_path, pind)

        if pind >= ind:
            ind = pind

        xref[0, 0] = global_path[0, ind]
        xref[1, 0] = global_path[1, ind]
        xref[2, 0] = global_path[2, ind]
        xref[3, 0] = global_path[3, ind]
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
                xref[3, i] = global_path[3, i + ind]

            else:
                xref[0, i] = global_path[0, ncourse - 1]
                xref[1, i] = global_path[1, ncourse - 1]
                xref[2, i] = global_path[2, ncourse - 1]
                xref[3, i] = global_path[3, ncourse - 1]

        # xref = self.expand_path(xref, self.ds)
        return xref, ind

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
        planlist = []
        target_ind, _ = self.calc_nearest_index(state, global_path, 0)
        # cyaw = smooth_yaw(cyaw)
        alpha = 0.3
        tick = int(alpha * self.window)
        vm, om = None, None

        while self.MAX_TIME >= time:
            xref, target_ind = self.calc_ref_trajectory(state, global_path, target_ind)

            if (time > 15.) and (np.max(xref[2, :]) == np.max(xref[2, :])):
                break

            x0 = np.array([state.x, state.y, state.yaw, state.v, state.omega])

            if tick % int(alpha * self.window) == 0:
                smooth_traj = self.local_planner(x0, xref)
                planlist.append(smooth_traj)

            pp = 0
            if tick % 1 == 0:
                kk = tick % int(alpha * self.window)
                # am, dm = self.iterative_linear_mpc_control(smooth_traj, x0, vm, om)
                am, dm = self.linear_error_mpc_control(kk, smooth_traj, x0)
                self.fb_a.append(am[pp])
                self.fb_o_rate.append(dm[pp])
                state = self.update_state(state, 1 * am[pp] + smooth_traj[5, kk], 1 * dm[pp] + smooth_traj[6, kk])
                print("qp a:{:.3f} dmax:{:.3f}".format(am[pp], np.max(dm)))
                self.ref_v.append(smooth_traj[3, kk])
                self.ref_omega.append(smooth_traj[4, kk])
                self.ref_a.append(smooth_traj[5, kk])
                self.ref_omega_rate.append(smooth_traj[6, kk])
                time = time + self.dt

                x.append(state.x)
                y.append(state.y)
                yaw.append(state.yaw)
                v.append(state.v)
                omega.append(state.omega)
                elist.append(e)
                ethlist.append(e_th)
                t.append(time)

                # if self.check_goal(state, goal, target_ind, len(global_path.T)):
                #     break

                if self.show_animation:  # pragma: no cover
                    plt.cla()
                    # for stopping simulation with the esc key.
                    plt.gcf().canvas.mpl_connect('key_release_event',
                                                 lambda event: [exit(0) if event.key == 'escape' else None])
                    if smooth_traj is not None:
                        plt.plot(smooth_traj[0, :], smooth_traj[1, :], "xc", label="MPC smooth")

                    plt.plot(global_path[0, :], global_path[1, :], color="orange", label="course")
                    plt.plot(x, y, "xr", label="realtime traj")
                    plt.plot(xref[0, :], xref[1, :], color="purple", label="xref")
                    plt.plot(global_path[0, target_ind], global_path[1, target_ind], "xg", label="target")
                    self.plot_car(state)
                    plt.axis("equal")
                    plt.grid(True)
                    plt.title("Time[s]:{:.2f}, speed[m/s]:{:.2f}".format(time, state.v))
                    # f = plt.figure()
                    # self.plot_info_realtime(x, y, yaw, v, omega, elist, ethlist)
                    if tick % 10 == 0:
                        plt.pause(0.001)

            tick += 1

            if self.check_goal(state, goal, target_ind, len(global_path.T)):
                print("Goal")
                break

        real_traj = np.array([x, y, yaw, v, omega])
        errors = np.array([elist, ethlist])
        return t, real_traj, errors, planlist

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

    def plot_info_realtime(self, x, y, yaw, v, omega, elist, ethlist):
        plt.cla()

        ax = plt.subplot(211)
        ax.plot(np.rad2deg(yaw), "-r", label="lqr yaw")
        ax.plot(np.rad2deg(omega), "-c", label="lqr omega")
        ax.plot(np.rad2deg(self.ref_omega), "-b", label="ref omega")
        ax.plot(np.rad2deg(ethlist), color="tab:purple", label="heading error")
        ax.grid(True)
        ax.legend()
        ax.set_xlabel("line length[m]")
        ax.set_ylabel("lqr yaw angle[deg]")

        # f3 = plt.figure()
        # ax = plt.subplot(211)
        # # ax.plot(op_input[0, :], "-g", label="mpc v")
        # ax.legend()
        # ax.grid(True)
        # ax.set_ylabel("speed [m/s]")
        ax = plt.subplot(212)
        ax.plot(v, "-r", label="lqr v")
        ax.plot(self.ref_v, "-b", label="ref v")
        ax.plot(self.ref_a, "-b", label="ref a")
        ax.plot(elist, color="tab:purple", label="lateral error")
        ax.grid(True)
        ax.legend()
        ax.set_xlabel("line length[m]")
        ax.set_ylabel("speed [m/s]")

    def plot_info(self, global_path, flow, error, planlist, plot_controller_u=True):

        diff_s = np.diff(global_path[:2, :], axis=1)
        sum_s = np.sum(np.hypot(diff_s[0], diff_s[1]))
        # plt.cla()
        s_realtime = np.linspace(0, sum_s, len(flow.T))
        ds = s_realtime[1]
        s_reference = np.linspace(0, sum_s, len(global_path.T))
        # sline = np.arange(0, ds * len(self.fb_a[3]), ds)

        f2 = plt.figure()
        ax = plt.subplot(221)
        # ax.plot(s_reference, np.rad2deg(global_path[0, :]), "-r", label="ref x")
        # ax.plot(s_reference, np.rad2deg(global_path[0, :]), "-r", label="ref x")
        ax.plot(s_realtime, flow[3, :], "-r", label="real v")
        ax.plot(s_realtime, self.ref_v, "-b", label="used ref v")
        ax.plot(s_realtime, self.ref_a, "-.", color="tab:blue", label="used ref a")
        # for i, line in enumerate(self.fb_a):
        #     ax.plot(i * ds + sline, line, "-.", color='crimson', label="fb a")
        if plot_controller_u:
            ax.plot(s_realtime, self.fb_a, "-.", color='crimson', label="fb a")

        ax.plot(s_realtime, error[0, :], color="tab:purple", label="lateral error")
        ax.grid(True)
        ax.legend()
        ax.set_xlabel("line length[m]")
        ax.set_ylabel("y [m]")

        ax = plt.subplot(223)
        ax.plot(s_reference, np.rad2deg(global_path[2, :]), "-g", label="ref yaw")
        ax.plot(s_realtime, np.rad2deg(flow[2, :]), "-r", label="real yaw")
        ax.plot(s_realtime, np.rad2deg(self.ref_omega), "-b", label="used ref omega")
        ax.plot(s_realtime, np.rad2deg(self.ref_omega_rate), "-.", color="tab:blue", label="used ref omega_rate")
        # for i, line in enumerate(self.fb_o_rate):
        #     ax.plot(i * ds + sline, np.rad2deg(line), "-.", color='crimson', label="fb o rate")
        if plot_controller_u:
            ax.plot(s_realtime, np.rad2deg(self.fb_o_rate), "-.", color='crimson', label="fb o rate")

        ax.plot(s_realtime, np.rad2deg(flow[4, :]), "-c", label="lqr omega")
        ax.plot(s_realtime, np.rad2deg(error[1, :]), color="tab:purple", label="heading error")
        ax.grid(True)
        ax.legend()
        ax.set_xlabel("line length[m]")
        ax.set_ylabel("angle [deg]")

        ax = plt.subplot(222)
        ax.plot(global_path[0, :], "tab:blue", label="ref x")
        ax.plot(flow[0, :], "-r", label="real x")
        ax.plot(error[0, :], color="tab:purple", label="lateral error")
        ax.grid(True)
        ax.legend()
        ax.set_xlabel("line length[m]")
        ax.set_ylabel("x [m]")

        ax = plt.subplot(224)
        ax.plot(global_path[1, :], "tab:blue", label="ref y")
        ax.plot(flow[1, :], "-r", label="real y")
        ax.grid(True)
        ax.legend()
        ax.set_xlabel("line length[m]")
        ax.set_ylabel("y [m]")

        f2 = plt.figure()
        ax = plt.subplot(211)
        ax.plot(s_realtime, flow[3, :], "-r", label="real v")
        ax.plot(s_realtime, self.ref_v, "-b", label="used ref v")
        strecke = sum_s / len(planlist)
        for i, plan in enumerate(planlist):
            sline = np.linspace(i * strecke, (i + 1) * strecke, len(plan.T))
            ax.plot(sline, plan[3, :], color="tab:green", label="plan v")

        ax.grid(True)
        ax.legend()
        ax.set_xlabel("line length[m]")
        ax.set_ylabel("y [m]")

        ax = plt.subplot(212)
        for i, plan in enumerate(planlist):
            sline = np.linspace(i * strecke, (i + 1) * strecke, len(plan.T))
            ax.plot(sline, np.abs(plan[4, :] / (plan[3, :] + 1e-5)), color="tab:green", label="cal curvature")
        ax.plot(s_reference, global_path[3, :], color="green", label="ref curvature")
        ax.grid(True)
        # ax.legend()
        ax.set_xlabel("line length[m]")
        ax.set_ylabel("curvature")

        f = plt.figure()
        ax = plt.subplot(111)
        ax.plot(global_path[0, :], global_path[1, :], "-g", label="waypoints")
        # ax.plot(cx, cy, "-r", label="target course")
        ax.plot(flow[0, :], flow[1, :], "xb", label="tracking")
        ax.grid(True)
        ax.axis("equal")
        ax.set_xlabel("x[m]")
        ax.set_ylabel("y[m]")
        ax.legend()


def get_raw_course(mpc, dl):
    ax = [0.0, 5.0, 10.0, 10]
    ay = [0.0, -5.0, 2.0, 10]
    control_points = np.array([ax, ay])

    return mpc.expand_path(control_points, ds=dl)


def main():
    address = "../../config_differ_smoother.yaml"

    load_file = False
    dl = 0.5
    address = "../config_differ_smoother.yaml"
    with open(address, 'r', encoding='utf-8') as f:
        param = yaml.load(f)
    ut = UTurnMPC()
    # ut.set_parameters(param)
    mpclqr = MPC_LQR_Controller()

    if not load_file:
        raw_course = get_raw_course(mpclqr, dl)
        # global_path = mpclqr.expand_path(raw_course, 0.05)
        global_path = mpclqr.expand_path(raw_course, mpclqr.ds)
        ran = np.random.rand(5)
        x0_ = global_path[0, 0] - 2.5 * 1
        y0_ = global_path[1, 0] - .5 * 1
        yaw0_ = global_path[2, 0] + ran[2] * 1
        v0_ = 0.
        omega0_ = 0.

        state0 = mpclqr.State(x=x0_, y=y0_, yaw=yaw0_, v=v0_, omega=omega0_)

        t, real_traj, error_traj, planlist = mpclqr.do_simulation(global_path, state0)

        mpclqr.plot_info(global_path, real_traj, error_traj, planlist)

        # np.savez("../../data/smoothed_traj_differ", dt=dt, traj=op_path, control=op_input,
        #          refpath=raw_course)

    plt.show()


if __name__ == '__main__':
    main()
