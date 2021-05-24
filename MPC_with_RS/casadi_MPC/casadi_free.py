import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from mpc_motion_plot import UTurnMPC
from gears.polygon_generator import cal_coeff_mat


class CasADi_MPC:
    def __init__(self):
        self.base = 3.0
        self.nx = 8
        self.ng = 6
        self.model = None
        self.x_opt = None
        self.obst_num = 0
        self.horizon = 0

        self.wg = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
        self.v_max = 3.
        self.steer_max = ca.pi / 2.
        self.omega_max = ca.pi / 4.
        self.a_max = 2.
        self.steer_rate_max = ca.pi / 8.
        self.jerk_max = 4.

    def Array2SX(self, array):
        rows, cols = np.shape(array)
        sx = ca.SX.zeros(rows * cols, 1)
        array_ = array.T.flatten()
        for i in range(rows * cols):
            sx[i] = array_[i]

        return ca.reshape(sx, rows, cols)

    def init_dynamic_constraints(self, x, dt, gx):
        for i in range(self.horizon - 1):
            x_ = x[0, i]
            y_ = x[1, i]
            th_ = x[2, i]
            v_ = x[3, i]
            steer_ = x[4, i]
            a_ = x[5, i]
            steer_rate_ = x[6, i]
            jerk_ = x[7, i]

            dx = x_ + v_ * ca.cos(th_) * dt
            dy = y_ + v_ * ca.sin(th_) * dt
            dth = th_ + v_ / self.base * ca.tan(steer_) * dt
            dv = v_ + a_ * dt
            dsteer = steer_ + steer_rate_ * dt
            da = a_ + jerk_ * dt

            gx[0, i] = dx - x[0, i + 1]
            gx[1, i] = dy - x[1, i + 1]
            gx[2, i] = dth - x[2, i + 1]
            gx[3, i] = dv - x[3, i + 1]
            gx[4, i] = dsteer - x[4, i + 1]
            gx[5, i] = da - x[5, i + 1]

        return gx

    def init_objects(self, x, dt, ref_path):
        sum_total_dist = 0
        sum_time = 0
        sum_vel = 0
        sum_steer = 0
        sum_a = 0
        sum_steer_rate = 0
        sum_dist_to_ref = 0

        for i in range(self.horizon - 1):
            sum_time += dt
            sum_vel += ca.power(x[3, i], 2)
            sum_steer += ca.power(x[4, i], 2)
            sum_a += ca.power(x[5, i], 2)
            sum_steer_rate += ca.power(x[6, i], 2)
            sum_dist_to_ref += ca.sumsqr(x[:3, i] - ref_path[:3, i])

            if i > 1:
                sum_total_dist += ca.sumsqr(x[:2, i] - x[:2, i - 1])

        obj = self.wg[5] * sum_total_dist + self.wg[5] * sum_time \
              + self.wg[2] * sum_vel + self.wg[4] * sum_steer \
              + self.wg[4] * sum_a + self.wg[3] * sum_steer_rate \
              + self.wg[2] * sum_dist_to_ref

        return obj

    def init_bounds(self, start, goal):
        lbx = ca.DM(self.nx, self.horizon)
        ubx = ca.DM(self.nx, self.horizon)
        lbg = ca.DM(self.ng, self.horizon - 1)
        ubg = ca.DM(self.ng, self.horizon - 1)

        for i in range(self.horizon - 1):
            lbx[0, i] = -ca.inf  # x
            lbx[1, i] = -ca.inf  # y
            lbx[2, i] = -ca.pi  # th
            lbx[3, i] = -self.v_max  # v
            lbx[4, i] = -self.steer_max  # steer
            lbx[5, i] = -self.a_max  # a
            lbx[6, i] = -self.steer_rate_max  # steer_rate
            lbx[7, i] = -self.jerk_max  # jerk

            ubx[0, i] = ca.inf  # x
            ubx[1, i] = ca.inf  # y
            ubx[2, i] = ca.pi  # th
            ubx[3, i] = self.v_max  # v
            ubx[4, i] = self.steer_max  # steer
            ubx[5, i] = self.a_max  # a
            ubx[6, i] = self.steer_rate_max  # steer_rate
            ubx[7, i] = self.jerk_max  # jerk

        lbx[0, 0] = start[0]
        lbx[1, 0] = start[1]
        lbx[2, 0] = start[2]
        lbx[3:8, 0] = 0.

        ubx[0, 0] = start[0]
        ubx[1, 0] = start[1]
        ubx[2, 0] = start[2]
        ubx[3:8, 0] = 0.

        lbx[0, -1] = goal[0]
        lbx[1, -1] = goal[1]
        lbx[2, -1] = goal[2]
        lbx[3:8, -1] = 0.

        ubx[0, -1] = goal[0]
        ubx[1, -1] = goal[1]
        ubx[2, -1] = goal[2]
        ubx[3:8, -1] = 0.

        lbx_ = ca.reshape(lbx, -1, 1)
        ubx_ = ca.reshape(ubx, -1, 1)
        lbx_ = ca.vertcat(0.1, lbx_)
        ubx_ = ca.vertcat(1, ubx_)

        lbg_ = ca.reshape(lbg, -1, 1)
        ubg_ = ca.reshape(ubg, -1, 1)

        return lbx_, ubx_, lbg_, ubg_

    def states_initialization(self, reference_path):

        x0 = ca.DM(self.nx, self.horizon)
        for i in range(self.horizon):
            x0[0, i] = reference_path[0, i]
            x0[1, i] = reference_path[1, i]
            x0[2, i] = reference_path[2, i]
            x0[3:, i] = 0.

        return x0

    def dual_variables_initialization(self):
        t0 = ca.DM(4 * (self.obst_num + 1), self.horizon)
        return t0

    def init_model(self, reference_path):
        self.horizon = reference_path.shape[1]
        x0_ = self.states_initialization(reference_path)

        # states: (x ,y ,theta ,v , steer, a, steer_rate, jerk)
        x = ca.SX.sym("x", self.nx, self.horizon)
        dt = ca.SX.sym("dt")
        gx = ca.SX.sym("g", self.ng, self.horizon - 1)
        gx = self.init_dynamic_constraints(x, dt, gx)

        obj = self.init_objects(x, dt, x0_)
        x_ = ca.reshape(x, -1, 1)
        X = ca.vertcat(dt, x_)
        G = ca.reshape(gx, -1, 1)

        nlp = {"x": X, "f": obj, "g": G}
        opts_setting = {"expand": True,
                        "ipopt.hessian_approximation": "limited-memory",
                        'ipopt.max_iter': 100,
                        'ipopt.print_level': 2,
                        'print_time': 1,
                        'ipopt.acceptable_tol': 1e-8,
                        'ipopt.acceptable_obj_change_tol': 1e-6}

        Sol = ca.nlpsol('S', 'ipopt', nlp, opts_setting)

        diff_s = np.diff(reference_path[:2, :], axis=1)
        sum_s = np.sum(np.hypot(diff_s[0], diff_s[1]))
        dt_ = sum_s / self.v_max
        x0 = ca.reshape(x0_, -1, 1)
        X0 = ca.vertcat(dt_, x0)

        lbx, ubx, lbg, ubg = self.init_bounds(x0_[:, 0], x0_[:, -1])
        result = Sol(x0=X0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
        self.x_opt = result["x"]

    def get_result(self):
        op_dt = float(self.x_opt[0])
        cal_traj = ca.reshape(self.x_opt[1:], self.nx, self.horizon)
        op_controls = np.array(cal_traj[3:, :])
        op_trajectories = np.array(cal_traj[:3, :])
        return op_dt, op_trajectories, op_controls


if __name__ == '__main__':

    tracking = True
    test_mpc_rs = False
    test_mpc_obca = True

    loadtraj = np.load("../../saved_hybrid_a_star.npz")
    ref_traj = loadtraj["saved_traj"]
    loadmap = np.load("saved_obmap.npz")
    ob = loadmap["pointmap"]
    ob_constraint_mat = loadmap["constraint_mat"]
    obst = []
    obst.append(ob_constraint_mat[:4, :])
    obst.append(ob_constraint_mat[4:8, :])
    obst.append(ob_constraint_mat[8:12, :])
    ut = UTurnMPC()
    start = ref_traj[:, 0]
    goal = ref_traj[:, -1]
    car_outline = np.array([
        [ut.LF, ut.LF, -ut.LB, -ut.LB, ut.LF],
        [ut.W / 2, -ut.W / 2, - ut.W / 2, ut.W / 2, ut.W / 2]])
    shape = cal_coeff_mat(car_outline.T)
    # states: (x ,y ,theta ,v , steer, a, steer_rate, jerk)
    cmpc = CasADi_MPC()
    # cmpc.init_model(ref_traj)
    cmpc.init_model(ref_traj)
    op_dt, op_trajectories, op_controls = cmpc.get_result()

    ut.predicted_trajectory = op_trajectories
    zst = start
    trajectory = np.copy(zst)

    ut.cal_distance(op_trajectories[:2, :], cmpc.horizon)
    ut.dt = op_dt
    print("Time resolution:{:.3f}s, total time:{:.3f}s".format(ut.dt, ut.dt * len(ref_traj.T)))
    # np.savez("saved_traj", traj=predicted_trajectory)

    fig = plt.figure()
    ax1 = plt.subplot(111)
    ax1.plot(op_controls[0, :], label="v")
    ax1.plot(op_controls[1, :], label="steer")
    ax1.plot(op_controls[2, :], "-.", label="acc")
    ax1.plot(op_controls[3, :], "-.", label="steer rate")
    ax1.grid()
    ax1.legend()

    if tracking:
        trajectory = ut.try_tracking(zst, op_controls, trajectory, obst=ob.T, ref_traj=ref_traj)
        print("Done")

    plt.show()
