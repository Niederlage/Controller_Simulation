import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from mpc_motion_plot import UTurnMPC, get_car_shape, initialize_saved_data


class CasADi_MPC_OBCA:
    def __init__(self):
        self.base = 2.0
        self.nx = 8
        self.ng = 6
        self.obst_num = 0
        self.horizon = 0
        self.model = None
        self.x_opt = None
        self.op_lambda0 = None
        self.op_mu0 = None

        self.wg = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
        self.v_max = 2.5
        self.steer_max = ca.pi / 2.
        self.omega_max = ca.pi / 4.
        self.a_max = 2.
        self.steer_rate_max = ca.pi / 8.
        self.jerk_max = 4.
        self.dmin = 0.

    def adapt_g_shape(self):
        self.ng = self.obst_num * 4 + 6

    def Array2SX(self, array):
        rows, cols = np.shape(array)
        sx = ca.SX.zeros(rows * cols, 1)
        array_ = array.T.flatten()
        for i in range(rows * cols):
            sx[i] = array_[i]

        return ca.reshape(sx, rows, cols)

    def Array2DM(self, array):
        rows, cols = np.shape(array)
        sx = ca.DM.zeros(rows * cols, 1)
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

    def init_OBCA_constraints(self, x_, lambda_, mu_, shape, obst, gx):
        gT = self.Array2SX(shape[:, 2][:, None].T)
        GT = self.Array2SX(shape[:, :2].T)

        l_ob = len(obst)
        for i in range(self.horizon - 1):
            mu_i = mu_[:, i]
            yaw_i = x_[2, i]
            # offset = ca.SX.zeros(2, 1)
            # offset[0, 0] = 1 * ca.cos(yaw_i)
            # offset[1, 0] = 1 * ca.sin(yaw_i)
            t_i = x_[:2, i]
            rotT_i = ca.SX.zeros(2, 2)
            rotT_i[0, 0] = ca.cos(yaw_i)
            rotT_i[1, 1] = ca.cos(yaw_i)
            rotT_i[1, 0] = -ca.sin(yaw_i)
            rotT_i[0, 1] = ca.sin(yaw_i)

            for j in range(l_ob):
                Aj = self.Array2SX(obst[j][:, :2])
                bj = self.Array2SX(obst[j][:, 2][:, None])
                lambdaj = lambda_[(4 * j):(4 * (j + 1)), i]

                constraint1 = rotT_i @ Aj.T @ lambdaj + GT @ mu_i
                constraint2 = (Aj @ t_i - bj).T @ lambdaj - gT @ mu_i - self.dmin
                temp = Aj.T @ lambdaj
                constraint3 = ca.power(temp[0, 0], 2) + ca.power(temp[1, 0], 2) - 1.

                gx[6 + j, i] = constraint1[0, 0]
                gx[6 + j + l_ob, i] = constraint1[1, 0]
                gx[6 + j + 2 * l_ob, i] = constraint2
                gx[6 + j + 3 * l_ob, i] = constraint3

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

        obj = self.wg[3] * sum_total_dist + self.wg[1] * sum_time \
              + self.wg[3] * sum_vel + self.wg[4] * sum_steer \
              + self.wg[3] * sum_a + self.wg[4] * sum_steer_rate \
              + self.wg[3] * sum_dist_to_ref

        return obj

    def init_bounds_OBCA(self, start, goal):
        lbx = ca.DM(self.nx + (self.obst_num + 1) * 4, self.horizon)
        ubx = ca.DM(self.nx + (self.obst_num + 1) * 4, self.horizon)
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
            lbx[8:, i] = 0.  # lambda, mu

            ubx[0, i] = ca.inf  # x
            ubx[1, i] = ca.inf  # y
            ubx[2, i] = ca.pi  # th
            ubx[3, i] = self.v_max  # v
            ubx[4, i] = self.steer_max  # steer
            ubx[5, i] = self.a_max  # a
            ubx[6, i] = self.steer_rate_max  # steer_rate
            ubx[7, i] = self.jerk_max  # jerk
            ubx[8:, i] = ca.inf  # lambda, mu

            # lbg[6:6 + 2 * self.obst_num, i] = 0.
            # ubg[6:6 + 2 * self.obst_num, i] = 0.

            # constraint2 (Aj @ t_i - bj).T @ lambdaj - gT @ mu_i
            lbg[6 + 2 * self.obst_num:6 + 3 * self.obst_num, i] = 0.
            ubg[6 + 2 * self.obst_num:6 + 3 * self.obst_num, i] = ca.inf

            # constraint3  norm_2(Aj.T @ lambdaj) - 1
            lbg[6 + 3 * self.obst_num:6 + 4 * self.obst_num, i] = -ca.inf
            ubg[6 + 3 * self.obst_num:6 + 4 * self.obst_num, i] = 0.

        lbx[0, 0] = start[0]
        lbx[1, 0] = start[1]
        lbx[2, 0] = start[2]
        lbx[3:, 0] = 0.

        ubx[0, 0] = start[0]
        ubx[1, 0] = start[1]
        ubx[2, 0] = start[2]
        ubx[3:, 0] = 0.

        lbx[0, -1] = goal[0]
        lbx[1, -1] = goal[1]
        lbx[2, -1] = goal[2]
        lbx[3:, -1] = 0.

        ubx[0, -1] = goal[0]
        ubx[1, -1] = goal[1]
        ubx[2, -1] = goal[2]
        ubx[3:, -1] = 0.

        lbx_ = ca.reshape(lbx, -1, 1)
        ubx_ = ca.reshape(ubx, -1, 1)
        lbx_ = ca.vertcat(0.01, lbx_)
        ubx_ = ca.vertcat(0.15, ubx_)

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
        t0 = ca.DM.zeros(4 * (self.obst_num + 1), self.horizon)
        if (self.op_lambda0 is not None) and (self.op_mu0 is not None):
            t0[:4 * self.obst_num, :] = self.Array2DM(self.op_lambda0)
            t0[4 * self.obst_num:, :] = self.Array2DM(self.op_mu0)

        return t0

    def init_model_OBCA(self, reference_path, shape_m, obst_m):
        self.horizon = reference_path.shape[1]
        self.obst_num = len(obst_m)
        self.adapt_g_shape()
        x0_ = self.states_initialization(reference_path)
        t0 = self.dual_variables_initialization()

        # initialize variables
        x = ca.SX.sym("x", self.nx, self.horizon)  # (x ,y ,theta ,v , steer, a, steer_rate, jerk)
        dt = ca.SX.sym("dt")
        lambda_ = ca.SX.sym("lambda", 4 * self.obst_num, self.horizon)
        mu_ = ca.SX.sym("mu", 4, self.horizon)

        # initialize constraints
        gx = ca.SX.sym("g", self.ng, self.horizon - 1)
        gx = self.init_dynamic_constraints(x, dt, gx)
        gx = self.init_OBCA_constraints(x, lambda_, mu_, shape_m, obst_m, gx)

        X, G = self.organize_variables(dt, x, lambda_, mu_, gx)
        X0, XL, XU, GL, GU = self.organize_bounds(reference_path, x0_, t0)

        # initialize objectives
        F = self.init_objects(x, dt, x0_)

        nlp = {"x": X, "f": F, "g": G}
        opts_setting = {"expand": True,
                        "ipopt.hessian_approximation": "limited-memory",
                        'ipopt.max_iter': 1000,
                        'ipopt.print_level': 3,
                        'print_time': 1,
                        'ipopt.acceptable_tol': 1e-8,
                        'ipopt.acceptable_obj_change_tol': 1e-6}

        Sol = ca.nlpsol('S', 'ipopt', nlp, opts_setting)

        result = Sol(x0=X0, lbx=XL, ubx=XU, lbg=GL, ubg=GU)
        self.x_opt = result["x"]

    def organize_variables(self, dt, vx, vl, vm, cg):
        vs = ca.vertcat(vx, vl)
        vs = ca.vertcat(vs, vm)
        x_ = ca.reshape(vs, -1, 1)
        X = ca.vertcat(dt, x_)
        G = ca.reshape(cg, -1, 1)

        return X, G

    def organize_bounds(self, reference_path, x0, y0):
        diff_s = np.diff(reference_path[:2, :], axis=1)
        sum_s = np.sum(np.hypot(diff_s[0], diff_s[1]))
        dt_ = sum_s / self.v_max
        lbx, ubx, lbg, ubg = self.init_bounds_OBCA(x0[:, 0], x0[:, -1])

        x_all = ca.vertcat(x0, y0)
        x0_ = ca.reshape(x_all, -1, 1)
        X0 = ca.vertcat(dt_, x0_)

        return X0, lbx, ubx, lbg, ubg

    def get_result_OBCA(self):
        op_dt = float(self.x_opt[0])
        cal_traj = ca.reshape(self.x_opt[1:], self.nx + (self.obst_num + 1) * 4, self.horizon)
        op_controls = np.array(cal_traj[3:8, :])
        op_trajectories = np.array(cal_traj[:3, :])
        vl = np.array(cal_traj[8:20, :])
        vm = np.array(cal_traj[20:, :])
        return op_dt, op_trajectories, op_controls


if __name__ == '__main__':

    tracking = True
    ut = UTurnMPC()

    # states: (x ,y ,theta ,v , steer, a, steer_rate, jerk)
    cmpc = CasADi_MPC_OBCA()
    ref_traj, ob, obst = initialize_saved_data()
    shape = get_car_shape(ut)

    cmpc.init_model_OBCA(ref_traj, shape, obst)
    op_dt, op_trajectories, op_controls = cmpc.get_result_OBCA()

    ut.predicted_trajectory = op_trajectories
    zst = ref_traj[:, 0]
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
        trajectory = ut.try_tracking(zst, op_controls, trajectory, obst=ob, ref_traj=ref_traj)
        print("Done")

    plt.show()
