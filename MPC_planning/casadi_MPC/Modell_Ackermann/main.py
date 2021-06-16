import time

import numpy as np
import yaml
from gears.cubic_spline_planner import Spline2D
from mpc_motion_plot import UTurnMPC
from Modell_Ackermann.casadi_OBCA_warmup import CasADi_MPC_WarmUp
from Modell_Ackermann.casadi_OBCA import CasADi_MPC_OBCA
from Modell_Ackermann.casadi_TDROBCA import CasADi_MPC_TDROBCA
# from casadi_MPC.casadi_TDROBCA_v2 import CasADi_MPC_TDROBCA
# from casadi_MPC.casadi_TDROBCA_v3 import CasADi_MPC_TDROBCA
from MPC_planning.HybridAStar.hybrid_a_star import HybridAStar


def hybrid_a_star_initialization(large):
    if large:
        address = "../config_OBCA_large.yaml"
    else:
        address = "../config_OBCA.yaml"
    with open(address, 'r', encoding='utf-8') as f:
        param = yaml.load(f)

    coarse_planner = HybridAStar()
    coarse_planner.show_animation = False
    sx = param["start"]
    ex = param["goal"]
    start = [sx[0], sx[1], np.deg2rad(sx[2])]
    goal = [ex[0], ex[1], np.deg2rad(ex[2])]
    if large:
        coarse_planner.car.set_parameters(param)

    ob, obst, obst_points = get_all_obsts()
    path = coarse_planner.hybrid_a_star_planning(start, goal, obst_points)

    if path is not None:
        x = path.x_list
        y = path.y_list
        yaw = path.yaw_list
        saved_path = np.array([x, y, yaw])
    else:
        saved_path = None

    return saved_path, param, obst, ob


def get_bounds():
    pa = [-10, -10]
    pb = [20, 20]

    edge = np.mgrid[pa[0]:pb[0]:0.1, pa[1]:pb[1]:0.1]
    lx = edge[0, :, 0].flatten()
    ly = edge[0, :, 0].flatten()
    x_ = np.ones((len(lx),))
    y_ = np.ones((len(ly),))
    horizon_l = np.vstack((lx, pa[1] * x_))[:, :edge.shape[1]]
    horizon_u = np.vstack((lx, pb[1] * x_))[:, :edge.shape[1]]
    vertical_l = np.vstack((pa[0] * y_, ly))[:, :edge.shape[2]]
    vertical_u = np.vstack((pb[0] * y_, ly))[:, :edge.shape[2]]
    return np.block([horizon_l, horizon_u, vertical_l, vertical_u])


def expand_path(refpath, ds):
    x = refpath[0, :]
    y = refpath[1, :]
    theta0 = refpath[2, 0]
    sp = Spline2D(x, y)
    s = np.arange(0, sp.s[-1], ds)

    rx, ry, ryaw, rk = [], [], [], []
    yaw_last = theta0
    for i_s in s:
        ix, iy = sp.calc_position(i_s)
        rx.append(ix)
        ry.append(iy)
        yaw_ = sp.calc_yaw(i_s)
        ryaw.append(yaw_)
        yaw_last = yaw_
        # rk.append(sp.calc_curvature(i_s))
    return np.array([rx, ry, ryaw])


def get_all_obsts():
    loadmap = np.load("../data/saved_obmap.npz", allow_pickle=True)
    ob1 = loadmap["pointmap"][0]
    ob2 = loadmap["pointmap"][1]
    ob3 = loadmap["pointmap"][2]
    bounds = get_bounds()
    obst_points = np.block([ob1.T, ob2.T, ob3.T, bounds])
    ob = [ob1, ob2, ob3, bounds.T]
    # obst_points = np.block([ob2.T, bounds])
    # ob = ob2

    ob_constraint_mat = loadmap["constraint_mat"]
    obst = []
    obst.append(ob_constraint_mat[4:8, :])
    obst.append(ob_constraint_mat[:4, :])
    obst.append(ob_constraint_mat[8:12, :])
    return ob, obst, obst_points


def run_HOBCA_mpc(ref_traj, shape, obst):
    warmup_time = time.time()
    # states: (x ,y ,theta ,v , steer, a, steer_rate, jerk)
    warmup_qp = CasADi_MPC_WarmUp()
    warmup_qp.init_model_warmup(ref_traj, shape, obst)
    op_dist, op_lambda, op_mu = warmup_qp.get_result_warmup()
    print("warm up time:{:.3f}s".format(time.time() - warmup_time))

    obca = CasADi_MPC_OBCA()
    obca.op_lambda0 = op_lambda
    obca.op_mu0 = op_mu
    obca.init_model_OBCA(ref_traj, shape, obst)
    op_dt, op_trajectories, op_controls, op_lambda, op_mu = obca.get_result_OBCA()

    return op_dt, op_trajectories, op_controls


def run_segment_OBCA_mpc(param, ref_traj, shape, obst):
    s_half = int(len(ref_traj.T) / 2)
    s_half2 = len(ref_traj.T) - s_half

    op_trajectories = ref_traj[:, :s_half]
    op_dist = np.zeros((len(obst), s_half))
    op_lambda = np.zeros((4 * len(obst), s_half))
    op_mu = np.zeros((4, s_half))

    all_traj = []
    for i in range(2):
        warmup_time = time.time()
        # states: (x ,y ,theta ,v , steer, a, steer_rate, jerk)
        if i > 0:
            op_trajectories = ref_traj[:, s_half - 1:]
            op_dist = np.zeros((len(obst), s_half2 + 1))
            op_lambda = np.zeros((4 * len(obst), s_half2 + 1))
            op_mu = np.zeros((4, s_half2 + 1))

        obca = CasADi_MPC_TDROBCA()
        obca.get_dt(ref_traj)
        obca.set_parameters(param)
        obca.op_lambda0 = op_lambda
        obca.op_mu0 = op_mu
        obca.op_d0 = op_dist
        obca.init_model_OBCA(op_trajectories, shape, obst)
        op_dt, op_trajectories, op_controls, op_lambda, op_mu = obca.get_result_OBCA()
        all_traj.append([op_dt, op_trajectories, op_controls, op_lambda, op_mu])

    dt_op = (all_traj[0][0] + all_traj[1][0]) / 2
    traj_op = np.block([all_traj[0][1], all_traj[1][1]])
    controls_op = np.block([all_traj[0][2], all_traj[1][2]])

    return dt_op, traj_op, controls_op


def run_TDROBCA_mpc(param, ref_traj, shape, obst):
    warmup_time = time.time()

    # pathonly = CasADi_MPC_OBCA_PathOnly()
    # pathonly.set_parameters(param)
    # pathonly.init_model_OBCA(ref_path, shape, obst)
    # ref_traj, kappa, vl, vm = pathonly.get_result_OBCA()

    # states: (x ,y ,theta ,v , steer, a, steer_rate, jerk)
    warmup_qp = CasADi_MPC_WarmUp()
    warmup_qp.set_parameters(param)
    warmup_qp.init_model_warmup(ref_traj, shape, obst)
    op_dist, op_lambda, op_mu = warmup_qp.get_result_warmup()
    print("warm up time:{:.3f}s".format(time.time() - warmup_time))

    obca = CasADi_MPC_TDROBCA()
    obca.get_dt(ref_traj)
    obca.set_parameters(param)
    obca.op_lambda0 = op_lambda
    obca.op_mu0 = op_mu
    obca.op_d0 = op_dist
    obca.init_model_OBCA(ref_traj, shape, obst)
    op_dt, op_trajectories, op_controls, op_lambda, op_mu = obca.get_result_OBCA()

    return op_dt, op_trajectories, op_controls


def main():
    address = "../config_OBCA_large.yaml"
    HOBCA_mpc = False
    try_segment = False
    load_file = False
    large = True
    ds = 0.1

    if not load_file:
        ref_path, param, obst, ob_points = hybrid_a_star_initialization(large)
        ref_traj = expand_path(ref_path, ds)
        # ref_traj = ref_path
        if len(ref_path.T) > 500:
            ref_traj = None

        if ref_traj is not None:

            ut = UTurnMPC()
            ut.set_parameters(param)
            ut.reserve_footprint = True
            shape = ut.get_car_shape()

            start_time = time.time()

            if HOBCA_mpc:
                op_dt, op_trajectories, op_controls = run_HOBCA_mpc(ref_traj, shape, obst)

            else:
                if try_segment:
                    op_dt, op_trajectories, op_controls = run_segment_OBCA_mpc(param, ref_traj, shape, obst)
                else:
                    op_dt, op_trajectories, op_controls = run_TDROBCA_mpc(param, ref_traj, shape, obst)

            print("warm up OBCA total time:{:.3f}s".format(time.time() - start_time))
            # np.savez("../data/smoothed_traj", dt=op_dt, traj=op_trajectories, control=op_controls, refpath=ref_traj)

            ut.plot_results(op_dt, op_trajectories, op_controls, ref_traj, ob_points, four_states=True)

        else:
            print("Hybrid A Star initialization failed ....")

    else:
        loads = np.load("../data/smoothed_traj.npz")
        op_dt = loads["dt"]
        op_trajectories = loads["traj"]
        op_controls = loads["control"]
        ref_traj = loads["refpath"]

        print("load file to check!")
        ut = UTurnMPC()
        with open(address, 'r', encoding='utf-8') as f:
            param = yaml.load(f)
        ob_points, obst, ob_array = get_all_obsts()
        ut.set_parameters(param)
        ut.reserve_footprint = True
        ut.plot_results(op_dt, op_trajectories, op_controls, ref_traj, ob_points, four_states=True)


if __name__ == '__main__':
    main()
