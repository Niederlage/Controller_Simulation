import time

import numpy as np
import yaml
from mpc_motion_plot import UTurnMPC
from casadi_MPC.casadi_OBCA_warmup import CasADi_MPC_WarmUp
from casadi_MPC.casadi_OBCA import CasADi_MPC_OBCA
from casadi_MPC.casadi_TDROBCA import CasADi_MPC_TDROBCA
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

    loadmap = np.load("../data/saved_obmap.npz", allow_pickle=True)
    ob1 = loadmap["pointmap"][0]
    ob2 = loadmap["pointmap"][1]
    ob3 = loadmap["pointmap"][2]
    bounds = coarse_planner.get_bounds()
    obst_points = np.block([ob1.T, ob2.T, ob3.T, bounds])
    ob = [ob1, ob2, ob3]
    # obst_points = np.block([ob2.T, bounds])
    # ob = ob2

    ob_constraint_mat = loadmap["constraint_mat"]
    obst = []
    obst.append(ob_constraint_mat[:4, :])
    obst.append(ob_constraint_mat[4:8, :])
    obst.append(ob_constraint_mat[8:12, :])

    path = coarse_planner.hybrid_a_star_planning(start, goal, obst_points)

    if path is not None:
        x = path.x_list
        y = path.y_list
        yaw = path.yaw_list
        saved_path = np.array([x, y, yaw])
    else:
        saved_path = None

    return saved_path, param, obst, ob, [ob1, ob2, ob3, bounds.T]


def run_iterative_mpc(ref_traj, shape, obst):
    op_trajectories = ref_traj
    op_controls = np.zeros((5, len(ref_traj.T)))
    op_dist = np.zeros((len(obst), len(ref_traj.T)))
    op_lambda = np.zeros((4 * len(obst), len(ref_traj.T)))
    op_mu = np.zeros((4, len(ref_traj.T)))
    op_dt = 0.1

    for i in range(5):
        warmup_time = time.time()
        # states: (x ,y ,theta ,v , steer, a, steer_rate, jerk)
        warmup_qp = CasADi_MPC_WarmUp()
        warmup_qp.op_dist0 = op_dist
        warmup_qp.op_lambda0 = op_lambda
        warmup_qp.op_mu0 = op_mu
        warmup_qp.init_model_warmup(op_trajectories, shape, obst)
        op_dist, op_lambda, op_mu = warmup_qp.get_result_warmup()
        print("warm up time:{:.3f}s".format(time.time() - warmup_time))

        obca = CasADi_MPC_OBCA()
        obca.op_lambda0 = op_lambda
        obca.op_mu0 = op_mu
        obca.init_model_OBCA(op_trajectories, shape, obst)
        op_dt, op_trajectories, op_controls, op_lambda, op_mu = obca.get_result_OBCA()
        obca.op_control0 = op_controls

    return op_dt, op_trajectories, op_controls


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
    iterative_mpc = False
    HOBCA_mpc = False
    try_segment = False
    large = True

    ref_traj, param, obst, ob, ob_points = hybrid_a_star_initialization(large)

    if len(ref_traj.T) > 500:
        ref_traj = None

    if ref_traj is not None:

        ut = UTurnMPC()
        ut.set_parameters(param)
        ut.reserve_footprint = True
        shape = ut.get_car_shape()

        start_time = time.time()

        if iterative_mpc:
            op_dt, op_trajectories, op_controls = run_iterative_mpc(ref_traj, shape, obst)

        elif HOBCA_mpc:
            op_dt, op_trajectories, op_controls = run_HOBCA_mpc(ref_traj, shape, obst)

        else:
            if try_segment:
                op_dt, op_trajectories, op_controls = run_segment_OBCA_mpc(param, ref_traj, shape, obst)
            else:
                op_dt, op_trajectories, op_controls = run_TDROBCA_mpc(param, ref_traj, shape, obst)

        print("warm up OBCA total time:{:.3f}s".format(time.time() - start_time))
        ut.plot_results(op_dt, op_trajectories, op_controls, ref_traj, ob_points, four_states=True)

    else:
        print("Hybrid A Star initialization failed ....")


if __name__ == '__main__':
    main()
