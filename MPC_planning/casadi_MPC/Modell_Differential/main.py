import time

import numpy as np
import yaml
from mpc_motion_plot import UTurnMPC
from casadi_differ_reference_line import CasADi_MPC_differ
from MPC_planning.HybridAStar.hybrid_a_star import HybridAStar


def hybrid_a_star_initialization(large):
    if large:
        address = "../../config_OBCA_large.yaml"
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


def get_all_obsts():
    loadmap = np.load("../../data/saved_obmap.npz", allow_pickle=True)
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
    obst.append(ob_constraint_mat[:4, :])
    obst.append(ob_constraint_mat[4:8, :])
    obst.append(ob_constraint_mat[8:12, :])
    return ob, obst, obst_points


def main():
    address = "../../config_OBCA_large.yaml"
    load_file = False
    large = True

    if not load_file:
        ref_traj, param, obst, ob_points = hybrid_a_star_initialization(large)

        if len(ref_traj.T) > 500:
            ref_traj = None

        if ref_traj is not None:

            ut = UTurnMPC()
            ut.set_parameters(param)
            ut.reserve_footprint = True

            start_time = time.time()
            # states: (x ,y ,theta ,v , steer, a, steer_rate, jerk)
            cmpc = CasADi_MPC_differ()

            ref_traj, ob, obst = ut.initialize_saved_data()
            cmpc.init_model_reference_line(ref_traj)
            op_dt, op_trajectories, op_controls = cmpc.get_result_reference_line()

            print("warm up OBCA total time:{:.3f}s".format(time.time() - start_time))
            np.savez("../../data/smoothed_traj_differ", dt=op_dt, traj=op_trajectories, control=op_controls, refpath=ref_traj)

            ut.plot_results(op_dt, op_trajectories, op_controls, ref_traj, ob_points, four_states=True)

        else:
            print("Hybrid A Star initialization failed ....")

    else:
        loads = np.load("../../data/smoothed_traj_differ.npz")
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
