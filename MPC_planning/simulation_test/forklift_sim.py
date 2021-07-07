import time
import numpy as np
import yaml
from gears.cubic_spline_planner import Spline2D
from motion_plot.ackermann_motion_plot import UTurnMPC
# from motion_plot.differ_motion_plot import UTurnMPC
from MPC_planning.HybridAStar.hybrid_a_star import HybridAStar
from Modell_Ackermann.casadi_forklift import CasADi_MPC_reference_line_fortklift
from Modell_Differential.casadi_differ_reference_line_forklift import CasADi_MPC_differ_forklift
from Modell_Differential.casadi_differ_TDROBCA import CasADi_MPC_differ_TDROBCA
import matplotlib.pyplot as plt


def initialize(address):
    with open(address, 'r', encoding='utf-8') as f:
        param = yaml.load(f)

    sx = param["start"]
    ex = param["goal"]
    start = np.array([sx[0], sx[1], sx[2] * np.pi / 180])
    goal = np.array([ex[0], ex[1], ex[2] * np.pi / 180])
    # if goal[0] - start[0] < 0 and abs(start[2]) > np.pi / 2:
    #     start[2] = normalize_angle(start[2])
    #     goal[2] = normalize_angle(goal[2])

    print("start:", start, "\ngoal:", goal)
    return start, goal, param


def plot_tf_curves(start, goal, refpath):
    tf_path = coordinate_transform(start[2], start[:2], refpath, "world to body")
    tf_back_path = coordinate_transform(start[2], start[:2], tf_path, "body to world")
    length = 0.8
    width = 0.4

    f = plt.Figure()
    plt.arrow(start[0], start[1], length * np.cos(start[2]), length * np.sin(start[2]),
              fc="r", ec="k", head_width=width, head_length=width, alpha=0.4)
    plt.arrow(goal[0], goal[1], length * np.cos(goal[2]), length * np.sin(goal[2]),
              fc="g", ec="k", head_width=width, head_length=width, alpha=0.4)

    plot_curve(refpath, "spline interpolation")
    plot_curve(tf_path, "spline tf")
    plot_curve(tf_back_path, "spline tf back")
    plt.axis("equal")
    plt.grid()
    plt.legend()
    plt.show()


def plot_arrow(refpath, p1, fc):
    length = 0.8
    width = 0.4
    plt.arrow(refpath[0, p1],
              refpath[1, p1],
              length * np.cos(refpath[2, p1]),
              length * np.sin(refpath[2, p1]),
              fc="b", ec="k",
              head_width=width, head_length=width,
              alpha=0.4)


def plot_curve(refpath, curve_name):
    plt.plot(refpath[0, :], refpath[1, :], label=curve_name)
    p1 = int(len(refpath.T) * 0.2)
    p2 = int(len(refpath.T) * 0.4)
    p3 = int(len(refpath.T) * 0.6)
    p4 = int(len(refpath.T) * 0.8)

    plot_arrow(refpath, 0, "r")
    plot_arrow(refpath, p1, "b")
    plot_arrow(refpath, p2, "b")
    plot_arrow(refpath, p3, "b")
    plot_arrow(refpath, p4, "b")
    plot_arrow(refpath, -1, "g")


def set_control_points(start, goal):
    lds = 0.5
    start2 = np.array([start[0] + lds * np.cos(start[2]), start[1] + lds * np.sin(start[2])])
    goal2 = np.array([goal[0] + lds * np.cos(goal[2]), goal[1] + lds * np.sin(goal[2])])

    if np.linalg.norm(start2 - goal[:2]) > np.linalg.norm(start[:2] - goal[:2]):
        start2 = np.array([start[0] - lds * np.cos(start[2]), start[1] - lds * np.sin(start[2])])
    #
    if np.linalg.norm(start[:2] - goal2) > np.linalg.norm(start[:2] - goal[:2]):
        goal2 = np.array([goal[0] - lds * np.cos(start[2]), goal[1] - lds * np.sin(start[2])])

    x = [start[0], start2[0], goal2[0], goal[0]]
    y = [start[1], start2[1], goal2[1], goal[1]]
    return x, y


def spline_reference(start, goal, ds):
    is_reverse = True

    x, y = set_control_points(start, goal)
    sp = Spline2D(x, y)
    s = np.arange(0, sp.s[-1], ds)

    rx, ry, ryaw, rk = [], [], [], []

    for i_s in s:
        ix, iy = sp.calc_position(i_s)
        rx.append(ix)
        ry.append(iy)
        yaw_ = sp.calc_yaw(i_s)
        ryaw.append(yaw_)
        rk.append(sp.calc_curvature(i_s))

    refpath = np.array([rx, ry, ryaw])
    if is_reverse:
        refpath[2, :] += np.pi
        for i in range(len(refpath.T)):
            refpath[2, i] = normalize_angle(refpath[2, i])

    refpath[:, 0] = start
    refpath[:, -1] = goal

    return refpath


def hybrid_a_star_reference(start, goal, param, res):
    coarse_planner = HybridAStar()
    coarse_planner.MOTION_RESOLUTION = res
    coarse_planner.show_animation = False
    coarse_planner.car.set_parameters(param)
    coarse_planner.obmap.generate_polygon_map()
    obst = coarse_planner.generate_obmap()
    path = coarse_planner.hybrid_a_star_planning(start, goal, obst)

    if path is not None:
        x = path.x_list
        y = path.y_list
        yaw = path.yaw_list
        saved_path = np.array([x, y, yaw])
    else:
        saved_path = None

    return saved_path


def coordinate_transform(yaw, t, path, mode):
    yaw = normalize_angle(yaw)
    yawlist = np.array([normalize_angle(i) for i in path[2, :]])

    rot = np.array([[np.cos(yaw), -np.sin(yaw)],
                    [np.sin(yaw), np.cos(yaw)]])
    trans = np.repeat(t[:, None], len(path.T), axis=1)

    if mode == "body to world":
        newpath = rot @ path[:2, :] + trans
        newyaw = yawlist + yaw
    else:
        newpath = rot.T @ (path[:2, :] - trans)
        newyaw = yawlist - yaw

    newyaw = [normalize_angle(i) for i in newyaw]

    return np.vstack((newpath, newyaw))


def normalize_angle(yaw):
    return (yaw + np.pi) % (2 * np.pi) - np.pi


def run_reference_line_mpc(param, ref_traj):
    warmup_time = time.time()
    forkplan = CasADi_MPC_reference_line_fortklift()
    forkplan.set_parameters(param)
    forkplan.init_model_reference_line(ref_traj)
    op_dt, op_trajectories, op_controls = forkplan.get_result_reference_line()

    return op_dt, op_trajectories, op_controls


def run_reference_line_mpc_forklift(param, ref_traj):
    warmup_time = time.time()
    # forkplan = CasADi_MPC_reference_line_fortklift()
    forkplan = CasADi_MPC_differ_forklift()
    forkplan.set_parameters(param)
    forkplan.init_model_reference_line(ref_traj[:, 0], ref_traj)
    op_dt, op_trajectories, op_controls = forkplan.get_result_reference_line()

    return op_dt, op_trajectories, op_controls


def main():
    address = "../config_forklift.yaml"
    load_file = False
    ds = 0.2
    start, goal, param = initialize(address)

    ut = UTurnMPC()
    ut.obmap.generate_polygon_map()
    ut.car.model_type = param["use_model_type"]
    ut.car.set_parameters(param)
    ut.reserve_footprint = True
    ut.use_Runge_Kutta = False
    start_time = time.time()
    ut.obmap.show_obstacles = True
    if start[0] < -4.:
        ref_traj = spline_reference(start, goal, ds)
    else:
        ref_traj = hybrid_a_star_reference(start, goal, param, ds)

    plot_tf_curves(start, goal, ref_traj)
    tf_path = coordinate_transform(start[2], start[:2], ref_traj, "world to body")

    if ref_traj is not None:

        # op_dt, op_trajectories, op_controls = run_reference_line_mpc(param, ref_traj)
        op_dt, op_trajectories, op_controls = run_reference_line_mpc_forklift(param, tf_path)
        tf_back_oppath = coordinate_transform(start[2], start[:2], op_trajectories, "body to world")
        print(" total time:{:.3f}s".format(time.time() - start_time))

        ut.plot_results(op_dt, tf_back_oppath, op_controls, ref_traj, four_states=True)
        # ut.plot_results_differ(op_dt, op_trajectories, op_controls, ref_traj, four_states=True)

    else:
        print("Hybrid A Star initialization failed ....")


if __name__ == '__main__':
    main()
