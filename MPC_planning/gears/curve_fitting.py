import numpy as np
import scipy.linalg as sl
import matplotlib.pyplot as plt


def A_(n, m):
    if m == 1:
        return n
    return n * A_(n - 1, m - 1)


def cal_coefficient(refpath):
    Ns = len(refpath.T)
    num = int(Ns / 4.5)
    # spath = np.hstack((refpath[:2, ::num], refpath[:2, -1][:, None]))
    s = get_s_(refpath[:2, :])
    spath = np.append(s[::num], s[-1])
    # S = get_S_mat(spath)
    S = get_S_mat(s)

    # ct = sl.solve(Q_, refpath[2, ::int(len(refpath.T)/5)])
    # xpath = np.append(refpath[0, ::num], refpath[0, -1])
    # ypath = np.append(refpath[1, ::num], refpath[1, -1])
    # tpath = np.append(refpath[2, ::num], refpath[2, -1])

    xpath = refpath[0, :]
    ypath = refpath[1, :]
    tpath = refpath[2, :]
    cx = solve_coeff(S, xpath)
    cy = solve_coeff(S, ypath)
    ct = solve_coeff(S, tpath)

    return cx, cy, ct, s


def solve_coeff(S, vals):
    Q_ = S.T @ S
    # ct = sl.solve(Q_, refpath[2, ::int(len(refpath.T)/5)])
    ct = sl.solve(Q_, S.T @ vals)
    # ct = sl.solve(S, vals)
    return ct


def get_s_(path0):
    dpath = np.diff(path0, axis=1)
    s2 = np.square(dpath)
    s_ = np.cumsum(np.sqrt(np.sum(s2, axis=0)))
    s_ = np.append(0., s_)
    return s_


def get_S_mat(s_):
    S = np.zeros((len(s_), 6))
    S[:, 0] = 1.
    for i in range(1, 6):
        S[:, i] = np.power(s_, i)

    return S


def get_curve(ct, res, ends):
    s_ = np.arange(0, ends, res)
    yaw = ct[0] + ct[1] * s_ + ct[2] * np.power(s_, 2) + ct[3] * np.power(s_, 3) + ct[4] * np.power(s_, 4) + ct[
        5] * np.power(s_, 5)
    return yaw


def cal_path_use_theta(start, yaw, res, ends):
    s_ = np.arange(0, ends, res)
    x = np.cumsum(res * np.cos(yaw))
    y = np.cumsum(res * np.sin(yaw))
    return x + start[0], y + start[1]


if __name__ == '__main__':
    res = 0.01
    # load = np.load("../data/saved_hybrid_a_star.npz")
    # ref_traj = load["saved_traj"]
    load = np.load("../data/smoothed_traj.npz")
    ref_traj = load["refpath"]

    np.savetxt("path.txt", ref_traj[:2, :].T)
    theta = ref_traj[2, :]
    coeffx, coeffy, coefft, s0 = cal_coefficient(ref_traj)
    yaw_ = get_curve(coefft, res, s0[-1])
    x_ = get_curve(coeffx, res, s0[-1])
    y_ = get_curve(coeffy, res, s0[-1])
    x, y = cal_path_use_theta(ref_traj[:2, 0], yaw_, res, s0[-1])


    s_ = np.arange(0, s0[-1], res)
    print("ct:", coefft, "\ncx:", coeffx, "\ncy:", coeffy)

    f = plt.figure()
    ax = plt.subplot(211)
    ax.plot(s_, np.rad2deg(yaw_), "-r", label="yaw")
    ax.grid(True)
    ax.legend()
    ax.set_xlabel("line length[m]")
    ax.set_ylabel("yaw angle[deg]")

    ax = plt.subplot(212)
    ax.plot(np.rad2deg(theta), "-r", label="yaw_ref")
    ax.grid(True)
    ax.legend()
    ax.set_xlabel("line length[m]")
    ax.set_ylabel("yaw angle[deg]")

    f2 = plt.figure()
    ax = plt.subplot(111)
    ax.plot(x_, y_, "-r", label="curve from s")
    ax.plot(x, y, "-b", label="curve from theta")
    ax.plot(ref_traj[0, :], ref_traj[1, :], "-g", label="refpath")
    ax.grid(True)
    ax.legend()
    plt.axis("equal")
    plt.show()
