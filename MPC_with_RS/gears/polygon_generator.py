import numpy as np
import matplotlib.pyplot as plt


def cal_coeff_mat(vertices):
    edges = len(vertices)
    if edges <= 3:
        print("invalid! vertices <= 2")
        return None
    if np.all(vertices[0] == vertices[-1]):
        coeff = []
        for i in range(edges - 1):
            c_b = vertices[i + 1, 0] - vertices[i, 0]
            c_a = -vertices[i + 1, 1] + vertices[i, 1]
            c_c = c_a * vertices[i, 0] + c_b * vertices[i, 1]
            coeff.append(np.array([c_a, c_b, c_c]))
        return np.array(coeff)
    else:
        print("shape not close!")
        return None


def Euclidean_Transform(form, t_vec):
    yaw = t_vec[2]
    rot = np.array([[np.cos(yaw), -np.sin(yaw)],
                    [np.sin(yaw), np.cos(yaw)]])
    trans = t_vec[:2][:, None]
    rotted_list = rot @ form.T + np.repeat(trans, len(form), axis=1)
    return rotted_list.T


def Scale_Transformation(form, s_vec):
    scaled_formx = form[:, 0] * s_vec[0]
    scaled_formy = form[:, 1] * s_vec[1]
    return np.array([scaled_formx, scaled_formy]).T


def monte_carlo_sample_test(constraints):
    A_ = constraints[:, :2]
    b_ = constraints[:, 2]
    Num = int(5e3)
    rdn_list = []
    test_circle = False
    for i in range(Num):
        x_ = np.random.rand() * 30
        y_ = np.random.rand() * 30
        p_ = np.array([x_, y_])
        if test_circle:
            r = np.linalg.norm(x_)
            if r < 2:
                rdn_list.append(x_)
        else:
            error = A_ @ p_ - b_
            if np.all(error <= 0):
                rdn_list.append(p_)

    return np.array(rdn_list)


def get_polygon_map(use_sample_test=False):
    obst1 = np.array([[0, 0],
                      [0, 10],
                      [16, 10],
                      [16, 0],
                      [0, 0]])
    tf_1 = np.array([22, 0, 0])
    obst2 = Euclidean_Transform(obst1, tf_1)
    tf_2 = np.array([38 / 16, 1.])[:, None]
    obst3 = np.copy(Scale_Transformation(obst1, tf_2))
    obst3 = Euclidean_Transform(obst3, np.array([0, 16, 0]))
    # obst3 = Scale_Transformation(obst3, np.array([0.6, 1.1]))
    obst_ = [obst1, obst2, obst3]
    samplelist = []

    if use_sample_test:
        for ob_i in obst_:
            coeff_mat = cal_coeff_mat(ob_i)
            sample = monte_carlo_sample_test(coeff_mat)
            samplelist.append(sample)
        samples = np.block([samplelist[0].T, samplelist[1].T, samplelist[2].T]).T
    else:
        for ob_i in obst_:
            coeff_mat = cal_coeff_mat(ob_i)
            samplelist.append(coeff_mat)
        samples = np.block([samplelist[0].T, samplelist[1].T, samplelist[2].T]).T

    return obst_, samples


def get_point_map(obstmap, step_size):
    shape_list = []
    for obst in obstmap:
        plist = None
        for i in range(len(obst) - 1):

            points = interpolate_track(obst[i:i + 2, :], step_size)
            if i == 0:
                plist = points
            if i > 0:
                plist = np.vstack((plist, points))

        shape_list.append(plist)

    return shape_list


def interpolate_track(tracks, res):
    path = []
    dtracks = tracks[-1, :] - tracks[0, :]
    l_dist = np.hypot(dtracks[0], dtracks[1])
    l_path_size = int(np.ceil(l_dist / res))
    for i in range(l_path_size):
        if (i == 0):
            point = tracks[0, :]

        elif i == l_path_size:
            point = tracks[-1, :]
        else:
            point = tracks[0, :] + i * dtracks / l_path_size

        path.append(point)

    return np.array(path)


if __name__ == '__main__':
    # points = np.array([[0, 0],
    #                    [-1, 2],
    #                    [0, 3],
    #                    [2, 3],
    #                    [3, 3],
    #                    [0, 0]])
    #
    # tf_vec = np.array([1, 1, np.deg2rad(60)])
    # points_ = Euclidean_Transform(points, tf_vec)
    # c_mat = cal_coeff_mat(points_)
    # samples = monte_carlo_sample_test(c_mat)

    plot_obmap = True
    save_map = True

    if save_map:
        sample_test = False
    else:
        sample_test = True

    obmap, samples = get_polygon_map(use_sample_test=sample_test)
    pointmap = get_point_map(obmap, 0.5)

    if save_map:
        pointmap_ = np.block([pointmap[0].T, pointmap[1].T, pointmap[2].T]).T
        print("total points num:", len(pointmap_))
        np.savez("../saved_obmap.npz", polygons=obmap, constraint_mat=samples, pointmap=pointmap_)
        print(1)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.cla()

    if sample_test:
        if len(samples) > 0:
            ax.plot(samples[:, 0], samples[:, 1], "ro")
        else:
            print("no samples found!")

    if plot_obmap:
        for ob_ in obmap:
            ax.plot(ob_[:, 0], ob_[:, 1], "b-")
        for ob_ in pointmap:
            ax.plot(ob_[:, 0], ob_[:, 1], "o", color="black")
        ax.plot(15, 6, "gx", label="start")
        ax.plot(10, 2, "rx", label="goal")
    ax.grid()
    plt.axis("equal")
    plt.show()
