import numpy as np
import matplotlib.pyplot as plt


class Obstacles:
    def __init__(self):
        self.resolution = 0.2
        self.bounds_left_down = [-2., -5.5]
        self.bounds_right_up = [4.5, 5.5]
        # self.bounds_left_down = [-15., -5.]
        # self.bounds_right_up = [15., 20.]
        self.bounds = self.get_bounds()

        self.obst_keypoints = None
        self.obst_pointmap = None
        self.coeff_mat = None
        self.samples = []

        self.show_obstacles = True
        self.show_bounds = True
        self.sample_test = False
        self.save_map_as_fig = False

    def cal_coeff_mat(self, vertices):
        edges = len(vertices)
        if edges <= 3:
            print("invalid! vertices <= 2")
            return None

        if np.all(vertices[0] == vertices[-1]):
            coeff = []

            x1 = vertices[0, 0]
            y1 = vertices[0, 1]
            x2 = vertices[1, 0]
            y2 = vertices[1, 1]
            if x1 * y2 - x2 * y1 > 0:
                clockwise = 1
            else:
                clockwise = -1

            for i in range(edges - 1):
                c_b = (vertices[i + 1, 0] - vertices[i, 0]) * -1 * clockwise  # x2 - x1
                c_a = (vertices[i + 1, 1] - vertices[i, 1]) * clockwise  # y2 - y1
                c_c = c_a * vertices[i + 1, 0] + c_b * vertices[i + 1, 1]
                if c_b == 0.:
                    c_c /= abs(c_a)
                    c_a /= abs(c_a)
                    c_b = 0.
                elif c_a == 0.:
                    c_c /= abs(c_b)
                    c_b /= abs(c_b)
                    c_a = 0.
                coeff.append(np.array([c_a, c_b, c_c]))
            return np.array(coeff)
        else:
            print("shape not close!")
            return None

    def Euclidean_Transform(self, form, t_vec):
        yaw = t_vec[2]
        rot = np.array([[np.cos(yaw), -np.sin(yaw)],
                        [np.sin(yaw), np.cos(yaw)]])
        trans = t_vec[:2][:, None]
        rotted_list = rot @ form.T + np.repeat(trans, len(form), axis=1)
        return rotted_list.T

    def Scale_Transformation(self, form, s_vec):
        scaled_formx = form[:, 0] * s_vec[0]
        scaled_formy = form[:, 1] * s_vec[1]
        return np.array([scaled_formx, scaled_formy]).T

    def monte_carlo_sample_test(self, constraints):
        A_ = constraints[:, :2]
        b_ = constraints[:, 2]
        Num = int(5e3)
        rdn_list = []
        test_circle = False
        for i in range(Num):
            x_ = np.random.rand() * 50 - 25
            y_ = np.random.rand() * 50 - 25
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

    def generate_polygon_map1(self):

        obst1 = np.array([[-10, 6.],
                          [-3.5, 6.],
                          [-3.5, 0.],
                          [-10., 0.],
                          [-10., 6.]])

        tf_1 = np.array([13.5, 0, 0])
        obst2 = self.Euclidean_Transform(obst1, tf_1)
        tf_2 = np.array([15 / 10, 0.5])[:, None]
        obst3 = np.copy(self.Scale_Transformation(obst1, tf_2))
        obst3 = self.Euclidean_Transform(obst3, np.array([10, 10, 0]))
        tf_3 = np.array([12 / 10, 1.])[:, None]
        obst4 = np.copy(self.Scale_Transformation(obst1, tf_3))
        obst4 = self.Euclidean_Transform(obst4, np.array([5, 18, 0]))

        obst_ = [obst1, obst2, obst3, obst4]
        mat_list = []

        for i, ob_i in enumerate(obst_):
            coeff_mat = self.cal_coeff_mat(ob_i)
            mat_list.append(coeff_mat)

            if self.sample_test:
                sample = self.monte_carlo_sample_test(coeff_mat)
                self.samples.append(sample)

        self.obst_keypoints = obst_
        self.coeff_mat = mat_list
        self.obst_pointmap = self.get_point_map(obst_, 0.2)

    def generate_polygon_map(self):
        xoffset = 3.
        yoffset = -0.55

        obst1 = np.array([[xoffset, yoffset],
                          [xoffset + 1.3, yoffset],
                          [xoffset + 1.3, yoffset + .05],
                          [xoffset, yoffset + .05],
                          [xoffset + .8, yoffset]])

        tf_1 = np.array([0, 1.05, 0])
        obst2 = self.Euclidean_Transform(obst1, tf_1)

        tf_2 = np.array([15 / 10, 0.5])[:, None]
        # obst3 = np.copy(self.Scale_Transformation(obst1, tf_2))
        # obst3 = self.Euclidean_Transform(obst3, np.array([10, 10, 0]))
        # tf_3 = np.array([12 / 10, 1.])[:, None]
        # obst4 = np.copy(self.Scale_Transformation(obst1, tf_3))
        # obst4 = self.Euclidean_Transform(obst4, np.array([5, 18, 0]))

        obst_ = [obst1, obst2]
        mat_list = []

        for i, ob_i in enumerate(obst_):
            coeff_mat = self.cal_coeff_mat(ob_i)
            mat_list.append(coeff_mat)

            if self.sample_test:
                sample = self.monte_carlo_sample_test(coeff_mat)
                self.samples.append(sample)

        self.obst_keypoints = obst_
        self.coeff_mat = mat_list
        self.obst_pointmap = self.get_point_map(obst_, 0.1)

    def get_point_map(self, obstmap, step_size):
        shape_list = []
        for obst in obstmap:
            plist = None
            for i in range(len(obst) - 1):

                points = self.interpolate_track(obst[i:i + 2, :], step_size)
                if i == 0:
                    plist = points
                if i > 0:
                    plist = np.vstack((plist, points))

            shape_list.append(plist)

        return shape_list

    def interpolate_track(self, tracks, res):
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

    def get_bounds(self):

        pa = self.bounds_left_down
        pb = self.bounds_right_up
        dis_x = np.arange(pa[0], pb[0], self.resolution)
        # dis_x = np.append(dis_x, -pa[0])
        dis_y = np.arange(pa[1], pb[1], self.resolution)
        # dis_y = np.append(dis_y, -pa[1])

        bound_left = np.vstack([[pa[0]] * len(dis_y), dis_y])
        bound_right = np.vstack([[pb[0]] * len(dis_y), dis_y])
        bound_down = np.vstack([dis_x, [pa[1]] * len(dis_x)])
        bound_up = np.vstack([dis_x, [pb[1]] * len(dis_x)])
        return np.block([bound_up[:, ::-1], bound_left, bound_down, bound_right]).T

    def plot_obst(self, ax):
        # for ob_ in self.obst_keypoints:
        #     ax.plot(ob_[:, 0], ob_[:, 1], "rx")
        if self.obst_pointmap is not None and self.show_obstacles:
            for ob_ in self.obst_pointmap:
                ax.plot(ob_[:, 0], ob_[:, 1], color="black")
        if self.show_bounds:
            ax.plot(self.bounds[:, 0], self.bounds[:, 1], ".-", color="firebrick")

    def plot_obst_on_fig(self):
        if self.save_map_as_fig:
            fig = plt.figure(figsize=(3, 3), dpi=100)
        else:
            fig = plt.figure()

        ax = fig.add_subplot(111)
        ax.cla()
        self.plot_obst(ax)
        ax.grid()
        plt.axis("equal")

        if self.sample_test:
            for sample in self.samples:
                if len(sample) > 0:
                    ax.plot(sample[:, 0], sample[:, 1], "r.")
            else:
                print("no samples found!")

        if self.save_map_as_fig:
            plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
            plt.axis("equal")
            plt.axis('off')
            plt.savefig("fig1.png")

    def save_obmap(self, obcamap=False):

        if obcamap:
            savepath = "../data/saved_obmap_obca.npz"
        else:
            savepath = "../data/saved_obmap.npz"

        np.savez(savepath,
                 constraint_mat=self.coeff_mat,
                 pointmap=self.obst_pointmap,
                 bounds=self.bounds,
                 polygons=self.obst_keypoints)


if __name__ == '__main__':

    plot_obmap = True
    save_map = True

    use_obca_map = True

    if save_map:
        sample_test = False
    else:
        sample_test = True

    obst = Obstacles()
    if use_obca_map:
        obst.bounds_left_down = [-11., 3.]
        obst.bounds_right_up = [11., 20.]
        obst.bounds = obst.get_bounds()
        obst.generate_polygon_map1()
    else:
        obst.generate_polygon_map()

    obst.save_obmap(obcamap=use_obca_map)
    obst.plot_obst_on_fig()
    plt.show()
