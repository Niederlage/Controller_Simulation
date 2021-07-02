import numpy as np
import cv2
import matplotlib.pyplot as plt

def search_corner_downwards(edges, first_loc, xmax, ymax):
    search_up = edges[max(0, first_loc[0] - 1), first_loc[1]]
    search_down = edges[min(xmax, first_loc[0] + 1), first_loc[1]]

    if search_down != 0:
        next_loc = [min(xmax, first_loc[0] + 1, xmax), first_loc[1]]
        return search_corner_downwards(edges, next_loc, xmax, ymax)

    elif search_up != 0 and search_down == 0:
        end_loc = first_loc
        return end_loc

    elif search_up == 0 and search_down == 0:
        # print("this is horizontal edges! ")
        return first_loc
    else:
        print("no vaild start point for rightwards searching! ")
        return None


def search_corner_rightwards(edges, first_loc, xmax, ymax):
    search_right = edges[first_loc[0], min(ymax, first_loc[1] + 1)]
    search_left = edges[first_loc[0], max(0, first_loc[1] - 1)]

    if search_right != 0:
        next_loc = [first_loc[0], min(ymax, first_loc[1] + 1)]
        return search_corner_rightwards(edges, next_loc, xmax, ymax)

    elif search_left != 0 and search_right == 0:
        end_loc = first_loc
        return end_loc

    elif search_left == 0 and search_right == 0:
        # print("this is vertical edges! ")
        return first_loc
    else:
        print("no vaild start point for downwards searching! ")
        return None


def get_rough_coordinates(filename):
    img = cv2.imread(filename, 0)
    # edges = cv2.Canny(img, 1, 200)
    # dst = cv2.cornerHarris(img, 2, 3, 0.06)
    # edges = np.zeros(img.shape)
    # cv2.normalize(src=img, dst=edges, alpha=0, beta=255, norm_type=cv2.NORM_L2)
    img[img != 0] = -1
    edges = img + 1
    hmax = len(edges[0])
    wmax = len(edges[1])

    found_first_loc = False
    edges_list = []

    for i in range(hmax):
        first_loc = []
        for j in range(wmax):
            if found_first_loc:
                continue

            if edges[i, j] != 0 and not found_first_loc:
                right_end = search_corner_rightwards(edges, [i, j], hmax, wmax)
                down_end = search_corner_downwards(edges, [i, j], hmax, wmax)
                if right_end is not None and down_end is not None:
                    first_loc.append([i, j])
                    first_loc.append(right_end)
                    first_loc.append(down_end)
                    edges[i, j:right_end[1]] = 0
                    edges[i:down_end[0], j] = 0
                    found_first_loc = True

        if found_first_loc:
            edges_list.append(first_loc)
            found_first_loc = False
            continue

    edges_ = remove_identicals(edges_list)
    # edges_ = pixel2coordinates(edges_, 1, wmax, hmax)
    return edges_


def remove_identicals(edges):
    new_list = []
    for edge in edges:
        origin = np.array(edge[0])
        down = np.array(edge[1])
        right = np.array(edge[2])
        difference_d_r = (down - right)

        if len(new_list) == 0:
            if np.any(difference_d_r) != 0:
                new_list.append(edge)

        else:
            last_origin = np.array(new_list[-1][0])
            difference_e = np.abs(origin - last_origin)

            if np.any(difference_e > 1):
                if np.any(difference_d_r != 0):
                    new_list.append(edge)

    return new_list


def remove_identicals1(edges):
    new_list = []
    for edge in edges:
        origin = np.array(edge[0])
        down = np.array(edge[1])
        right = np.array(edge[2])
        difference_d_r = (down - right)

        if np.any(difference_d_r == 0):
            new_list.append(edge)

    return new_list


def pixel2coordinates(edges, resolution, xmax, ymax):
    rot = np.array([[0, 1], [-1, 0]])
    newlist = []
    trans = np.array([0, ymax])
    for edge in edges:
        a0 = (rot @ np.array(edge[0]) + trans) * resolution
        a1 = (rot @ np.array(edge[1]) + trans) * resolution
        a2 = (rot @ np.array(edge[2]) + trans) * resolution
        newlist.append([[a0[0], a0[1]], [a1[0], a1[1]], [a2[0], a2[1]]])

    return newlist


if __name__ == '__main__':

    convert2coordinates = True
    file = "../obstacles/fig1.png"
    edges_list = get_rough_coordinates(file)
    print("size edge list:", len(edges_list))
    img = cv2.imread(file, 0)
    fig = plt.figure()
    plt.imshow(img, cmap='gray')
    if convert2coordinates:
        for edge in edges_list:
            plt.plot(edge[0][1], edge[0][0], "bo")
            plt.plot(edge[1][1], edge[1][0], "ro")
            plt.plot(edge[2][1], edge[2][0], "go")

    plt.show()

    print(1)
