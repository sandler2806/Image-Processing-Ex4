
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import filters


def disparitySSD(img_l: np.ndarray, img_r: np.ndarray, disp_range: (int, int), k_size: int) -> np.ndarray:
    """
    img_l: Left image
    img_r: Right image
    range: Minimum and Maximum disparity range. Ex. (10,80)
    k_size: Kernel size for computing the SSD, kernel.shape = (k_size*2+1,k_size*2+1)

    return: Disparity map, disp_map.shape = Left.shape
    """
    min_shift_x = disp_range[0]
    max_shift_x = disp_range[1]
    disparity = np.zeros((img_l.shape[0], img_l.shape[1], max_shift_x - min_shift_x))
    filtered_img_left = filters.uniform_filter(img_l, k_size)
    filtered_img_right = filters.uniform_filter(img_r, k_size)
    diff_img_left = img_l - filtered_img_left
    diff_img_right = img_r - filtered_img_right

    for shift in range(min_shift_x, max_shift_x):
        right_rolled = np.roll(diff_img_right, shift)
        filtered_right = filters.uniform_filter(right_rolled ** 2, k_size)
        filtered_left = filters.uniform_filter(diff_img_left ** 2, k_size)

        L = diff_img_left / filtered_left
        R = right_rolled / filtered_right
        ssd = filters.uniform_filter(np.square(R - L), k_size) ** 2

        disparity[:, :, shift - min_shift_x] = ssd

    return np.argmin(disparity, axis=2)


def disparityNC(img_l: np.ndarray, img_r: np.ndarray, disp_range: (int, int), k_size: int) -> np.ndarray:
    """
    img_l: Left image
    img_r: Right image
    range: The Maximum disparity range. Ex. 80
    k_size: Kernel size for computing the NormCorolation, kernel.shape = (k_size*2+1,k_size*2+1)

    return: Disparity map, disp_map.shape = Left.shape
    """
    min_shift_x = disp_range[0]
    max_shift_x = disp_range[1]
    disparity = np.zeros((img_l.shape[0], img_l.shape[1], max_shift_x - min_shift_x))
    filtered_img_left = filters.uniform_filter(img_l, k_size)
    filtered_img_right = filters.uniform_filter(img_r, k_size)
    diff_img_left = img_l - filtered_img_left
    diff_img_right = img_r - filtered_img_right

    for shift in range(min_shift_x, max_shift_x):
        right_rolled = np.roll(diff_img_right, shift)

        filtered = filters.uniform_filter(diff_img_left * right_rolled, k_size)
        filtered_right = filters.uniform_filter(right_rolled ** 2, k_size)
        filtered_left = filters.uniform_filter(diff_img_left ** 2, k_size)

        nc = filtered / np.sqrt(filtered_right * filtered_left)
        disparity[:, :, shift - min_shift_x] = nc

    return np.argmax(disparity, axis=2)


def computeHomography(src_pnt: np.ndarray, dst_pnt: np.ndarray) -> (np.ndarray, float):
    """
    Finds the homography matrix, M, that transforms points from src_pnt to dst_pnt.
    returns the homography and the error between the transformed points to their
    destination (matched) points. Error = np.sqrt(sum((M.dot(src_pnt)-dst_pnt)**2))

    src_pnt: 4+ keypoints locations (x,y) on the original image. Shape:[4+,2]
    dst_pnt: 4+ keypoints locations (x,y) on the destenation image. Shape:[4+,2]

    return: (Homography matrix shape:[3,3], Homography error)
    """
    mat = []
    error = 0
    for src, dest in zip(src_pnt, dst_pnt):
        mat.append([src[0], src[1], 1, 0, 0, 0, -dest[0] * src[0], -dest[0] * src[1], -dest[0]])
        mat.append([0, 0, 0, src[0], src[1], 1, -dest[1] * src[0], -dest[1] * src[1], -dest[1]])

    svd = np.linalg.svd(mat)[2]

    homography = svd[8] / svd[8, 8]
    homography = homography.reshape(3, 3)

    for src, dest in zip(src_pnt, dst_pnt):
        src = np.append(src, 1)
        dest = np.append(dest, 1)
        error = np.sqrt(sum(homography @ src / (homography @ src)[2] - dest) ** 2)

    return homography, error


def warpImag(src_img: np.ndarray, dst_img: np.ndarray) -> None:
    """
    Displays both images, and lets the user mark 4 or more points on each image.
    Then calculates the homography and transforms the source image on to the destination image.
    Then transforms the source image onto the destination image and displays the result.

    src_img: The image that will be ’pasted’ onto the destination image.
    dst_img: The image that the source image will be ’pasted’ on.

    output: None.
    """

    dst_p = []
    fig1 = plt.figure()

    def onclick_1(event):
        x = event.xdata
        y = event.ydata
        print("Loc: {:.0f},{:.0f}".format(x, y))

        plt.plot(x, y, '*r')
        dst_p.append([x, y])

        if len(dst_p) == 4:
            plt.close()
        plt.show()

    # display image 1
    cid = fig1.canvas.mpl_connect('button_press_event', onclick_1)
    plt.imshow(dst_img)
    plt.show()
    dst_p = np.array(dst_p)

    ##### Your Code Here ######
    src_p = []
    fig2 = plt.figure()

    def onclick_2(event):
        x = event.xdata
        y = event.ydata
        print("Loc: {:.0f},{:.0f}".format(x, y))

        plt.plot(x, y, '*r')
        src_p.append([x, y])

        if len(src_p) == 4:
            plt.close()
        plt.show()

    # display image 2
    cid = fig2.canvas.mpl_connect('button_press_event', onclick_2)
    plt.imshow(src_img)
    plt.show()
    src_p = np.array(src_p)
    max_x = int(max([x[0] for x in src_p]))
    max_y = int(max([x[1] for x in src_p]))
    min_x = int(min([x[0] for x in src_p]))
    min_y = int(min([x[1] for x in src_p]))

    homography, _ = computeHomography(src_p, dst_p)

    src_out = np.zeros(dst_img.shape)
    for y in range(min_y, max_y):
        for x in range(min_x, max_x):
            src_point = np.array([x, y, 1])
            dest_point = homography @ src_point.T
            x_dest = int(dest_point[0] / dest_point[2])
            y_dest = int(dest_point[1] / dest_point[2])
            src_out[y_dest, x_dest] = src_img[y, x]
    mask = src_out == 0
    out = dst_img * mask + src_out * (1 - mask)
    plt.imshow(out)
    plt.show()
