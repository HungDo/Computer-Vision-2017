import cv2
from matplotlib import pyplot as plt
import numpy as np

images = [
    cv2.imread('selected/img0.jpg', cv2.COLOR_BGR2RGB),
    cv2.imread('selected/img1.jpg', cv2.COLOR_BGR2RGB),
    cv2.imread('selected/img2.jpg', cv2.COLOR_BGR2RGB),
    cv2.imread('selected/img3.jpg', cv2.COLOR_BGR2RGB),
    cv2.imread('selected/img4.jpg', cv2.COLOR_BGR2RGB)
]

gray_imgs = [
    cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY),
    cv2.cvtColor(images[1], cv2.COLOR_BGR2GRAY),
    cv2.cvtColor(images[2], cv2.COLOR_BGR2GRAY),
    cv2.cvtColor(images[3], cv2.COLOR_BGR2GRAY),
    cv2.cvtColor(images[4], cv2.COLOR_BGR2GRAY)
]

dx_images = [
    cv2.Sobel(gray_imgs[0], cv2.CV_64F, 1, 0, 5),
    cv2.Sobel(gray_imgs[1], cv2.CV_64F, 1, 0, 5),
    cv2.Sobel(gray_imgs[2], cv2.CV_64F, 1, 0, 5),
    cv2.Sobel(gray_imgs[3], cv2.CV_64F, 1, 0, 5),
    cv2.Sobel(gray_imgs[4], cv2.CV_64F, 1, 0, 5)
]

dy_images = [
    cv2.Sobel(gray_imgs[0], cv2.CV_64F, 0, 1, 5),
    cv2.Sobel(gray_imgs[1], cv2.CV_64F, 0, 1, 5),
    cv2.Sobel(gray_imgs[2], cv2.CV_64F, 0, 1, 5),
    cv2.Sobel(gray_imgs[3], cv2.CV_64F, 0, 1, 5),
    cv2.Sobel(gray_imgs[4], cv2.CV_64F, 0, 1, 5)
]

def findlines(lines, orig):
    modified = orig.copy()
    for line in lines:
        x0, y0, x1, y1 = line[0]
        cv2.line(modified, (x0, y0), (x1, y1), (0, 255, 0), 3, cv2.LINE_AA)
    return modified

def convert_to_xy(houghTransform):
    converted = []
    for line in houghTransform:
        rho, theta = line[0]

        a = np.cos(theta)
        b = np.sin(theta)

        x = int(a * rho)
        y = int(b * rho)

        # print x, y

        x0 = int(x + 2000 * (-b))
        y0 = int(y + 2000 * a)
        x1 = int(x - 2000 * (-b))
        y1 = int(y - 2000 * a)
        # print x0, y0, x1, y1

        converted.append([[x0, y0, x1, y1]])

        # try:
        #     m = (y1 - y0) / (x1 - x0)
        #     b = y1 - (m * x1)
        #
        #     # x = 1200
        #     ymax = m * 1200 + b
        #
        #     # x = 0
        #     ymin = b
        #
        #     # y = 1600
        #     xmax = (1600 - b)/m
        #
        #     # y = 0
        #     xmin = b/m
        #
        #     converted.append([[xmin, ymin, xmax, ymax]])
        #
        #     print [xmin, ymin, xmax, ymax]
        #     # converted.append([[x0, y0, x1, y1]])
        # except ZeroDivisionError:
        #     converted.append([[x0, 0, x1, 1200]])
    return converted

def filter_vert_lines(houghLines):
    filtered = []
    for line in houghLines:
        rho, theta = line[0]
        if np.abs(np.cos(theta)) < 0.12: # vertical filter
            continue
        filtered.append([line[0]])
    return filtered

def filter_horiz_lines(houghLines):
    filtered = []
    for line in houghLines:
        rho, theta = line[0]
        if np.abs(np.cos(theta)) > 0.996: # horiztonal filter
            continue
        filtered.append([line[0]])
    return filtered

def gradient_filter(houghLines, imageIdx):
    lines = []

    # filter using gradient and orientation


    # filter vertical lines
    # filter_vert = filter_vert_lines(houghLines)

    # filter horizontal lines
    # filter_horiz = filter_horiz_lines(filter_vert)

    for line in convert_to_xy(houghLines):
        x0, y0, x1, y1 = line[0]

        # dx0 = dx_images[imageIdx][x0][y0]
        # dy0 = dy_images[imageIdx][x0][y0]
        #
        # dx1 = dx_images[imageIdx][x1][y1]
        # dy1 = dy_images[imageIdx][x1][y1]
        # print dx0/dy0, dx1/dy1
        # if np.abs(dx0/dy0 - dx1/dy1) <

        lines.append([line[0]])
    return lines

def filter_lines(houghLines, imageIdx):
    lines = filter_vert_lines(houghLines)
    lines = filter_horiz_lines(lines)
    lines = gradient_filter(lines, imageIdx)
    return lines

def find_intersects(houghlines):
    intersect_pts = []

    for i in range(0, len(houghlines), 2):
        if i + 1 >= len(houghlines):
            continue

        x00, y00, x01, y01 = houghlines[i][0]
        x10, y10, x11, y11 = houghlines[i+1][0]

        try:
            m0 = (y01 - y00) / (x01 - x00)
            m1 = (y11 - y10) / (x11 - x10)

            b0 = y00 - (m0 * x00)
            b1 = y10 - (m1 * x10)

            x_int = (b1 - b0) / (m0 - m1)
            y_int = m0 * x_int + b0

            y2 = m1 * x_int + b1

            intersect_pts.append((x_int, y_int))

            # print x_int, y_int, y2
        except ZeroDivisionError:
            continue

    return intersect_pts

def apply_vanishing_pts(image, points):
    modified = image.copy()

    # print "Num of vanishing pts", len(points)
    for pt in points:
        cv2.circle(modified, pt, 30, (0, 255, 0), 3)
    return modified

ksizes = [(3, 3), (5, 5), (5, 5), (5, 5), (5, 5)]
sigmas = [0, 0, 0, 0, 0]

aperture = 3
min_threshold = [50, 50, 50, 50, 50]
max_threshold = [170, 170, 170, 170, 170]

canny = [
    cv2.Canny(images[0], min_threshold[0], max_threshold[0], aperture),
    cv2.Canny(images[1], min_threshold[1], max_threshold[1], aperture),
    cv2.Canny(images[2], min_threshold[2], max_threshold[2], aperture),
    cv2.Canny(images[3], min_threshold[3], max_threshold[3], aperture),
    cv2.Canny(images[4], min_threshold[4], max_threshold[4], aperture)
]

rho = 1
theta = np.pi/180
threshold = 255

houghTransform = [
    cv2.HoughLines(canny[0], rho, theta, threshold),
    cv2.HoughLines(canny[1], rho, theta, threshold),
    cv2.HoughLines(canny[2], rho, theta, threshold),
    cv2.HoughLines(canny[3], rho, theta, threshold),
    cv2.HoughLines(canny[4], rho, theta, threshold)
]

min_line_length = 4
max_line_gap = 8

houghTransformP = [
    cv2.HoughLinesP(canny[0], rho, theta, threshold, min_line_length, max_line_gap),
    cv2.HoughLinesP(canny[1], rho, theta, threshold, min_line_length, max_line_gap),
    cv2.HoughLinesP(canny[2], rho, theta, threshold, min_line_length, max_line_gap),
    cv2.HoughLinesP(canny[3], rho, theta, threshold, min_line_length, max_line_gap),
    cv2.HoughLinesP(canny[4], rho, theta, threshold, min_line_length, max_line_gap)
]

houghApplied = [
    findlines(convert_to_xy(houghTransform[0]), images[0]),
    findlines(convert_to_xy(houghTransform[1]), images[1]),
    findlines(convert_to_xy(houghTransform[2]), images[2]),
    findlines(convert_to_xy(houghTransform[3]), images[3]),
    findlines(convert_to_xy(houghTransform[4]), images[4])
]

houghPApplied = [
    findlines(houghTransformP[0], images[0]),
    findlines(houghTransformP[1], images[1]),
    findlines(houghTransformP[2], images[2]),
    findlines(houghTransformP[3], images[3]),
    findlines(houghTransformP[4], images[4])
]

# cv2.imwrite("canny_img_0.png", canny[0])
# cv2.imwrite("canny_img_1.png", canny[1])
# cv2.imwrite("canny_img_2.png", canny[2])
# cv2.imwrite("canny_img_3.png", canny[3])
# cv2.imwrite("canny_img_4.png", canny[4])
#
# cv2.imwrite('hough_lines_0.png', houghApplied[0])
# cv2.imwrite('hough_lines_1.png', houghApplied[1])
# cv2.imwrite('hough_lines_2.png', houghApplied[2])
# cv2.imwrite('hough_lines_3.png', houghApplied[3])
# cv2.imwrite('hough_lines_4.png', houghApplied[4])
#
cv2.imwrite('houghp_lines_0.png', houghPApplied[0])
cv2.imwrite('houghp_lines_1.png', houghPApplied[1])
cv2.imwrite('houghp_lines_2.png', houghPApplied[2])
cv2.imwrite('houghp_lines_3.png', houghPApplied[3])
cv2.imwrite('houghp_lines_4.png', houghPApplied[4])

filtered = [
    filter_lines(houghTransform[0], 0),
    filter_lines(houghTransform[1], 1),
    filter_lines(houghTransform[2], 2),
    filter_lines(houghTransform[3], 3),
    filter_lines(houghTransform[4], 4)
]

filterApplied = [
    findlines(filtered[0], images[0]),
    findlines(filtered[1], images[1]),
    findlines(filtered[2], images[2]),
    findlines(filtered[3], images[3]),
    findlines(filtered[4], images[4])
]

vanishingPoints = [
    find_intersects(filtered[0]),
    find_intersects(filtered[1]),
    find_intersects(filtered[2]),
    find_intersects(filtered[3]),
    find_intersects(filtered[4])
]

# print len(vanishingPoints[0])

appliedVanishingPoints = [
    apply_vanishing_pts(filterApplied[0], vanishingPoints[0]),
    apply_vanishing_pts(filterApplied[1], vanishingPoints[1]),
    apply_vanishing_pts(filterApplied[2], vanishingPoints[2]),
    apply_vanishing_pts(filterApplied[3], vanishingPoints[3]),
    apply_vanishing_pts(filterApplied[4], vanishingPoints[4])
]

# cv2.imwrite('vanishing_points_0.png', appliedVanishingPoints[0])
# cv2.imwrite('vanishing_points_1.png', appliedVanishingPoints[1])
# cv2.imwrite('vanishing_points_2.png', appliedVanishingPoints[2])
# cv2.imwrite('vanishing_points_3.png', appliedVanishingPoints[3])
# cv2.imwrite('vanishing_points_4.png', appliedVanishingPoints[4])

for i in range(0, 1):
    plt.figure(i)

    # plt.subplot(231), plt.imshow(images[i])
    # plt.title('Grayscale'), plt.xticks([]), plt.yticks([])
    #
    # plt.subplot(232), plt.plot(histograms[i])
    # plt.title('Histogram')

    plt.subplot(231), plt.imshow(filterApplied[i])
    plt.title('Filtered'), plt.xticks([]), plt.yticks([])

    plt.subplot(232), plt.imshow(appliedVanishingPoints[i])
    plt.title('Vanishing Points'), plt.xticks([]), plt.yticks([])

    plt.subplot(233), plt.imshow(canny[i], cmap="gray")
    plt.title('Canny Edge'), plt.xticks([]), plt.yticks([])

    plt.subplot(234), plt.imshow(houghApplied[i])
    plt.title('Hough Transform with Lines'), plt.xticks([]), plt.yticks([])

    plt.subplot(235), plt.imshow(houghPApplied[i])
    plt.title('Probablistic Hough with Lines'), plt.xticks([]), plt.yticks([])

plt.show()
