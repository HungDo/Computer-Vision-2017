import cv2
from matplotlib import pyplot as plt
import numpy as np

# Figure 0
#        - peaks at 50 - 60 and ~242
#
# Figure 1
#        - peaks at 50 - 125 and ~240
#
# Figure 2
#        - peaks at 25, 45, 120 - 140, 240
#
# Figure 3
#        - peaks at 30 - 40, 100 - 150, 240
#
# Figure 4
#        - peaks at 50 - 75, 240

images = [
    cv2.imread('selected/img0.jpg', cv2.COLOR_BGR2RGB),
    cv2.imread('selected/img1.jpg', cv2.COLOR_BGR2RGB),
    cv2.imread('selected/img2.jpg', cv2.COLOR_BGR2RGB),
    cv2.imread('selected/img3.jpg', cv2.COLOR_BGR2RGB),
    cv2.imread('selected/img4.jpg', cv2.COLOR_BGR2RGB)
]

def findlines(lines, orig):
    modified = orig.copy()
    for line in lines:
        x0, y0, x1, y1 = line[0]
        cv2.line(modified, (x0, y0), (x1, y1), (255, 0, 0), 3, cv2.LINE_AA)
    return modified

def convert_to_xy(houghTransform):
    lineLength = 2000
    converted = []
    for line in houghTransform:
        rho, theta = line[0]

        a = np.cos(theta)
        b = np.sin(theta)

        x0 = int(a * rho)
        y0 = int(b * rho)

        pt1x = int(x0 + lineLength * (-b))
        pt1y = int(y0 + lineLength * a)
        pt2x = int(x0 - lineLength * (-b))
        pt2y = int(y0 - lineLength * a)

        converted.append([[pt1x, pt1y, pt2x, pt2y]])
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

def filter_lines(houghLines):
    lines = []

    # filter using gradient and orientation

    # filter vertical lines
    filter_vert = filter_vert_lines(houghLines)

    # filter horizontal lines
    filter_horiz = filter_horiz_lines(filter_vert)

    for line in convert_to_xy(filter_horiz):
        x0, y0, x1, y1 = line[0]

        if (x0 - x1) == 0:
            continue

        m0 = (y0 - y1) / (x0 - x1)

        lines.append([line[0]])
    return lines

def find_intersects(houghlines, w, h):
    intersect_pts = []

    for first in houghlines:
        for second in houghlines:
            if first[0] == second[0]:
                continue

            x00, y00, x01, y01 = first[0]

            x10, y10, x11, y11 = second[0]

            if (x01 - x00) == 0 or (x11 - x10) == 0:
                continue

            m0 = (y01 - y00) / (x01 - x00)
            m1 = (y11 - y10) / (x11 - x10)

            if (m0 - m1) == 0 or x00 == 0 or x10 == 0:
                continue

            b0 = y00 - (m0 / x00)
            b1 = y10 - (m1 / x10)

            xinter = (b1 - b0) / (m0 - m1)
            yinter = (m0 * xinter) + b0

            if xinter >= 0 and xinter <= w and yinter >= 0 and yinter <= h:
                # print xinter, yinter
                intersect_pts.append((xinter, yinter))

    return intersect_pts

def apply_vanishing_pts(image, points):
    modified = image.copy()
    for pt in points:
        cv2.circle(modified, pt, 30, (255, 0, 0), 3)
    return modified

ksizes = [(3, 3), (5, 5), (5, 5), (5, 5), (5, 5)]
sigmas = [0, 0, 0, 0, 0]

aperture = 3
min_threshold = [50, 50, 50, 50, 50]
max_threshold = [170, 170, 170, 170, 170]

min_thresh_smoothed = [0,0,0,0,0]
max_thresh_smoothed = [255,255,255,255,255]

canny = [
    cv2.Canny(images[0], min_threshold[0], max_threshold[0], aperture),
    cv2.Canny(images[1], min_threshold[1], max_threshold[1], aperture),
    cv2.Canny(images[2], min_threshold[2], max_threshold[2], aperture),
    cv2.Canny(images[3], min_threshold[3], max_threshold[3], aperture),
    cv2.Canny(images[4], min_threshold[4], max_threshold[4], aperture)
]

rho = 1
theta = np.pi/180
threshold = 250

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

line_length = 2000

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

filtered = [
    filter_lines(houghTransform[0]),
    filter_lines(houghTransform[1]),
    filter_lines(houghTransform[2]),
    filter_lines(houghTransform[3]),
    filter_lines(houghTransform[4])
]

filterApplied = [
    findlines(filtered[0], images[0]),
    findlines(filtered[1], images[1]),
    findlines(filtered[2], images[2]),
    findlines(filtered[3], images[3]),
    findlines(filtered[4], images[4])
]

vanishingPoints = [
    find_intersects(filtered[0], len(images[0][0]), len(images[0])),
    find_intersects(filtered[1], len(images[1][0]), len(images[1])),
    find_intersects(filtered[2], len(images[2][0]), len(images[2])),
    find_intersects(filtered[3], len(images[3][0]), len(images[3])),
    find_intersects(filtered[4], len(images[4][0]), len(images[4]))
]

print len(vanishingPoints[0])

appliedVanishingPoints = [
    apply_vanishing_pts(filterApplied[0], vanishingPoints[0]),
    apply_vanishing_pts(filterApplied[1], vanishingPoints[1]),
    apply_vanishing_pts(filterApplied[2], vanishingPoints[2]),
    apply_vanishing_pts(filterApplied[3], vanishingPoints[3]),
    apply_vanishing_pts(filterApplied[4], vanishingPoints[4])
]

# for i in range(0, 1):
#     plt.figure(i)
#
#     # plt.subplot(231), plt.imshow(images[i])
#     # plt.title('Grayscale'), plt.xticks([]), plt.yticks([])
#     #
#     # plt.subplot(232), plt.plot(histograms[i])
#     # plt.title('Histogram')
#
#     plt.subplot(231), plt.imshow(filterApplied[i])
#     plt.title('Filtered'), plt.xticks([]), plt.yticks([])
#
#     plt.subplot(232), plt.imshow(appliedVanishingPoints[i])
#     plt.title('Vanishing Points'), plt.xticks([]), plt.yticks([])
#
#     plt.subplot(233), plt.imshow(canny[i], cmap="gray")
#     plt.title('Canny Edge'), plt.xticks([]), plt.yticks([])
#
#     plt.subplot(234), plt.imshow(houghApplied[i])
#     plt.title('Hough Transform with Lines'), plt.xticks([]), plt.yticks([])
#
#     plt.subplot(235), plt.imshow(houghPApplied[i])
#     plt.title('Probablistic Hough with Lines'), plt.xticks([]), plt.yticks([])
#
# plt.show()



