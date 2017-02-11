import cv2
import numpy as np

# 2
orig = cv2.imread('orig.jpg', cv2.IMREAD_COLOR)
grayscale = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
cv2.imwrite('gray.png', grayscale)

# 10
median5x5 = cv2.medianBlur(orig, 5)  # 5x5
median3x3 = cv2.medianBlur(orig, 3)  # 3x3
median5x5g = cv2.medianBlur(grayscale, 5)  # 5x5
median3x3g = cv2.medianBlur(grayscale, 3)  # 3x3
gaussian1 = cv2.GaussianBlur(orig, (5, 5), 1)
gaussian2 = cv2.GaussianBlur(orig, (5, 5), 2)
gaussian3 = cv2.GaussianBlur(orig, (5, 5), 3)
gaussian1g = cv2.GaussianBlur(grayscale, (5, 5), 1)
gaussian2g = cv2.GaussianBlur(grayscale, (5, 5), 2)
gaussian3g = cv2.GaussianBlur(grayscale, (5, 5), 3)

# derivative sobel

dx = cv2.Sobel(orig, cv2.CV_64F, 1, 0, 5)
dy = cv2.Sobel(orig, cv2.CV_64F, 0, 1, 5)

dxg = cv2.Sobel(grayscale, cv2.CV_64F, 1, 0, 5)
dyg = cv2.Sobel(grayscale, cv2.CV_64F, 0, 1, 5)

# 2
# magnitude of original and gray
magnitude_gray = np.sqrt((dxg * dxg) + (dyg * dyg))
magnitude_orig = np.sqrt((dx * dx) + (dy * dy))

# 2*3 = 6
# magnitude of smoothing (orig and gray)
# orig
dxg1 = cv2.Sobel(gaussian1, cv2.CV_64F, 1, 0, 5)
dyg1 = cv2.Sobel(gaussian1, cv2.CV_64F, 0, 1, 5)
orig_mag_gauss1 = np.sqrt((dxg1 * dxg1) + (dyg1 * dyg1))
cv2.imwrite('orig_mag_gauss1.png', orig_mag_gauss1)

dxg2 = cv2.Sobel(gaussian2, cv2.CV_64F, 1, 0, 5)
dyg2 = cv2.Sobel(gaussian2, cv2.CV_64F, 0, 1, 5)
orig_mag_gauss2 = np.sqrt((dxg2 * dxg2) + (dyg2 * dyg2))
cv2.imwrite('orig_mag_gauss2.png', orig_mag_gauss2)

dxg3 = cv2.Sobel(gaussian3, cv2.CV_64F, 1, 0, 5)
dyg3 = cv2.Sobel(gaussian3, cv2.CV_64F, 0, 1, 5)
orig_mag_gauss3 = np.sqrt((dxg3 * dxg3) + (dyg3 * dyg3))
cv2.imwrite('orig_mag_gauss3.png', orig_mag_gauss3)

# gray
dxg1g = cv2.Sobel(gaussian1g, cv2.CV_64F, 1, 0, 5)
dyg1g = cv2.Sobel(gaussian1g, cv2.CV_64F, 0, 1, 5)
gray_mag_gauss1 = np.sqrt((dxg1g * dxg1g) + (dyg1g * dyg1g))
cv2.imwrite('gray_mag_gauss1.png', gray_mag_gauss1)

dxg2g = cv2.Sobel(gaussian2g, cv2.CV_64F, 1, 0, 5)
dyg2g = cv2.Sobel(gaussian2g, cv2.CV_64F, 0, 1, 5)
gray_mag_gauss2 = np.sqrt((dxg2g * dxg2g) + (dyg2g * dyg2g))
cv2.imwrite('gray_mag_gauss2.png', gray_mag_gauss2)

dxg3g = cv2.Sobel(gaussian3g, cv2.CV_64F, 1, 0, 5)
dyg3g = cv2.Sobel(gaussian3g, cv2.CV_64F, 0, 1, 5)
gray_mag_gauss3 = np.sqrt((dxg3g * dxg3g) + (dyg3g * dyg3g))
cv2.imwrite('gray_mag_gauss3.png', gray_mag_gauss3)

cv2.imshow('mag gauss3', gray_mag_gauss3)

# derivative approx

# ((orig + (1, 0, 0)) - (orig - (1, 0, 0)))/2
# orig_dx_approx, orig_dy_approx, orig_color = np.gradient(orig)
# gray_dx_approx, gray_dy_approx = np.gradient(grayscale)
# # cv2.imshow('dx_approx', gray_dx_approx)
# cv2.imwrite('approx/orig/orig_dx_approx.png', orig_dx_approx)
# cv2.imwrite('approx/orig/orig_dy_approx.png', orig_dy_approx)
# cv2.imwrite('approx/gray/gray_dx_approx.png', gray_dx_approx)
# cv2.imwrite('approx/gray/gray_dy_approx.png', gray_dy_approx)
#
# orig_approx_mag = np.sqrt((orig_dx_approx * orig_dx_approx) + (orig_dy_approx * orig_dy_approx))
# gray_approx_mag = np.sqrt((gray_dx_approx * gray_dx_approx) + (gray_dy_approx * gray_dy_approx))
# cv2.imwrite('approx/orig/orig_approx_magnitude.png', orig_approx_mag)
# cv2.imwrite('approx/gray/gray_approx_magnitude.png', gray_approx_mag)
#
# # gradients of median 5x5
# orig_dx_approx_med5, orig_dy_approx_med5, orig_med5_color = np.gradient(median5x5)
# gray_dx_approx_med5, gray_dy_approx_med5 = np.gradient(median5x5g)
# cv2.imwrite('approx/orig/med/orig_dx_approx_med5.png', orig_dx_approx_med5)
# cv2.imwrite('approx/orig/med/orig_dy_approx_med5.png', orig_dy_approx_med5)
# cv2.imwrite('approx/gray/med/gray_dx_approx_med5.png', gray_dx_approx_med5)
# cv2.imwrite('approx/gray/med/gray_dy_approx_med5.png', gray_dy_approx_med5)
#
# orig_approx_med5_mag = np.sqrt((orig_dx_approx_med5 * orig_dx_approx_med5) + (orig_dy_approx_med5 * orig_dy_approx_med5))
# gray_approx_med5_mag = np.sqrt((gray_dx_approx_med5 * gray_dx_approx_med5) + (gray_dy_approx_med5 * gray_dy_approx_med5))
# cv2.imwrite('approx/orig/med/orig_approx_mag_med5.png', orig_approx_med5_mag)
# cv2.imwrite('approx/gray/med/gray_approx_mag_med5.png', gray_approx_med5_mag)
#
# # gradients of median 3x3
# orig_dx_approx_med3, orig_dy_approx_med3, orig_med_3_color = np.gradient(median3x3)
# gray_dx_approx_med3, gray_dy_approx_med3 = np.gradient(median3x3g)
# cv2.imwrite('approx/orig/med/orig_dx_approx_med3.png', orig_dx_approx_med3)
# cv2.imwrite('approx/orig/med/orig_dy_approx_med3.png', orig_dy_approx_med3)
# cv2.imwrite('approx/gray/med/gray_dx_approx_med3.png', gray_dx_approx_med3)
# cv2.imwrite('approx/gray/med/gray_dy_approx_med3.png', gray_dy_approx_med3)
#
# orig_approx_med3_mag = np.sqrt((orig_dx_approx_med3 * orig_dx_approx_med3) + (orig_dy_approx_med3 * orig_dy_approx_med3))
# gray_approx_med3_mag = np.sqrt((gray_dx_approx_med3 * gray_dx_approx_med3) + (gray_dy_approx_med3 * gray_dy_approx_med3))
# cv2.imwrite('approx/orig/med/orig_approx_mag_med3.png', orig_approx_med3_mag)
# cv2.imwrite('approx/gray/med/gray_approx_mag_med3.png', gray_approx_med3_mag)
#
# # gradients of gaussian 1
# orig_dx_approx_gau1, orig_dy_approx_gau1, orig_gau1_color = np.gradient(gaussian1)
# gray_dx_approx_gau1, gray_dy_approx_gau1 = np.gradient(gaussian1g)
# cv2.imwrite('approx/orig/gauss/orig_dx_approx_gau1.png', orig_dx_approx_gau1)
# cv2.imwrite('approx/orig/gauss/orig_dy_approx_gau1.png', orig_dy_approx_gau1)
# cv2.imwrite('approx/gray/gauss/gray_dx_approx_gau1.png', gray_dx_approx_gau1)
# cv2.imwrite('approx/gray/gauss/gray_dy_approx_gau1.png', gray_dy_approx_gau1)
#
# orig_approx_gau1_mag = np.sqrt((orig_dx_approx_gau1 * orig_dx_approx_gau1) + (orig_dy_approx_gau1 * orig_dy_approx_gau1))
# gray_approx_gau1_mag = np.sqrt((gray_dx_approx_gau1 * gray_dx_approx_gau1) + (gray_dy_approx_gau1 * gray_dy_approx_gau1))
# cv2.imwrite('approx/orig/gauss/orig_approx_mag_gau1.png', orig_approx_gau1_mag)
# cv2.imwrite('approx/gray/gauss/gray_approx_mag_gau1.png', gray_approx_gau1_mag)
#
# # gradients of gaussian 2
# orig_dx_approx_gau2, orig_dy_approx_gau2, orig_gau2_color = np.gradient(gaussian2)
# gray_dx_approx_gau2, gray_dy_approx_gau2 = np.gradient(gaussian2g)
# cv2.imwrite('approx/orig/gauss/orig_dx_approx_gau2.png', orig_dx_approx_gau2)
# cv2.imwrite('approx/orig/gauss/orig_dy_approx_gau2.png', orig_dy_approx_gau2)
# cv2.imwrite('approx/gray/gauss/gray_dx_approx_gau2.png', gray_dx_approx_gau2)
# cv2.imwrite('approx/gray/gauss/gray_dy_approx_gau2.png', gray_dy_approx_gau2)
#
# orig_approx_gau2_mag = np.sqrt((orig_dx_approx_gau2 * orig_dx_approx_gau2) + (orig_dy_approx_gau2 * orig_dy_approx_gau2))
# gray_approx_gau2_mag = np.sqrt((gray_dx_approx_gau2 * gray_dx_approx_gau2) + (gray_dy_approx_gau2 * gray_dy_approx_gau2))
# cv2.imwrite('approx/orig/gauss/orig_approx_mag_gau2.png', orig_approx_gau2_mag)
# cv2.imwrite('approx/gray/gauss/gray_approx_mag_gau2.png', gray_approx_gau2_mag)
#
# # gradients of gaussian 3
# orig_dx_approx_gau3, orig_dy_approx_gau3, orig_gau3_color = np.gradient(gaussian3)
# gray_dx_approx_gau3, gray_dy_approx_gau3 = np.gradient(gaussian3g)
# cv2.imwrite('approx/orig/gauss/orig_dx_approx_gau3.png', orig_dx_approx_gau3)
# cv2.imwrite('approx/orig/gauss/orig_dy_approx_gau3.png', orig_dy_approx_gau3)
# cv2.imwrite('approx/gray/gauss/gray_dx_approx_gau3.png', gray_dx_approx_gau3)
# cv2.imwrite('approx/gray/gauss/gray_dy_approx_gau3.png', gray_dy_approx_gau3)
#
# orig_approx_gau3_mag = np.sqrt((orig_dx_approx_gau3 * orig_dx_approx_gau3) + (orig_dy_approx_gau3 * orig_dy_approx_gau3))
# gray_approx_gau3_mag = np.sqrt((gray_dx_approx_gau3 * gray_dx_approx_gau3) + (gray_dy_approx_gau3 * gray_dy_approx_gau3))
# cv2.imwrite('approx/orig/gauss/orig_approx_mag_gau3.png', orig_approx_gau3_mag)
# cv2.imwrite('approx/gray/gauss/gray_approx_mag_gau3.png', gray_approx_gau3_mag)
# cv2.imshow('magitude gaussian 3', gray_approx_gau3_mag)

# cv2.imshow('Color', orig)

k = cv2.waitKey(0) & 0xFF

if k == 27: # wait for ESC key to exit
    cv2.destroyAllWindows()

elif k == ord('m'): # median filtering (5x5, 3x3)
    cv2.imwrite('orig_median5x5.png', median5x5)
    cv2.imwrite('orig_median3x3.png', median3x3)

    cv2.imwrite('gray_median5x5.png', median5x5g)
    cv2.imwrite('gray_median3x3.png', median3x3g)

    # cv2.imshow('Median 5x5', median5x5)
    # cv2.imshow('Median 3x3', median3x3)

elif k == ord('s'): # gaussian smoothing (sigma = 1,2,3)
    cv2.imwrite('orig_gauss_sigma1.png', gaussian1)
    cv2.imwrite('orig_gauss_sigma2.png', gaussian2)
    cv2.imwrite('orig_gauss_sigma3.png', gaussian3)

    cv2.imwrite('gray_gauss_sigma1.png', gaussian1g)
    cv2.imwrite('gray_gauss_sigma2.png', gaussian2g)
    cv2.imwrite('gray_gauss_sigma3.png', gaussian3g)

    # cv2.imshow('gaussian sigma 1', gaussian1)
    # cv2.imshow('gaussian sigma 2', gaussian2)
    # cv2.imshow('gaussian sigma 3', gaussian3)

elif k == ord('d'): # orig derivatives in x,y for original and smoothed origs, magnitude origs
    # cv2.imshow("dx", dx)
    cv2.imwrite('orig_sobel_dx.png', dx)
    cv2.imwrite('orig_sobel_dy.png', dy)

    cv2.imwrite('gray_sobel_dx.png', dxg)
    cv2.imwrite('gray_sobel_dy.png', dyg)

    cv2.imwrite('orig_magnitude.png', magnitude_orig)

    cv2.imwrite('gray_magnitude.png', magnitude_gray)

# Post the results and your programs/scripts on your webpage

# write a report describing your work. Your report must be clear and as brief as possible without compromising comprehension

# Email zrajabi@gmu.edu the link and the report. Make sure that you put CS 682, Homework 1 in the subject of the message