import cv2
import numpy as np

# 2
orig = cv2.imread('orig.jpg', cv2.IMREAD_COLOR)
grayscale = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
cv2.imwrite('gray.png', grayscale)

# median filter
median5x5 = cv2.medianBlur(orig, 5)  # 5x5
median3x3 = cv2.medianBlur(orig, 3)  # 3x3
median5x5g = cv2.medianBlur(grayscale, 5)  # 5x5
median3x3g = cv2.medianBlur(grayscale, 3)  # 3x3

# gaussian smoothing
gaussian1 = cv2.GaussianBlur(orig, (5, 5), 1)
gaussian2 = cv2.GaussianBlur(orig, (5, 5), 2)
gaussian3 = cv2.GaussianBlur(orig, (5, 5), 3)
gaussian1g = cv2.GaussianBlur(grayscale, (5, 5), 1)
gaussian2g = cv2.GaussianBlur(grayscale, (5, 5), 2)
gaussian3g = cv2.GaussianBlur(grayscale, (5, 5), 3)

# write median filters to file
cv2.imwrite('orig_median5x5.png', median5x5)
cv2.imwrite('orig_median3x3.png', median3x3)
cv2.imwrite('gray_median5x5.png', median5x5g)
cv2.imwrite('gray_median3x3.png', median3x3g)

# write gaussian smoothing to file
cv2.imwrite('orig_gauss_sigma1.png', gaussian1)
cv2.imwrite('orig_gauss_sigma2.png', gaussian2)
cv2.imwrite('orig_gauss_sigma3.png', gaussian3)
cv2.imwrite('gray_gauss_sigma1.png', gaussian1g)
cv2.imwrite('gray_gauss_sigma2.png', gaussian2g)
cv2.imwrite('gray_gauss_sigma3.png', gaussian3g)

# derivative sobel
dx = cv2.Sobel(orig, cv2.CV_64F, 1, 0, 5)
dy = cv2.Sobel(orig, cv2.CV_64F, 0, 1, 5)
dxg = cv2.Sobel(grayscale, cv2.CV_64F, 1, 0, 5)
dyg = cv2.Sobel(grayscale, cv2.CV_64F, 0, 1, 5)

# magnitude of original and gray
magnitude_gray = np.sqrt((dxg * dxg) + (dyg * dyg))
magnitude_orig = np.sqrt((dx * dx) + (dy * dy))

# magnitude of smoothing (orig and gray)
# orig derivatives
# sigma = 1
dxg1 = cv2.Sobel(gaussian1, cv2.CV_64F, 1, 0, 5)
dyg1 = cv2.Sobel(gaussian1, cv2.CV_64F, 0, 1, 5)
orig_mag_gauss1 = np.sqrt((dxg1 * dxg1) + (dyg1 * dyg1))
# sigma = 2 derivatives
dxg2 = cv2.Sobel(gaussian2, cv2.CV_64F, 1, 0, 5)
dyg2 = cv2.Sobel(gaussian2, cv2.CV_64F, 0, 1, 5)
orig_mag_gauss2 = np.sqrt((dxg2 * dxg2) + (dyg2 * dyg2))
# sigma = 3
dxg3 = cv2.Sobel(gaussian3, cv2.CV_64F, 1, 0, 5)
dyg3 = cv2.Sobel(gaussian3, cv2.CV_64F, 0, 1, 5)
orig_mag_gauss3 = np.sqrt((dxg3 * dxg3) + (dyg3 * dyg3))

# grayscale smoothed image derivatives
# sigma = 1
dxg1g = cv2.Sobel(gaussian1g, cv2.CV_64F, 1, 0, 5)
dyg1g = cv2.Sobel(gaussian1g, cv2.CV_64F, 0, 1, 5)
gray_mag_gauss1 = np.sqrt((dxg1g * dxg1g) + (dyg1g * dyg1g))
# sigma = 2
dxg2g = cv2.Sobel(gaussian2g, cv2.CV_64F, 1, 0, 5)
dyg2g = cv2.Sobel(gaussian2g, cv2.CV_64F, 0, 1, 5)
gray_mag_gauss2 = np.sqrt((dxg2g * dxg2g) + (dyg2g * dyg2g))
# sigma = 3
dxg3g = cv2.Sobel(gaussian3g, cv2.CV_64F, 1, 0, 5)
dyg3g = cv2.Sobel(gaussian3g, cv2.CV_64F, 0, 1, 5)
gray_mag_gauss3 = np.sqrt((dxg3g * dxg3g) + (dyg3g * dyg3g))

# write magnitudes of smoothed images to file
cv2.imwrite('orig_mag_gauss1.png', orig_mag_gauss1)
cv2.imwrite('orig_mag_gauss2.png', orig_mag_gauss2)
cv2.imwrite('orig_mag_gauss3.png', orig_mag_gauss3)
cv2.imwrite('gray_mag_gauss1.png', gray_mag_gauss1)
cv2.imwrite('gray_mag_gauss2.png', gray_mag_gauss2)
cv2.imwrite('gray_mag_gauss3.png', gray_mag_gauss3)