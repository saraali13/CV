import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import feature, exposure
from IPython.display import display
import pandas as pd

# Read image (OpenCV loads as BGR)
image = cv2.imread('image.jpg')

# Read as grayscale directly
gray = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Check if image loaded
if image is None:
    print("Error: Image not found")

# Convert BGR → RGB (for Matplotlib)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert BGR → Grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Convert BGR → HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Display with Matplotlib
plt.imshow(image_rgb)
plt.title('Title')
plt.axis('off')
plt.show()

# Display grayscale
plt.imshow(gray, cmap='gray')
plt.axis('off')
plt.show()

# Subplot display
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.title("Original")
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(gray, cmap='gray')
plt.title("Grayscale")
plt.axis("off")
plt.tight_layout()
plt.show()

# Get image dimensions
height, width = image.shape[:2]
channels = image.shape[2] if len(image.shape) == 3 else 1

#PHOTOMETRIC / POINT TRANSFORMATIONS
# S = c * log(1 + r) log trans
c = 255 / np.log(1 + np.max(gray))
log_image = c * np.log(1 + gray)
log_image = np.array(log_image, dtype=np.uint8)

plt.imshow(log_image, cmap='gray')
plt.title('Log Transformation')
plt.show()

# S = c * r^gamma Gamma (Power-Law) Transformation
gamma = 0.5   # Try 0.4, 0.6, 3.0
gamma_image = np.array(255 * (gray / 255) ** gamma, dtype='uint8')

plt.imshow(gamma_image, cmap='gray')
plt.title(f'Gamma = {gamma}')
plt.show()

# Piecewise-Linear (Contrast Stretching)
# Stretch input range [r1, r2] to output [s1, s2]
r1, r2 = 100, 200
s1, s2 = 0, 255
stretched = np.interp(gray, [r1, r2], [s1, s2]).astype(np.uint8)

plt.imshow(stretched, cmap='gray')
plt.title('Contrast Stretched')

#Intensity-Level Slicing
# Highlight a specific intensity range
lower, upper = 100, 150
sliced = np.where((gray >= lower) & (gray <= upper), 255, 0).astype(np.uint8)

plt.imshow(sliced, cmap='gray')
plt.title('Intensity Slicing')

#Histogram Equalization
equalized = cv2.equalizeHist(gray)

plt.imshow(equalized, cmap='gray')
plt.title('Equalized Image')
plt.show()
plt.show()
plt.show()

#Histogram Calculation & Plot
hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

plt.plot(hist)
plt.title('Histogram')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.show()

#CLAHE (Adaptive Histogram Equalization)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_image = clahe.apply(gray)
plt.imshow(clahe_image, cmap='gray')
plt.show()

#GEOMETRIC TRANSFORMATIONS
#Translation
tx, ty = 100, 50
M = np.float32([[1, 0, tx],
                [0, 1, ty]])
translated = cv2.warpAffine(image, M, (width, height))

#Rotation
center = (width // 2, height // 2)
angle = 45
scale = 1.0
M = cv2.getRotationMatrix2D(center, angle, scale)
rotated = cv2.warpAffine(image, M, (width, height))

#Scaling
# Method 1: fx, fy
scaled = cv2.resize(image, None, fx=2, fy=2, 
                    interpolation=cv2.INTER_LINEAR)

# Method 2: explicit size
scaled = cv2.resize(image, (new_width, new_height))

# Interpolation options:
# cv2.INTER_NEAREST   - fastest
# cv2.INTER_LINEAR    - default, good
# cv2.INTER_CUBIC     - slower, better
# cv2.INTER_AREA      - best for shrinking

#Shearing
# X-axis shear
kx = 0.5
M = np.float32([[1, kx, 0],
                [0, 1,  0]])
new_width = int(width + abs(kx) * height)
sheared = cv2.warpAffine(image, M, (new_width, height))

# Y-axis shear
ky = 0.5
M = np.float32([[1, 0, 0],
                [ky, 1, 0]])
new_height = int(height + abs(ky) * width)
sheared = cv2.warpAffine(image, M, (width, new_height))

#Reflection
# Horizontal flip
flipped = cv2.flip(image, 1)
# Vertical flip
flipped = cv2.flip(image, 0)
# Both
flipped = cv2.flip(image, -1)

#Rigid Transformation (Rotation + Translation)
center = (width // 2, height // 2)
M = cv2.getRotationMatrix2D(center, 30, 1.0)
M[0, 2] += 100  # Add translation X
M[1, 2] += 50   # Add translation Y
rigid = cv2.warpAffine(image, M, (width, height))

# Affine Transformation (3 point pairs)
pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
M = cv2.getAffineTransform(pts1, pts2)
affine = cv2.warpAffine(image, M, (width, height))

#Perspective (Homography, 4 point pairs)
src = np.float32([[0, 0], [width-1, 0], 
                  [width-1, height-1], [0, height-1]])
dst = np.float32([[100, 50], [width-150, 0], 
                  [width-50, height-80], [50, height-20]])
M = cv2.getPerspectiveTransform(src, dst)
warped = cv2.warpPerspective(image, M, (width, height))

# IMAGE FILTERING & CONVOLUTION
#Box blur
kernel = np.ones((3, 3), dtype=np.float32) / 9
blurred = cv2.filter2D(image, -1, kernel)
#Gaussian Blur
# Method 1: Direct
blurred = cv2.GaussianBlur(image, (5, 5), 0)
# (5,5) = kernel size (must be odd), 0 = auto sigma
# Method 2: Manual kernel
kernel = cv2.getGaussianKernel(5, 1.0)
blurred = cv2.filter2D(image, -1, kernel)
#Median Blur
blurred = cv2.medianBlur(image, 5)
#Bilateral Filter
blurred = cv2.bilateralFilter(image, 9, 75, 75)

#Convolution with Custom Kernel
# Emboss
kernel = np.array([[-2, -1, 0],
                   [-1,  1, 1],
                   [ 0,  1, 2]], dtype=np.float32)
embossed = cv2.filter2D(image, -1, kernel)

# Sharpen
kernel = np.array([[ 0, -1,  0],
                   [-1,  5, -1],
                   [ 0, -1,  0]], dtype=np.float32)
sharpened = cv2.filter2D(image, -1, kernel)

#Convolution Border Types
# Valid convolution (no padding)
result = cv2.filter2D(image, -1, kernel, 
                      borderType=cv2.BORDER_CONSTANT)
# Same convolution (with padding)
result = cv2.filter2D(image, -1, kernel, 
                      borderType=cv2.BORDER_REFLECT)

#Edge detection
#Sobel
gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)  # X gradient
gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)  # Y gradient
magnitude = np.sqrt(gx**2 + gy**2)
direction = np.arctan2(gy, gx) * 180 / np.pi

#Scharr (better than Sobel)
gx = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
gy = cv2.Scharr(gray, cv2.CV_64F, 0, 1)

#Laplacian
laplacian = cv2.Laplacian(gray, cv2.CV_64F)
laplacian_abs = cv2.convertScaleAbs(laplacian)

#Canny (Best)
blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
edges = cv2.Canny(blurred, 50, 150)  # lower, upper thresholds
#Blur → Gradient → Non-Max Suppression → Hysteresis

#Laplacian of Gaussian (LoG)
blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
laplacian_abs = cv2.convertScaleAbs(laplacian)

#THRESHOLDING
#Global Thresholding
_, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
#Otsu's Thresholding
_, otsu = cv2.threshold(gray, 0, 255, 
                        cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#Adaptive Thresholding
# Gaussian
adaptive = cv2.adaptiveThreshold(gray, 255, 
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
# Mean
adaptive = cv2.adaptiveThreshold(gray, 255, 
    cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
#Color-Based Thresholding
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower = np.array([30, 50, 50])
upper = np.array([60, 255, 255])
mask = cv2.inRange(hsv, lower, upper)
result = cv2.bitwise_and(image, image, mask=mask)

#MORPHOLOGICAL OPERATIONS
kernel = np.ones((5, 5), np.uint8)

# Erosion (shrinks white)
eroded = cv2.erode(image, kernel, iterations=1)

# Dilation (expands white)
dilated = cv2.dilate(image, kernel, iterations=1)

# Opening (erosion → dilation) - removes noise
opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

# Closing (dilation → erosion) - fills holes
closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

# Gradient (dilation - erosion) - outline
gradient = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)

# Top Hat (original - opening)
tophat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)

# Black Hat (closing - original)
blackhat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)

#HISTOGRAM-BASED FEATURES
#HOG
from skimage.feature import hog
from skimage import exposure

features, hog_image = hog(gray, 
                          pixels_per_cell=(8, 8),
                          cells_per_block=(2, 2), 
                          visualize=True)

hog_image_rescaled = exposure.rescale_intensity(hog_image, 
                                                in_range=(0, 10))

plt.figure(figsize=(12, 6))
plt.subplot(121); plt.imshow(gray, cmap='gray'); plt.title('Original')
plt.subplot(122); plt.imshow(hog_image_rescaled, cmap='gray'); plt.title('HOG')
plt.show()

print("HOG Feature Vector:", features)

# LBP (Local Binary Pattern)
from skimage import feature

radius = 1
n_points = 8
lbp = feature.local_binary_pattern(gray, n_points, radius, 
                                   method='uniform')

hist, _ = np.histogram(lbp.ravel(), 
                       bins=np.arange(0, n_points + 3), 
                       range=(0, n_points + 2))
hist = hist.astype("float")
hist /= (hist.sum() + 1e-6)

plt.figure(figsize=(12, 6))
plt.subplot(121); plt.imshow(lbp, cmap='gray'); plt.title('LBP')
plt.subplot(122); plt.bar(range(len(hist)), hist); plt.title('LBP Hist')
plt.show()

#Color Histogram
hist = cv2.calcHist([image], [0, 1, 2], None, 
                    [8, 8, 8], [0, 256, 0, 256, 0, 256])
hist = cv2.normalize(hist, hist).flatten()

#HED (Histogram of Edge Directions)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
gx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
gy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
orientation = np.arctan2(gy, gx) * 180 / np.pi
hist, _ = np.histogram(orientation, bins=8, range=(0, 360))

#Texture Energy & Contrast
n = 3
energy = cv2.filter2D(gray**2, -1, np.ones((n, n)))
contrast = cv2.filter2D(gray, -1, np.ones((n, n)))

energy_hist, _ = np.histogram(energy, bins=256, range=(0, energy.max()))
contrast_hist, _ = np.histogram(contrast, bins=256, range=(0, contrast.max()))

#HOUGH TRANSFORM
#Hough Lines (Probabilistic)
edges = cv2.Canny(gray, 50, 150)
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 
                        threshold=100, 
                        minLineLength=50, 
                        maxLineGap=10)

if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

#Hough Lines (Standard)
lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
if lines is not None:
    for rho, theta in lines[:, 0]:
        a, b = np.cos(theta), np.sin(theta)
        x0, y0 = a*rho, b*rho
        x1, y1 = int(x0 + 1000*(-b)), int(y0 + 1000*(a))
        x2, y2 = int(x0 - 1000*(-b)), int(y0 - 1000*(a))
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

# Hough Circles
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 
                           dp=1, minDist=20,
                           param1=50, param2=30, 
                           minRadius=0, maxRadius=0)
if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)

#SIFT FEATURE EXTRACTION
sift = cv2.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(gray, None)

# Draw keypoints
image_kp = cv2.drawKeypoints(image_rgb, keypoints, None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.imshow(image_kp)
plt.show()

print("Number of keypoints:", len(keypoints))
print("Descriptor shape:", descriptors.shape)

#SIFT Matching (BFMatcher)
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

result = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None,
                         flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(result)
plt.show()

#SIFT Matching with Ratio Test (FLANN)
flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), 
                               dict(checks=50))
matches = flann.knnMatch(des1, des2, k=2)

good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append(m)

#TEXT ON IMAGE
cv2.putText(image_rgb, "Hello World", (150, 150),
            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)
#cropping
# Syntax: image[y_start:y_end, x_start:x_end]
# Left half
cropped = image_rgb[:, :width//2]
# Right half
cropped = image_rgb[:, width//2:]
# Top half
cropped = image_rgb[:height//2, :]
# Specific region
cropped = image_rgb[100:300, 200:400]

#blending
# cv2.add - saturated addition (caps at 255)
blended = cv2.add(image1, image2)
# cv2.addWeighted - weighted blend
blended = cv2.addWeighted(image1, 0.7, image2, 0.3, 0)
# Requirement: same dimensions
image1_resized = cv2.resize(image1, (image2.shape[1], image2.shape[0]))

#SEGMENTATION
#Region Growing
def region_growing(image, seed, threshold):
    mask = np.zeros_like(image, dtype=np.uint8)
    stack = [seed]
    seed_intensity = image[seed]
    
    while stack:
        x, y = stack.pop()
        if x < 0 or x >= image.shape[0] or y < 0 or y >= image.shape[1]:
            continue
        if mask[x, y] == 0:
            if abs(int(image[x, y]) - int(seed_intensity)) < threshold:
                mask[x, y] = 255
                stack.extend([(x+1, y), (x-1, y), (x, y+1), (x, y-1)])
    return mask

segmented = region_growing(gray, (10, 10), 50)

#Watershed
_, thresh = cv2.threshold(gray, 0, 255, 
    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

kernel = np.ones((5, 5), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

sure_bg = cv2.dilate(opening, kernel, iterations=3)
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
_, sure_fg = cv2.threshold(dist_transform, 
    0.2 * dist_transform.max(), 255, 0)

sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

_, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown == 255] = 0

cv2.watershed(image, markers)
image[markers == -1] = [255, 0, 0]

#K-Means Segmentation
pixel_values = image.reshape((-1, 3))
pixel_values = np.float32(pixel_values)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 
            100, 0.2)
K = 4

_, labels, centers = cv2.kmeans(pixel_values, K, None, criteria, 
                                10, cv2.KMEANS_RANDOM_CENTERS)

centers = np.uint8(centers)
segmented = centers[labels.flatten()]
segmented = segmented.reshape(image.shape)

# WAVELET TRANSFORMS
import pywt

# Discrete Wavelet Transform (DWT)
coeffs = pywt.dwt2(gray, 'haar')
cA, (cH, cV, cD) = coeffs  # Approximation, Horizontal, Vertical, Diagonal

# Inverse DWT
reconstructed = pywt.idwt2(coeffs, 'haar')


#Convert to uint8
image_uint8 = cv2.convertScaleAbs(image)
# or
image_uint8 = np.clip(image, 0, 255).astype(np.uint8)
#image stats
print("Shape:", image.shape)
print("Min:", image.min())
print("Max:", image.max())
print("Mean:", image.mean())
print("Std:", image.std())
#normalized
normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
#reshape
flattened = image.reshape(-1, 3)  # (H*W, 3)
#draw shapes
cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
cv2.circle(image, (cx, cy), radius, (0, 0, 255), 2)
cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
