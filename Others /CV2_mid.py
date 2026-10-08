import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import math
from pathlib import Path
from skimage.feature import hog
from skimage import exposure
from skimage import feature


image = cv2.imread("Sukuna.jpg")
IMAGE_PATH = "data/sample_xray.png"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUT = Path('outputs')
OUT.mkdir(exist_ok=True)
gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) #read image as grayscle

if image is None:
    print("Error: Image not found. Check the file path.")
#else do whatever
capture = cv2.VideoCapture("xyz.mp4") #video
#Create a display function
def show(img, title='', cmap=None, size=(12,7)):
    plt.figure(figsize=size)
    if img.ndim == 3:
        plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    else:
        plt.imshow(img, cmap=cmap or 'gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

canvas = np.zeros((800, 800, 3), dtype=np.uint8) #height = 800, width  = 800, colors = 3, creates a blank image
center = (canvas.shape[1] // 2, canvas.shape[0] // 2) #find center of the image
radii = [300, 240, 180, 120, 60] #radius of circles
colors = [    #in CV2 BRG image not RBG unless converted
    (255, 0, 0),      # Blue in BGR
    (255, 255, 255),  # White
    (0, 0, 255),      # Red
    (255, 255, 255),  # White
    (0, 255, 255),    # Yellow
]

for radius, color in zip(radii, colors): #zip paires each radius with a color
    cv2.circle(canvas, center, radius, color, thickness=-1) #draw circles, thickness=-1 -> fill the entire circle if +ve val =4, only circle's outline
 
cv2.rectangle(
    canvas,
    (center[0] - outer_radius, center[1] - outer_radius),
    (center[0] + outer_radius, center[1] + outer_radius),
    (0, 255, 0),
    thickness=4
) #draw a box around the largest circle radii[0]->outer radius 
cv2.rectangle(image, top_left, bottom_right, color, thickness)
cv2.circle(image, center, radius, color, thickness)

#convert BRG image to RGB image (matplot req RGB image)
rgb_image = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(8, 8))
plt.imshow(rgb_canvas)
plt.title("Target Board with Bounding Box")
plt.axis("off")
plt.savefig("Task4_output.png", bbox_inches="tight")

#apply gaussain blur of size (25x25) kernel size larger->more strong blur
blurred_image = cv2.GaussianBlur(rgb_image, (25, 25), 0) #0-? std dev will be calculated by opencv automatically
#Get image dimensions
height, width = rgb_image.shape[:2] #(height, width, channels)-> RGB shape, so 1st 2 vals
#ROI region of interest size
roi_size = min(300, height, width) #here we want 300x300 at max region, min-> if h and w are greater than 300, else from an image of less then 300 we cant take 300x300 region
#Image center integer div
center_x = width // 2
center_y = height // 2
#half of ROI as we want to go left and right
half = roi_size // 2
#start and end points
start_x = center_x - half
start_y = center_y - half
end_x = start_x + roi_size
end_y = start_y + roi_size

#numpy image indexing
image[y1:y2, x1:x2] #y 1st then x
#ROI on org and blurred image
original_roi = rgb_image[start_y:end_y, start_x:end_x]
blurred_roi = blurred_image[start_y:end_y, start_x:end_x]
#show using subplots adn save image
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(original_roi)
axes[0].set_title("Original Center ROI", fontsize=14, color="darkgreen")
axes[0].axis("off")
axes[1].imshow(blurred_roi)
axes[1].set_title("Blurred Center ROI", fontsize=14, color="darkblue")
axes[1].axis("off")
plt.tight_layout()
plt.savefig("Task5_output.png", bbox_inches="tight")
#creates copy of an image to draw on
overlay = rgb_image.copy() 
box_top = int(height * 0.80) #rec starts when height is 80% of the org one
cv2.rectangle(
    overlay,
    (0, box_top),
    (width, height),
    (255, 0, 0),
    -1
)
#Fusing images Alpha blending This combines two images.
result = cv2.addWeighted(overlay, 0.5, image, 0.5, 0) #result = image1(overlay) × alpha(0.5) + image2(image) × beta(0.5) -> ws per image (how imp it is)
#So each image contributes 50%. Because overlay has a blue rectangle at the bottom, blending it with the original image makes the blue rectangle transparent/semi-transparent
cv2.addWeighted(src1, alpha, src2, beta, gamma) #1st image and w, 2nd img and w , gamma extra brightness added to result

font_scale = max(0.7, width / 1000)
text_y = box_top + (height - box_top) // 2
cv2.putText(
    result,
    "Lab 1 task 6",
    (25, text_y),#text pos
    cv2.FONT_HERSHEY_SIMPLEX,
    font_scale,#text size
    (255, 255, 255),#color white
    2, #thickness 
    cv2.LINE_AA #smoother text edges
)
cv2.putText(image, text, position, font, fontScale, color, thickness, lineType)
#global threshold (convert grayscale image to black and white image), one threshold for entire image
_, global_threshold = cv2.threshold(
    gray_image,
    127, #cutoff point
    255, #val for white
    cv2.THRESH_BINARY #0-127 ->black, 127-255 ->white
)
cv2.threshold(image, threshold_value, max_value, threshold_type)

#adaptive thresholding, Adaptive threshold calculates different thresholds for different local areas.
adaptive_threshold = cv2.adaptiveThreshold(
    gray_image,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, #use gaussian weighted neighborhood to cal local threshold
    cv2.THRESH_BINARY,
    11, #look 11x11 neighborhood 
    2 #sub this from local threshold (adjustment to threshold)
)
cv2.adaptiveThreshold(
    image,
    maxValue,
    adaptiveMethod,
    thresholdType,
    blockSize,
    C
)
#rotating
height, width = adaptive_threshold.shape
center = (width // 2, height // 2)
rotation_matrix = cv2.getRotationMatrix2D(center, 45, 0.8)
cv2.getRotationMatrix2D(center, angle, scale) #create a rotation matrix
rotated_image = cv2.warpAffine(
    adaptive_threshold,
    rotation_matrix,
    (width, height),
    borderValue=255 #after rotation, some areas pixels might be empty so set their val=255
)
cv2.warpAffine(image, transformation_matrix, output_size) #apply rotation 

#resize an image
size = (500, 500)
image1 = cv2.resize(image1, size)
image2 = cv2.resize(image2, size)
#for empty mask / image
mask = np.zeros((500, 500), dtype=np.uint8)
cv2.circle( #draw whte circele on it 
    mask,
    (250, 250),
    170,
    255,
    thickness=-1
)
#bitwise op
foreground = cv2.bitwise_and( #Keep pixels where the mask is white.
    image1,
    image1,
    mask=mask
)
cv2.bitwise_and(image, image, mask=mask) #Select the area specified by the mask.
inverted_mask = cv2.bitwise_not(mask) #inverse back and white regions
background = cv2.bitwise_and( #now inverted white-> image 2
    image2,
    image2,
    mask=inverted_mask
)
combined = cv2.bitwise_or(foreground, background) #now combine
cv2.bitwise_or(image1, image2) #Combine the selected parts of two images.
#for an RGB image
    red = rgb_image[:, :, 0].flatten()
    green = rgb_image[:, :, 1].flatten()
    blue = rgb_image[:, :, 2].flatten()

    pixel_data = pd.DataFrame({
        "Red": red,
        "Green": green,
        "Blue": blue
    })

    print("Color Channel Statistical Summary:")
    print(pixel_data.describe().round(2))

    print("\nSelected statistics:")
    print(
        pixel_data.describe()
        .loc[["mean", "min", "max", "std"]]
        .round(2)
    )

#Color balancing->
def color_balance(image):
    """Simple gray-world color balance for a BGR image."""
    result = image.astype(np.float32)#pixel to floating point
    channel_means = result.mean(axis=(0, 1))#avg val of each channel
    target_mean = channel_means.mean() #overall target mean of channels mean
    scale = target_mean / (channel_means + 1e-6) #how much each color channel needs to be inc and dec
    return np.clip(result * scale, 0, 255).astype(np.uint8) #apply correction to image and conv to image format again (not float) and ensures pixel values remain between 0 and 255.
#Histogram Equalization-> spread the intensity values across a larger portion of the 0–255 range.
hist_equalized = cv2.equalizeHist(xray) #improve overall image contrast 
#False-color heatmap -> convert grayscale intensities into colors.
heatmap = cv2.applyColorMap(
    hist_equalized,
    cv2.COLORMAP_JET #Dark intensity   → blue/purple, Medium intensity → green/yellow, High intensity   → orange/red JET color map
    #cv2.COLORMAP_BONE  this one color map -> give a grayscale-like bone/medical visualization.
)
balanced_heatmap = color_balance(heatmap)#func created above, Jet colored one used
#finding and keeping the bright region only
threshold_value = 200
_, dense_tissue_mask = cv2.threshold(
    hist_equalized,
    threshold_value, 
    255,
    cv2.THRESH_BINARY #grayscale to black nd white-> greater than threshold -> white
)
dense_tissue = cv2.bitwise_and( #extract the white region
    hist_equalized,
    hist_equalized,
    mask=dense_tissue_mask
)
#log transformation ->s = c × log(1 + r) s=transformed, c=scaling cost, r=org pixel, expand diff in dark and compress diff in light regions,  dark details more visible.
c = 255 / np.log1p(np.max(xray)) #cal constant used for scaling
log_image = c * np.log1p( # cal log(1 + x)
    xray.astype(np.float32)
)
log_image = np.uint8(
    np.clip(log_image, 0, 255)
) #result is 8 bit image
#Power-law / Gamma transformation
gamma = 0.6 #gamma is less than 1, darker and middle intensity values are generally brightened, <1 -> bright, >1 -> dark
power_law = np.uint8(
    255 * (xray / 255.0) ** gamma
)
#for video
PANEL_WIDTH = 640
PANEL_HEIGHT = 480
#dim above
cv2.namedWindow( #creates a resizable window of OPENCV
    "Raw and Enhanced Echocardiogram",
    cv2.WINDOW_NORMAL
)
cv2.resizeWindow(
    "Raw and Enhanced Echocardiogram",
    PANEL_WIDTH * 2,
    PANEL_HEIGHT
)
#all inside a while loop
success, frame = capture.read() #reads one frame from the video.
gray = cv2.cvtColor(
    frame,
    cv2.COLOR_BGR2GRAY #frame to gray scale (each)
)
#after loop
capture.release() #close vid file
cv2.destroyAllWindows()

#get image dim
H_IMG, W_IMG = image.shape[:2] #image height and width
sx, sy = 3.0, 3.0 #scaling val-> 3x in both dim
S2 = np.array([   #2x2 scaling matrix
    [sx, 0.0],
    [0.0, sy]
], dtype=np.float32)

S3 = np.array([     #3x3 scaling -> homogeneous transformation matrix, allows translation
    [sx, 0.0, 0.0],
    [0.0, sy, 0.0],
    [0.0, 0.0, 1.0]
], dtype=np.float32)
#center of image
cx = (W_IMG - 1) / 2.0
cy = (H_IMG - 1) / 2.0
#scale around center
tx = (1.0 - sx) * cx
ty = (1.0 - sy) * cy
M_zoom = np.array([
    [sx, 0.0, tx],
    [0.0, sy, ty],
    [0.0, 0.0, 1.0]
], dtype=np.float32)
#before affine wrap -2x3 matrix not 3x3
M_zoom_2x3 = M_zoom[:2, :] #first 2 rows
zoom_same = cv2.warpAffine(
    image,
    M_zoom_2x3,
    (W_IMG, H_IMG),
    flags=cv2.INTER_CUBIC
)  #apply transformation aound center 

#transformation but grow the canvas-> scaling and translation
new_W = int(round(W_IMG * sx))
new_H = int(round(H_IMG * sy))
tx_grow = (new_W - W_IMG) / 2.0
ty_grow = (new_H - H_IMG) / 2.0
M_grow = np.array([
    [sx, 0.0, tx_grow],
    [0.0, sy, ty_grow],
    [0.0, 0.0, 1.0]
], dtype=np.float32)
zoom_grow = cv2.warpAffine(
    image,
    M_grow[:2, :],
    size_grow,
    flags=cv2.INTER_CUBIC
)
flags=cv2.INTER_CUBIC #Interpolation determines how OpenCV calculates new pixels during resizing/scaling.
#rotation
theta_deg = -45.0 #45 deg clockwise
theta = math.radians(theta_deg)
R = np.array([
    [math.cos(theta), -math.sin(theta)],
    [math.sin(theta),  math.cos(theta)]
], dtype=np.float32)
#new dim size of canvas after rotating else te iamge will cut off
new_w = math.ceil(
    abs(W_IMG * math.cos(theta)) +
    abs(H_IMG * math.sin(theta))
)
new_h = math.ceil(
    abs(H_IMG * math.cos(theta)) +
    abs(W_IMG * math.sin(theta))
)
M = cv2.getRotationMatrix2D( #instead of creating manual transformation built 2x3 affine transformation matrix.
    (w / 2.0, h / 2.0), #rotate around center
    theta_deg,
    1.0
)
#image to new canvas
M[0, 2] += (new_w - w) / 2.0
M[1, 2] += (new_h - h) / 2.0
result = cv2.warpAffine(
    img,
    M,
    (new_w, new_h),
    cv2.INTER_CUBIC
)
#Sher
shear = -0.30 # how much horizontal slant is introduced/corrected (sign -> direction)
Sh = np.array([
    [1.0, shear], #y not change, x changed only-> horizontal shear(w changes, h is same)
    [0.0, 1.0]
], dtype=np.float32)
#new W as some pixels may move outside the original image width
min_x = min(0.0, shear * h)
max_x = max(float(w), w + shear * h)
new_w = int(np.ceil(max_x - min_x))
M = np.array([
    [1.0, shear, -min_x], #sher and translation
    [0.0, 1.0, 0.0]
], dtype=np.float32) 
#apply affine wrap on all
#translation
tx = 150
ty = 80
T = np.array([
    [1.0, 0.0, tx],
    [0.0, 1.0, ty],
    [0.0, 0.0, 1.0]
], dtype=np.float32)
#new canvas
new_w = w + tx
new_h = h + ty
result = cv2.warpPerspective( #apply translation 3x3 (cv2.warpAffine()->2x3 matrix)
    img,
    T,
    (new_w, new_h)
)
#Transformation parameters-> combine 2 transformations
angle = -30.0
tx = 120.0
ty = 60.0
theta = math.radians(angle)
c = math.cos(theta)
s = math.sin(theta)
R = np.array([   #rotation matrix with  homogeneous coordinates
    [c, -s, 0.0],
    [s,  c, 0.0],
    [0.0, 0.0, 1.0]
])
T = np.array([
    [1.0, 0.0, tx],
    [0.0, 1.0, ty],
    [0.0, 0.0, 1.0]
])
M = T @ R #combines the two transformations by matrix mul M=TR (1st rotate then translate)
#now transform corners
corners = np.array([  #original one
    [0, 0, 1],
    [w, 0, 1],
    [0, h, 1],
    [w, h, 1]
]).T
#After rotation, the corners move, How far does the rotated image extend
transformed_corners = M @ corners #This applies the combined rotation + translation to all four corners.
x_coords = transformed_corners[0, :]
y_coords = transformed_corners[1, :]
#min and max pos
min_x = np.min(x_coords)
max_x = np.max(x_coords)
min_y = np.min(y_coords)
max_y = np.max(y_coords)
#shift coor-> Extra translation
shift_x = -min_x if min_x < 0 else 0
shift_y = -min_y if min_y < 0 else 0
#modify trans matix-> This adds the extra translation required to make sure the entire transformed image fits inside the canvas.
M[0, 2] += shift_x
M[1, 2] += shift_y
#new dim
new_w = int(math.ceil(max_x + shift_x))
new_h = int(math.ceil(max_y + shift_y))
#apply
result = cv2.warpPerspective(
    img,
    M,
    (new_w, new_h)
)
#scale, rotate, translate-> 3 matrices then
M = T @ R @ S #First scale, then rotate, then translate.
#then trans the corners and shifting to fit (new dim and all above)

#calculate the affine trans using source and dest points
src = np.array([
    [100, 100],
    [300, 100],
    [100, 300]
], dtype=np.float32) #points of org image
dst = np.array([
    [120, 80],
    [350, 120],
    [100, 350]
], dtype=np.float32) #where those points should move
for (x, y), (X, Y) in zip(src, dst):

    A.append([x, y, 1, 0, 0, 0])
    A.append([0, 0, 0, x, y, 1])

    B.append(X)
    B.append(Y)

A = np.array(A, dtype=np.float64)
B = np.array(B, dtype=np.float64)  
#6 var to find or affine matrix 2 eq-> each 3 points
p = np.linalg.solve(A, B) # solves the six simultaneous equations.
M = p.reshape(2, 3) #rsehape and apply warpAffine

#perspective correction or transformation (affine 3 points, here 4 points)
n = 600 #output size
src = np.array([
    [100, 100],
    [500, 120],
    [550, 500],
    [80, 480]
])
dst = np.array([
    [0, 0],
    [n - 1, 0],
    [n - 1, n - 1],
    [0, n - 1]
])
H = cv2.getPerspectiveTransform(src, dst)
result = cv2.warpPerspective(
    img,
    H,
    (n, n)
)

#tadium Panorama / Image Stitching,  combines two overlapping stadium images into one panorama.
a = cv2.imread(image1_path)
b = cv2.imread(image2_path)
p1 = np.array([
    [100, 100],
    [500, 100],
    [500, 400],
    [100, 400]
]) #points of image 1
p2 = np.array([
    [150, 120],
    [550, 100],
    [530, 420],
    [120, 400]
]) #points of image 2
#p1[i] and p2[i] must represent the same physical location in both images.
H, mask = cv2.findHomography(p2, p1, cv2.RANSAC) #Calculate homography, p2 → p1, Image 2 coordinate system → Image 1 coordinate system
#RANSAC helps when some matching points might be incorrect.
corners1 = np.float32([
    [0, 0],
    [w1, 0],
    [w1, h1],
    [0, h1]
]) #img 1
corners2 = np.float32([
    [0, 0],
    [w2, 0],
    [w2, h2],
    [0, h2]
]).reshape(-1, 1, 2) #img 2-> will be adjusted 
warped_corners = cv2.perspectiveTransform(
    corners2,
    H
)
#This gives you the corners from both images.
all_pts = np.vstack([
    corners1,
    warped_corners
])
#mins and maxs, them translate
T = np.array([
    [1.0, 0.0, -min_x],
    [0.0, 1.0, -min_y],
    [0.0, 0.0, 1.0]
])
T @ H
pano = cv2.warpPerspective(
    b,
    T @ H,
    (out_w, out_h)
)
mask2 = np.ones(
    (h2, w2),
    dtype=np.uint8
) * 255
warped_mask2 = cv2.warpPerspective(
    mask2,
    T @ H,
    (out_w, out_h)
)
#img 2 above, img 1 bellow
pano1 = np.zeros_like(pano)
pano1[
    y_offset:y_offset + h1,
    x_offset:x_offset + w1
] = a
mask1 = np.zeros(
    (out_h, out_w),
    dtype=np.uint8
)
#find diff regions
mask1_bool = mask1 > 0
mask2_bool = warped_mask2 > 0
only1 = mask1_bool & ~mask2_bool
overlap = mask1_bool & mask2_bool
final[overlap] = (
    0.5 * pano1[overlap]
    + 0.5 * pano[overlap]
)  #final combined image

#Putting a Painting Inside a Museum Frame
painting = cv2.imread(painting_path)
wall = cv2.imread(wall_path)
hp, wp = painting.shape[:2]
hw, ww = wall.shape[:2]
src = np.array([
    [0, 0],
    [wp - 1, 0],
    [wp - 1, hp - 1],
    [0, hp - 1]
]) #paintings 4 corners
frame = np.array([
    [330, 90],
    [566, 90],
    [566, 414],
    [330, 414]
])#frames 4 corners
H = cv2.getPerspectiveTransform(src, frame) # 3×3 homography matrix that maps painting and frame
warped = cv2.warpPerspective(
    painting,
    H,
    (ww, hw)
)
painting_mask = np.ones(
    (hp, wp),
    dtype=np.uint8
) * 255  #mask to remove extra painting area 
mask = cv2.warpPerspective(
    painting_mask,
    H,
    (ww, hw)
)
kernel = np.ones((3, 3), np.uint8)

mask = cv2.erode(
    mask,
    kernel,
    iterations=1
) #make mask smaller 
#final 
result = wall.copy()
mask_bool = mask > 0
result[mask_bool] = warped[mask_bool]
#thresholds
thresholds = [80, 120, 160] #T>=white, T<= =black
globals_ = []#global threshold results
for t in thresholds: #apply global threshold 
    _, mask = cv2.threshold(
        img,
        t,
        255,
        cv2.THRESH_BINARY
    )
    globals_.append(mask)
cv2.threshold(image, threshold, maximum_value, method)
#adaptive threshold
adaptive = cv2.adaptiveThreshold( #calculate local threshold for each region T(x,y)=local weighted mean−C
    img,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    11, #block size -> odd and greater than 1, small-> highly local, sensitive to outliers, large->  too global, may not respond well to rapid illumination changes
    2 #C lowers the threshold
)
#diff adaptive thresholds 
configs = [ #(method, blockSize, C) each block 

    ("Mean", cv2.ADAPTIVE_THRESH_MEAN_C, 7, 2),
    ("Mean", cv2.ADAPTIVE_THRESH_MEAN_C, 15, 5),
    ("Mean", cv2.ADAPTIVE_THRESH_MEAN_C, 31, 10), #mean -> avg intensity of neighborhood
    #There are two ways OpenCV can calculate the local threshold (mean and Gaussian)
    ("Gaussian", cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 7, 2), #Gaussian -> Gaussian-weighted average 
    ("Gaussian", cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 15, 5),
    ("Gaussian", cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 31, 10),
]
#apply all
for name, method, block, c in configs:
    thresholded = cv2.adaptiveThreshold(
        img,
        255,
        method,
        cv2.THRESH_BINARY,
        block,
        c
    )
#Otsu's Thresholding-> Otsu's method automatically finds a suitable threshold from the image histogram instaed of manual thresholding 
def otsu(x):
    threshold, mask = cv2.threshold(
        x,
        0, #OTSU replace this 
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU #OTSU-> cal best threshold ->maximize the separation between the two classes(dark and bright)
    )
    return threshold, mask
t1, m1 = otsu(gray) #apply
#CLAHE enhancement -> Contrast Limited Adaptive Histogram Equalization
clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8) #divide image into grids or small tiles
)
clahe_image = clahe.apply(gray) #apply
t2, m2 = otsu(clahe_image)
#HSV color segmentation ->Hue->actual color, Saturation-> how string/pure color is, Value-> brightness
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV) #for color segmentation-> we can Keep pixels whose hue looks yellow/orange
restrict = cv2.inRange( #check every pixel
    hsv,
    np.array([12, 180, 180]), #in between
    np.array([18, 255, 255])    #these vals of HSV -> while else black
)
final = cv2.inRange( #final mask ranges changed
    hsv,
    np.array([5, 100, 100]),
    np.array([25, 255, 255])
)
kernel = np.ones((3, 3), np.uint8) #morphological kernel-> used to clean the segmentation mask
final = cv2.morphologyEx( #Its main purpose here is to remove small isolated noise
    final,
    cv2.MORPH_OPEN, #opening
    kernel,
    iterations=1
)
final = cv2.morphologyEx( #It helps fill small holes and gaps inside detected regions.
    final,
    cv2.MORPH_CLOSE, #closing
    kernel,
    iterations=2
)
extracted_bgr = cv2.bitwise_and(
    bgr,
    bgr,
    mask=final #This uses the mask to keep only the selected pixels, apply the mask (final) to org image-> desired color only
)
extracted_rgb = cv2.cvtColor(
    extracted_bgr,
    cv2.COLOR_BGR2RGB
)
#Canny Edge Detection->  detect the boundaries/edges of objects -> works with image intensity rather than color
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
#Apply Gaussian blur-> to avoid noise
blur = cv2.GaussianBlur(gray, (5, 5), 1)
#Define Canny threshold pairs
pairs = [
    (30, 80), #(low threshold, high threshold)
    (50, 120),
    (50, 200)
]
edges = [  #apply
    cv2.Canny(blur, low, high) #result-> bin image-> edge= white
    for low, high in pairs
]
#hysteresis thresholding-> used by canny -> strong edges= greater than the high threshold, weak edges= between the two thresholds, non edge= lower then low threshold

#Manual Region Growing
gray_full = cv2.imread(
    "mri.png",
    cv2.IMREAD_GRAYSCALE
)
_, dark_mask = cv2.threshold( # find the large dark MRI area
    gray_full,
    80,
    255,
    cv2.THRESH_BINARY_INV #Pixels below 80 become white
)
contours, _ = cv2.findContours( #finds connected dark regions
    dark_mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)
if w > 0.4 * W and h > 0.4 * H:
    gray = gray_full[y:y + h, x:x + w].copy()
def grow(im, seed, tol):
    """
    Region growing using the seed pixel intensity as reference.

    seed = (row, column)
    tol  = allowed grayscale intensity difference
    """

    h, w = im.shape

    mask = np.zeros( #pixel accepted-> white
        (h, w),
        dtype=np.uint8
    )

    seen = np.zeros( #This prevents the algorithm from processing the same pixel repeatedly.
        (h, w),
        dtype=bool  
    )

    q = deque([seed])

    # Reference intensity, seed intensity, seed intensity = 180 and tol =15 so range 165 → 195
    ref = int(im[seed])

    while q:

        row, col = q.popleft()

        # Boundary check
        if (
            row < 0 or row >= h or
            col < 0 or col >= w
        ):
            continue
                # Already examined
        if seen[row, col]:
            continue

        seen[row, col] = True

        # Compare current pixel with seed intensity -> intensity diff
        difference = abs(
            int(im[row, col]) - ref
        )

        if difference <= tol: #check tol

            mask[row, col] = 255

            # After accepting a pixel, its neighbors are added:
            q.extend([
                (row - 1, col),
                (row + 1, col),
                (row, col - 1),
                (row, col + 1),

                (row - 1, col - 1),
                (row - 1, col + 1),
                (row + 1, col - 1),
                (row + 1, col + 1)
            ])

    return mask
seed_bright = (
    int(0.58 * h),
    int(0.32 * w)
)
seed_tissue = (
    int(0.48 * h),
    int(0.66 * w)
)
tolerances = [
    5,
    15,
    30
]
final = grow(
    gray,
    seed_bright,
    15
)
pixel_count = np.count_nonzero(mask)# no of pixel included in region
final = grow(
    gray,
    seed_bright,
    15
)
#Marker-Based Watershed Code Explanation
blur = cv2.GaussianBlur(gray, (5, 5), 0)
_, th = cv2.threshold( #Otsu thresholding
    blur,
    0,
    255,
    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU #THRESH_BINARY_INV makes the coins white and the background black.
)
kernel = np.ones((3, 3), np.uint8)

opening = cv2.morphologyEx(
    th,
    cv2.MORPH_OPEN,
    kernel,
    iterations=2
)
sure_bg = cv2.dilate( #Dilation expands the foreground region, This gives us a region that we are confident belongs to the background around the objects
    opening,
    kernel,
    iterations=3
)
dist = cv2.distanceTransform( #This calculates the distance of each white pixel from the nearest black/background pixel
    opening,
    cv2.DIST_L2,
    5
)
_, sure_fg = cv2.threshold( #get the inner/core parts, Find sure foreground
    dist,
    0.5 * dist.max(),
    255,
    0
)
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract( #not sure
    sure_bg,
    sure_fg
)

n, markers = cv2.connectedComponents(sure_fg) #Finds each separate sure-foreground region
markers = markers + 1 #markers determine how many regions watershed can potentially separate
markers[unknown == 255] = 0
markers = cv2.watershed( #Watershed treats the marker regions like starting points and expands them,When two regions meet, a boundary is created
    bgr,
    markers
)
markers == -1 #watershed boundary
boundary = (markers == -1) #You remove the outer image boundary
boundary[0, :] = False
boundary[-1, :] = False
boundary[:, 0] = False
boundary[:, -1] = False
final[boundary] = [255, 0, 0]
#with diff threshold watershed boundary
fractions = [0.3, 0.4, 0.5, 0.6]
for f in fractions:

    # Create the sure foreground using the current fraction
    _, fg = cv2.threshold(
        dist,
        f * dist.max(),
        255,
        0
    )
    unknown = cv2.subtract(
     bg,
     fg
    )
    n, m = cv2.connectedComponents(fg)
    foreground = n - 1
    m = m + 1
    m[unknown == 255] = 0
    m = cv2.watershed(
    bgr.copy(),
    m
)
    separated = len([
    value
    for value in np.unique(m)
    if value > 1
])


#k-means image segmentation
data = np.float32(
    rgb.reshape((-1, 3))
) #Convert the image into a list of pixels, number_of_pixels × 3

criteria = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    100, #stop -> max itr 100
    0.2  #Stop earlier if the cluster centers change by less than 0.2
)
#k mean seg func
def segment(K):
    """
    Segment the image using K-Means clustering.

    Parameters:
        K : Number of color clusters

    Returns:
        Segmented RGB image
    """

    # Set a fixed random seed for reproducible results, initial cluster centers
    cv2.setRNGSeed(42)

    # Apply K-Means clustering to the pixel colors
    _, labels, centers = cv2.kmeans( 
        data,
        K,
        None,
        criteria,
        10,
        cv2.KMEANS_PP_CENTERS #Uses K-Means++ initialization, which generally gives better starting cluster centers
    )
#k=2-> 2 clrs and so on

    # Convert cluster centers to 8-bit integers
    centers = np.uint8(centers)

    # Replace each pixel with the color of its cluster center
    segmented = centers[
        labels.flatten()
    ].reshape(rgb.shape)

    return segmented
#define ks 
Ks = [2, 4, 6]
results = [
    segment(k)
    for k in Ks
]    

#Detection using Hough Lines
gray = cv.cvtColor(
    img,
    cv.COLOR_BGR2GRAY
)#for canny edge detection
blur = cv.GaussianBlur(
    gray,
    (5, 5),
    1.2
)
edges = cv.Canny(
    blur,
    60,
    160
)
lines = cv.HoughLinesP( #detects line segments from the Canny edge image
    edges, #canny edges
    1, #Distance resolution in pixels.
    np.pi / 180, #angle resolution
    threshold=65, #A line needs enough Hough votes to be accepted, more th-> less lines
    minLineLength=50,
    maxLineGap=18
)
line_view = img.copy()
accepted = [] #store the lines that passed
H, W = gray.shape
border_margin = 15
if lines is not None:
    lines = lines.reshape(-1, 4)
    for x1, y1, x2, y2 in lines:
angle = abs( #line angle
    np.degrees(
        np.arctan2(
            y2 - y1,
            x2 - x1
        )
    )
)
#remove image border lines
near_left = max(x1, x2) <= border_margin
near_right = min(x1, x2) >= W - border_margin
near_top = max(y1, y2) <= border_margin
near_bottom = min(y1, y2) >= H - border_margin
if near_left or near_right or near_top or near_bottom:
    continue
if angle < 12 or angle > 78:#Keep Horizontal and Vertical Lines
    accepted.append(
    (x1, y1, x2, y2)
)
    cv.line(
    line_view,
    (x1, y1),
    (x2, y2),
    (0, 255, 0),
    2
)
show(
    edges,
    "Canny edges"
)

show(
    line_view,
    f"Accepted Hough lines: {len(accepted)}"
)
kernel = np.ones(
    (5, 5),
    np.uint8
)

closed = cv.morphologyEx(
    edges,
    cv.MORPH_CLOSE,
    kernel,
    iterations=1
)
contours, _ = cv.findContours(
    closed,
    cv.RETR_LIST,
    cv.CHAIN_APPROX_SIMPLE
)
min_area_ratio = 0.025
max_area_ratio = 0.25

min_width = 100
min_height = 80

min_aspect = 1.15
max_aspect = 2.0

for c in contours: #contor -> fine lines seg looks like a monitor
    x, y, w, h = cv.boundingRect(c)
    area = w * h

    area_ratio = (
        area / float(image_area)
    )
    if not (
    min_area_ratio
    <= area_ratio
    <= max_area_ratio):
        continue
    aspect = (
    w / float(max(h, 1))
)
1.15 <= aspect <= 2.0
perimeter = cv.arcLength(
    c,
    True
)
approx = cv.approxPolyDP( #simplifies the contour.
    c,
    0.02 * perimeter,
    True
)
if len(approx) < 4 or len(approx) > 6:
    continue

#iou func-> intersection over union-> overlap of 2 rec remove dup or nested recs
def iou(rect1, rect2):

    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2

    xa = max(x1, x2)
    ya = max(y1, y2)

    xb = min(
        x1 + w1,
        x2 + w2
    )

    yb = min(
        y1 + h1,
        y2 + h2
    )

    if xb <= xa or yb <= ya:
        return 0.0

    intersection = (
        (xb - xa)
        * (yb - ya)
    )

    area1 = w1 * h1
    area2 = w2 * h2

    union = (
        area1
        + area2
        - intersection
    )

    return intersection / float(
        max(union, 1)
    )
screens = sorted(
    screens,
    key=lambda r: r[2] * r[3],
    reverse=True
)
for candidate in screens:
    overlap = iou(
    candidate,
    existing
)
    if overlap > 0.45:
       center_distance = np.hypot(
    candidate_center[0] - existing_center[0],
    candidate_center[1] - existing_center[1]
)
        if center_distance < 30:
            screens = sorted(
    screens,
    key=lambda r: r[0]
)
for i, (x, y, w, h) in enumerate(screens):
    cv.rectangle(
    result,
    (x, y),
    (x + w, y + h),
    (0, 0, 255),
    3
)
    cv.putText(
    result,
    f"Screen {i + 1}",
    ...
)
# Asset Tracking using SIFT
REFERENCE_PATH = 'box.png'
SCENE_PATH = 'box_in_scene.png'
ref = cv.imread(REFERENCE_PATH)
scene = cv.imread(SCENE_PATH)
sift = cv.SIFT_create( #find distinctive points (key points)
    nfeatures=2500 #Maximum approximately 2500 important features considered.
)
gray_ref = cv.cvtColor(
    ref,
    cv.COLOR_BGR2GRAY
)

gray_scene = cv.cvtColor(
    scene,
    cv.COLOR_BGR2GRAY
)
#Keypoints + descriptors -> keypoints = where are the imp locations in the image(around the obj), des = what is the visual pattern near these kp (numerical representation of each keypoint.)
kp1, des1 = sift.detectAndCompute(
    gray_ref,
    None
)
kp2, des2 = sift.detectAndCompute(
    gray_scene,
    None
)
#FLANN matcher -> Fast Library for Approximate Nearest Neighbors, compare the descriptors  of both images 
FLANN_INDEX_KDTREE = 1
index_params = dict(
    algorithm=FLANN_INDEX_KDTREE, #use KD tree for searching
    trees=5
)
search_params = dict( #seaching acc and speed, checks high-> better matching but slow
    checks=80
)
flann = cv.FlannBasedMatcher(
    index_params,
    search_params
)
#KNN matching
knn = flann.knnMatch(
    des1,
    des2,
    k=2 #2 closest match
)
#Lowe's ratio test
good = []
for pair in knn:
    m, n = pair #2 closest, best match
    if len(pair) < 2:
        continue
    if m.distance < 0.72 * n.distance:
    good.append(m)
match_view = cv.drawMatches( #matches display (matches passed lowe's test)
    ref,
    kp1,
    scene,
    kp2,
    good,
    None,
    flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)
if len(good) < 4: #min 4 points for homography
    result = scene.copy()
MIN_MATCHES = 10
if len(good) < MIN_MATCHES:
src = np.float32([
    kp1[m.queryIdx].pt #index of ref des and .pt point(x,y)
    for m in good
]).reshape(-1, 1, 2)
dst = np.float32([
    kp2[m.trainIdx].pt
    for m in good
]).reshape(-1, 1, 2)
#find homography -> prespective for image 1 transformed on image 2
H, inliers = cv.findHomography(
    src,
    dst,
    cv.RANSAC, #to reject incorrect matches
    5.0 #RANSAC reproj error threshold
)
inlier_mask = inliers.ravel().astype(bool)
inlier_count = int(
    np.sum(inlier_mask)
)
inlier_ratio = (
    inlier_count / float(len(good))
)
#Reliability condition
MIN_INLIERS = 8
MIN_INLIER_RATIO = 0.35

h, w = ref.shape[:2]
box = np.float32([
    [0, 0],
    [w - 1, 0],
    [w - 1, h - 1],
    [0, h - 1]
]).reshape(-1, 1, 2)
#corners of image 1 projection on image 2
projected = cv.perspectiveTransform(
    box,
    H
)
projected_int = np.round(
    projected
).astype(np.int32)
polygon_area = abs( #projected obj area is resonable or not
    cv.contourArea(
        projected_int
    )
)
cv.polylines( #draw he detected obj
    result,
    [projected_int],
    True,
    (0, 255, 0),
    4,
    cv.LINE_AA
)
for i, match in enumerate(good):
#draw inliner points
    if inlier_mask[i]:
        cv.circle(
            result,
            (int(x), int(y)),
            4,
            (255, 0, 0),
            -1
        )

#Anomaly Detection in Sensor Data using Wavelets
np.random.seed(42)
N = 1000
t = np.arange(N) #create sensor signals
signal = (
    0.45 * np.sin(2 * np.pi * t / 90) #slow signals
    + 0.18 * np.sin(2 * np.pi * t / 19) #faster signals
    + 0.22 * np.random.randn(N) #noise
)
true_anomalies = np.array([
    170,
    171,
    425,
    700,
    701,
    850
])
#WAVELET-> break frequency/details in diff levels
wavelet = 'db4' #Daubechies 4 wavelet.
level = 4
coeffs = pywt.wavedec( #wavelet decomposition
    signal,
    wavelet,
    level=level
)
sigma = (
    np.median(np.abs(coeffs[-1])) #less outlier affected 
    / 0.6745
)
threshold = ( #wavelet threshold
    sigma
    * np.sqrt(2 * np.log(len(signal)))
)
#soft thresholding 
denoised_coeffs = [ 
    coeffs[0]
]
denoised_coeffs += [
    pywt.threshold(
        c,
        threshold,
        mode='soft'
    )
    for c in coeffs[1:]
]
#reconstruction of Denoised signal
denoised = pywt.waverec(
    denoised_coeffs,
    wavelet
)[:len(signal)]
#Residual = Original − Normal/Denoised signal -> remaining unusual behavior
residual = signal - denoised
residual_median = np.median(residual)
mad = np.median( #Median Absolute Deviation-> measure spread of residual
    np.abs(
        residual - residual_median
    )
)
robust_sigma = 1.4826 * mad #approx to standard-deviation-like scale
anomaly_threshold = (
    3.5 * robust_sigma
) # if residual is out of this threshold range (both +ve and -ve)-> anomaly
anomaly_mask = (
    np.abs(
        residual - residual_median
    )
    > anomaly_threshold
)
idx = np.where(
    anomaly_mask
)[0]

#object recognition in a video using SIFT features and homography.
REFERENCE_PATH = 'object.png'
VIDEO_PATH = 'object_video.mp4'
OUTPUT_PATH = 'task4_recognized.mp4'
ref = cv.imread(
    REFERENCE_PATH
)
cap = cv.VideoCapture(
    VIDEO_PATH
)
sift = cv.SIFT_create( #SIFT detector
    nfeatures=1800
)
gref = cv.cvtColor(
    ref,
    cv.COLOR_BGR2GRAY
)
kp1, des1 = sift.detectAndCompute( #key points and desc of obj
    gref,
    None
)
if des1 is None or len(kp1) < 4: #check if enough feature are found for homography
    raise RuntimeError(
        "Not enough SIFT features in the reference object."
    )
#get video properties-> frame rate
fps = cap.get(cv.CAP_PROP_FPS)
W = int( #dimensions of video
    cap.get(cv.CAP_PROP_FRAME_WIDTH)
)
H = int(
    cap.get(cv.CAP_PROP_FRAME_HEIGHT)
)
fourcc = cv.VideoWriter_fourcc( #video output creation
    *'mp4v'
)
out = cv.VideoWriter( #video writer->Every processed frame will eventually be written 
    OUTPUT_PATH,
    fourcc,
    fps,
    (W, H)
)
flann = cv.FlannBasedMatcher( #find similar descriptors
    dict(
        algorithm=1,
        trees=5
    ),
    dict(
        checks=60
    )
)
detected_frames = 0 # frames where the object was successfully recognized.
frame_count = 0 #total processed frames
#ivject Recognition thresholds
MIN_GOOD_MATCHES = 10
MIN_INLIERS = 8
MIN_INLIER_RATIO = 0.35

while True:
    ok, frame = cap.read() #read next vid frame
    if not ok:
        break #end of frame
    frame_count += 1
    gray_frame = cv.cvtColor( #SIFT works with grayscle
    frame,
    cv.COLOR_BGR2GRAY
)
    kp2, des2 = sift.detectAndCompute(
    gray_frame,
    None
)
    pairs = flann.knnMatch( #match the obj in vid frames
    des1,
    des2,
    k=2
)
    good = []

for pair in pairs:

    if len(pair) < 2:
        continue

    m, n = pair

    if m.distance < 0.72 * n.distance:
        good.append(m)
        
MIN_GOOD_MATCHES = 10
if len(good) >= MIN_GOOD_MATCHES:
src = np.float32([ #get coord of vid frame
            kp1[m.queryIdx].pt
            for m in good
        ]).reshape(-1, 1, 2)
#object localization-> homography -> how points in the reference image transform into the video frame.
M, inliers = cv.findHomography(
    src,
    dst,
    cv.RANSAC,
    5.0
)
#validate inliners
inlier_mask = (
    inliers.ravel().astype(bool)
)
inlier_ratio = (
    inlier_count
    / float(len(good))
)
if (
    inlier_count >= MIN_INLIERS
    and inlier_ratio >= MIN_INLIER_RATIO
):
    h, w = ref.shape[:2]
    poly = cv.perspectiveTransform(
    box,
    M   #project in vid
)
    poly = np.round(
    poly
).astype(np.int32)
    polygon_area = abs(
    cv.contourArea(poly)
)
    cv.polylines( #draw
    frame,
    [poly],
    True,
    (0, 255, 0),
    4,
    cv.LINE_AA
)
    cv.putText(
    frame,
    "OBJECT RECOGNIZED",
    (20, 40),
    cv.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 255, 0),
    3
)
    cap.release()
    out.release()
    
#panoramic image using SIFT-style feature matching through OpenCV’s Stitcher
IMAGE_PATHS = [
    'left(1).jpg',
    'right(1).jpg'
]
#panorama stitcher-> recieves images and combine them image1+image2
stitcher = cv.Stitcher_create(
    cv.Stitcher_PANORAMA
)
#feature extr, matching, camera estimation, image wraping etc
status, pano = stitcher.stitch( #now sticthe images status-> stiched or not, pano-> result image
    images
)
if status != cv.Stitcher_OK:
    raise RuntimeError(
    f"Stitching failed with status {status}. "
    "Ensure the images have approximately 30-50% "
    "overlap and contain enough textured content."
)
gray_pano = cv.cvtColor(
    pano,
    cv.COLOR_BGR2GRAY
)
_, mask = cv.threshold(
    gray_pano,
    1,
    255,
    cv.THRESH_BINARY
)
nonzero = cv.findNonZero(
    mask
)
if nonzero is None:
    raise RuntimeError(
    "Panorama contains no valid image pixels."
)
 x, y, w, h = cv.boundingRect( #smallest rectangle containing all valid pixels.
    nonzero
)   
cropped = pano[
    y:y + h,
    x:x + w
]

#Detection using Hough Lines
H, W = img.shape[:2]
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
blur = cv.GaussianBlur(
    gray,
    (5, 5),
    0
)
edges = cv.Canny(
    blur,
    50,
    150
)
roi = np.array([[ # trapezoidal ROI
    (int(0.10 * W), H),
    (int(0.45 * W), int(0.60 * H)),
    (int(0.55 * W), int(0.60 * H)),
    (int(0.90 * W), H)
]], dtype=np.int32)
mask = np.zeros_like(edges)
cv.fillPoly( #roi contains the coordinates of a polygon.
    mask,
    roi,
    255
)
roi_edges = cv.bitwise_and(
    edges,
    mask
)
lines = cv.HoughLinesP(
    roi_edges,
    1,
    np.pi / 180,
    threshold=45,
    minLineLength=40,
    maxLineGap=120
) #Probabilistic Hough Line Transform->Find straight lines inside the edge image.
left = []
right = []
lines = lines.reshape(-1, 4)
for x1, y1, x2, y2 in lines:
    if x2 == x1:
        continue
    slope = (
    (y2 - y1)
    / float(x2 - x1)
)
    if abs(slope) < 0.45:
        continue
    if slope > 0:
     left.append(
        (x1, y1, x2, y2)
    )
    else:
     right.append(
        (x1, y1, x2, y2)
    )
def line_parameters(lines):
def extrapolate(lines):
    
#Detection and Counting using Hough Circles
gray = cv.cvtColor(
    img,
    cv.COLOR_BGR2GRAY
)
gray = cv.medianBlur( #remove noise using median blur
    gray,
    7
)
circles = cv.HoughCircles( #Find circular objects in this image
    gray,
    cv.HOUGH_GRADIENT, #The gradient of the image helps identify where intensity changes occur
    dp=1.2, #controls the resolution 
    minDist=35, 
    param1=120, #higher threshold used internally by the Canny edge detector.
    param2=30, #circle detection threshold.
    minRadius=15,
    maxRadius=100
)
count = 0
circles = np.uint16(
    np.around(
        circles[0]
    )
)
count = len(circles)
for i, (x, y, r) in enumerate(
    circles,
    start=1
):
    cv.circle(
    result,
    (x, y),
    r,
    (0, 255, 0),
    3
)
    cv.circle(
    result,
    (x, y),
    3,
    (0, 0, 255),
    -1
)
    cv.putText(
    result,
    str(i),
    (x - 10, y + 5),
    cv.FONT_HERSHEY_SIMPLEX,
    0.6,
    (255, 0, 0),
    2
)
    cv.putText(
    result,
    f'Coin count: {count}',
    (20, 40),
    cv.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 0, 255),
    3
)
#Boundary Detection
fps = cap.get(
    cv.CAP_PROP_FPS
)
if fps <= 0:
    fps = 25
W = int(
    cap.get(
        cv.CAP_PROP_FRAME_WIDTH
    )
)
H = int(
    cap.get(
        cv.CAP_PROP_FRAME_HEIGHT
    )
)
fourcc = cv.VideoWriter_fourcc(
    *'mp4v'
)
out = cv.VideoWriter(
    OUTPUT_PATH,
    fourcc,
    fps,
    (W, H)
)
#define sec zone
zx1 = int(0.30 * W)
zy1 = int(0.35 * H)
zx2 = int(0.72 * W)
zy2 = int(0.90 * H)
#Create background subtractor-> motion detection.
sub = cv.createBackgroundSubtractorMOG2(
    history=300,
    varThreshold=35,
    detectShadows=True
)
k = np.ones(
    (5, 5),
    np.uint8
)
alarm_frames = 0
frame_count = 0
frame_count += 1
fg = sub.apply(frame) #foreground mask (moving obj)
fg[fg == 127] = 0# remove shadow pixel
fg = cv.morphologyEx(
    fg,
    cv.MORPH_OPEN,
    k,
    iterations=1
)
fg = cv.morphologyEx(
    fg,
    cv.MORPH_CLOSE,
    k,
    iterations=2
)
contours, _ = cv.findContours(
    fg,
    cv.RETR_EXTERNAL,
    cv.CHAIN_APPROX_SIMPLE
)
alarm = False
if cv.contourArea(c) < 1200:
    continue
x, y, w, h = cv.boundingRect(c) #rec arond the obj
intersection_width = max(
    0,
    min(x + w, zx2)
    - max(x, zx1)
) #intersection with security zone
if intersection_area > 0:
    alarm = True
    cv.rectangle(
    frame,
    (x, y),
    (x + w, y + h),
    (0, 0, 255),
    3
)
    cv.drawContours(
    frame,
    [c],
    -1,
    (0, 255, 255),
    2
)
    cv.rectangle(
    frame,
    (zx1, zy1),
    (zx2, zy2),
    (255, 0, 0),
    3
)
if alarm:
       alarm_frames += 1
       
#HOG->Histogram of Oriented Gradients, divides an image into cells, calculates gradient orientations, creates orientation histograms, and produces a feature vector representing image shape.
features, hog_image = hog( #HOG returns num feature vec and image (vis HOG feature rep)
    image,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    visualize=True
)
hog_image_rescaled = exposure.rescale_intensity(
    hog_image,
    in_range=(0, 10)
)
#LBP->Local Binary Pattern,
radius = 1
n_points = 8 * radius
lbp_image = feature.local_binary_pattern( #For each center pixel, LBP compares neighboring pixels with it, >= center-> 1 else 0
    image,
    n_points,
    radius,
    method='uniform' #Uniform LBP reduces the number of pattern categories by grouping patterns with limited transitions.
)
lbp_image = feature.local_binary_pattern(image, n_points, radius, method='uniform')

hist, _ = np.histogram(lbp_image.ravel(), bins=np.arange(0, n_points + 3), 
                       range=(0, n_points + 2))
hist = hist.astype("float")
hist /= (hist.sum() + 1e-6)
