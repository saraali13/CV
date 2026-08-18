# Task 7: Matrix-Based Rotation and Adaptive Thresholding

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

image_path = "high_contrast_image.png"

gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if gray_image is None:
    print("Error: Image not found. Check the file path.")

else:
    _, global_threshold = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

    adaptive_threshold = cv2.adaptiveThreshold(gray_image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

    height, width = adaptive_threshold.shape

    center = (width // 2, height // 2)

    rotation_matrix = cv2.getRotationMatrix2D(center, 45, 0.8)

    rotated_image = cv2.warpAffine(adaptive_threshold,rotation_matrix,(width, height),borderValue=255)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(global_threshold, cmap="gray")
    axes[0].set_title("Global Threshold")
    axes[0].axis("off")

    axes[1].imshow(adaptive_threshold, cmap="gray")
    axes[1].set_title("Adaptive Threshold")
    axes[1].axis("off")

    axes[2].imshow(rotated_image, cmap="gray")
    axes[2].set_title("Adaptive Threshold Rotated 45 Degrees")
    axes[2].axis("off")

    plt.tight_layout()

    plt.savefig("Task7_output.png", bbox_inches="tight")

    print("Image saved as Task7_output.png")
