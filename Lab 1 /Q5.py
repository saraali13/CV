# Task 5: Advanced Blurring and NumPy ROI Extraction

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

image = cv2.imread("high_resolution_image.jfif")

if image is None:
    print("Error: Image not found. Check the file path.")

else:
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    blurred_image = cv2.GaussianBlur(rgb_image, (25, 25), 0)

    height, width = rgb_image.shape[:2]
    roi_size = min(300, height, width)

    center_x = width // 2
    center_y = height // 2
    half = roi_size // 2

    start_x = center_x - half
    start_y = center_y - half
    end_x = start_x + roi_size
    end_y = start_y + roi_size

    original_roi = rgb_image[start_y:end_y, start_x:end_x]
    blurred_roi = blurred_image[start_y:end_y, start_x:end_x]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(original_roi)
    axes[0].set_title("Original Center ROI", fontsize=14, color="darkgreen")
    axes[0].axis("off")

    axes[1].imshow(blurred_roi)
    axes[1].set_title("Blurred Center ROI", fontsize=14, color="darkblue")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig("Task5_output.png", bbox_inches="tight")
    print("Image saved as Task5_output.png")
