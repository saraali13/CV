# Task 8: Masking with Bitwise Logic

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

image1 = cv2.imread("Sukuna.jpg")
image2 = cv2.imread("high_resolution_image.jfif")

if image1 is None or image2 is None:
    print("Error: One or both images were not found.")

else:
    size = (500, 500)

    image1 = cv2.resize(image1, size)
    image2 = cv2.resize(image2, size)

    # Create circular mask
    mask = np.zeros((500, 500), dtype=np.uint8)

    cv2.circle(mask,(250, 250),170,255,thickness=-1)

    # Take circle from first image
    foreground = cv2.bitwise_and(image1,image1,mask=mask)

    # Take outside of circle from second image
    inverted_mask = cv2.bitwise_not(mask)

    background = cv2.bitwise_and(image2,image2,mask=inverted_mask)

    # Combine both images
    combined = cv2.bitwise_or(foreground,background)

    images = [
        image1,
        image2,
        mask,
        combined
    ]

    titles = [
        "First Image",
        "Second Image",
        "Binary Mask",
        "Combined Result"
    ]

    fig, axes = plt.subplots(1, 4,figsize=(16, 4))

    for axis, current_image, title in zip(axes,images,titles):
        if len(current_image.shape) == 2:
            axis.imshow(current_image,cmap="gray")
        
        else:
            axis.imshow(cv2.cvtColor(current_image,cv2.COLOR_BGR2RGB))

        axis.set_title(title)
        axis.axis("off")

    plt.tight_layout()

    plt.savefig("Task8_output.png",bbox_inches="tight")

    print("Image saved as Task8_output.png")
