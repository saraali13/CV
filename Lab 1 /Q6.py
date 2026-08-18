# Task 6: Alpha Blending and Typography

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

image = cv2.imread("high_resolution_image.jfif")

if image is None:
    print("Error: Image not found. Check the file path.")

else:
    height, width = image.shape[:2]
    overlay = image.copy()

    box_top = int(height * 0.80)

    cv2.rectangle(overlay,(0, box_top),(width, height),(255, 0, 0),-1)

    result = cv2.addWeighted(overlay, 0.5, image, 0.5, 0)

    font_scale = max(0.7, width / 1000)

    text_y = box_top + (height - box_top) // 2

    cv2.putText(result,"Lab task 6",(25, text_y),cv2.FONT_HERSHEY_SIMPLEX,font_scale,(255, 255, 255),2,cv2.LINE_AA)

    rgb_result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 6))
    plt.imshow(rgb_result)
    plt.title("Image with Transparent Caption")
    plt.axis("off")

    plt.savefig("Task6_output.png", bbox_inches="tight")
    print("Image saved as Task6_output.png")
