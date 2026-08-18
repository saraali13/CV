# Task 3: Safe Image Loading and RGB Visualization

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

image = cv2.imread("Sukuna.jpg")

if image is None:
    print("Error: Image not found. Check the file path.")

else:
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(8, 6))
    plt.imshow(rgb_image)
    plt.title("King of Curses - Ryomen Sukuna",fontsize=16,color="darkred",pad=15)
    plt.axis("off")

    plt.savefig("Sukuna_RGB.png", bbox_inches="tight")
    print("Image saved as Sukuna_RGB.png")
