# Task 9: Statistical Image Profiling with Pandas

import cv2
import pandas as pd

image = cv2.imread("Sukuna.jpg")

if image is None:
    print("Error: Image not found. Check the file path.")

else:
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    red = rgb_image[:, :, 0].flatten()
    green = rgb_image[:, :, 1].flatten()
    blue = rgb_image[:, :, 2].flatten()

    pixel_data = pd.DataFrame({
        "Red": red,
        "Green": green,
        "Blue": blue
    })

    print("Color Channel Statistical Summary:")
    print(pixel_data.describe())

    print("\nSelected statistics:")
    print(pixel_data.describe().loc[["mean", "min", "max", "std"]])
