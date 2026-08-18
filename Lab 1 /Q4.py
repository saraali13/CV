# Task 4: Interactive Geometry and Blank Canvas

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

canvas = np.zeros((800, 800, 3), dtype=np.uint8)
center = (canvas.shape[1] // 2, canvas.shape[0] // 2)

radii = [300, 240, 180, 120, 60]
colors = [
    (255, 0, 0),      # Blue in BGR
    (255, 255, 255),  # White
    (0, 0, 255),      # Red
    (255, 255, 255),  # White
    (0, 255, 255),    # Yellow
]

for radius, color in zip(radii, colors):
    cv2.circle(canvas, center, radius, color, thickness=-1)

# Bounding box around the outermost circle
outer_radius = radii[0]
cv2.rectangle(canvas,(center[0] - outer_radius, center[1] - outer_radius),(center[0] + outer_radius, center[1] + outer_radius),(0, 255, 0),thickness=4)

print("Exact center coordinates:", center)

rgb_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(8, 8))
plt.imshow(rgb_canvas)
plt.title("Target Board with Bounding Box")
plt.axis("off")
plt.savefig("Task4_output.png", bbox_inches="tight")
print("Image saved as Task4_output.png")
