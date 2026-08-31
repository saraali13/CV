This task reads an echocardiogram video using OpenCV and processes it frame by frame. 
Each frame is converted to grayscale and enhanced using histogram equalization. 
A JET heatmap is applied to highlight intensity variations, followed by color balancing. 
Logarithmic transformation reveals darker heart chambers, 
while power-law transformation reduces excessively bright ultrasound noise. 
The raw and fully enhanced video frames are displayed side by side in real time, 
and a comparison screenshot is saved in the output folder. Press Q to stop the video.
