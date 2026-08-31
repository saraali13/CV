This task loads corresponding CT and MRI heart images and equalizes them separately to improve their contrast. 
The enhanced scans are converted into color maps and combined using weighted fusion. 
A higher weight is assigned to the CT image to preserve sharp anatomical boundaries, 
while the MRI image contributes soft-tissue information. 
Logarithmic and power-law transformations are then applied to prevent important details from becoming too dark or too bright. 
The CT, MRI, weighted fusion, and final enhanced fusion are displayed side by side and saved in the output folder.
