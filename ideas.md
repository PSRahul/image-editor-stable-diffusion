# Object Pose Editor

## Assumptions 

1. Assume every image contains only one instance of the queried class. If more than one instance of the object exists, only one of it will be chosen.
2. The class of the queried object needs to be a common object class that is part of the widely available large scale vision dataset. Otherwise, the pretrained models cannot be used in this approach to solve the problem.

## Task 1

### Problem Statement

Mask the queried object with red colour

### Method 1 

Is there a model that can directly identify the object based on the text prompt?

### Method 2

1. Get all the segmentation mask and bounding boxes in the image
2/ Crop the bounding box of each object instance
3. Use a Multi-model model such as CLIP that can classify each bounding box instance to text prompt class
4. Mask the assigned Bounding box with red color

### Solution Approach

For the solution, after an initial review of available open source models, I decided to use Method 1.

## Task 2
### Problem Statement

Change the pose of the object preserving the scene.

### Method 1
1. Inpaint the Image to remove the object
2. Synthesitze novel object view from the polar and azimuth angles
3. Push the object back into the image by anchoring the object center with the bounding box center.


pip install --no-build-isolation -e GroundingDINO
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118




