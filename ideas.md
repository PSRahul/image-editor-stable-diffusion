## Idea Brainstorming

### Assumptions

1. Assume every image contains only one instance of the queried class.
2. That promienent object is quite common, which means it must be part of the large scale datasets that most models are trained on.

### Problem Statement
Task -1 -> Mask the queried object with red colour
Task -2 -> Change the pose of the object preserving the scene.

## Task 1

### Method 1 

Is there a model that can directly identify the object based on the text prompt?

### Method 2

1. Get all the segmentation mask and bounding boxes in the image
2/ Crop the bounding box of each object instance
3. Use a Multi-model model such as CLIP that can classify each bounding box instance to text prompt class
4. Mask the assigned Bounding box with red color

## Task 2

### Method 1
1. Inpaint the Image to remove the object
2. Synthesitze novel object view from the polar and azimuth angles
3. Push the object back into the image by anchoring the object center with the bounding box center.


pip install --no-build-isolation -e GroundingDINO
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118




