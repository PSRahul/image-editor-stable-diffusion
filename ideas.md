## Idea Brainstorming

### Assumptions

1. Assume every image contains only one prominent object.
2. That promienent object is quite common, which means it must be part of the large scale datasets that most models are trained on.

### Steps
Task -1 -> Mask the object with red colour
Task -2 -> Change the pose of the object preserving the scene.

## Task 1

### Method 1 

Is there a model that can directly identify the object based on the text prompt?

### Method 2

Get all the segmentation mask in the image and match for the class in text prompt

## Task 2

### Method 1
1. Lift the 2D image to 3D object. Rotate it
2. Inpaint the Image
3. Push the object back into the image by anchoring the center.


pip install --no-build-isolation -e GroundingDINO
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118




