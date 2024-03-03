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
3. Use a Multi-model model such as [CLIP](https://openai.com/research/clip) that can classify each bounding box instance to text prompt class by similarity metrics from a unified text-image embedding.
4. Mask the assigned Bounding box with red color

### Solution Approach

For the solution, after an initial review of available open source models, I decided to use Method 1. I used the [Language Segment-Anything](https://github.com/luca-medeiros/lang-segment-anything) from Luca Medeiros to mask the queried object. This uses a combination of two promienent model
1. [Segment Anything Model (SAM)](https://segment-anything.com/) from Meta AI that aims at zero shot open-world image segmentation.
2. [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) from IDEA-Research that extends the self supervised [DINO](https://github.com/facebookresearch/dino) model from Facebook Research to open set object detection.

 This approach works extremely well for the sample images that are provided with the problem set. Here are the visual outputs of the same:
 
 | Input Image | Masked Image    |  Input Image | Masked Image      | Input Image | Masked Image      |
| :---:   | :---: | :---: | :---: |:---: | :---: |
| ![Alt text](sample_input_images/chair.jpg) | ![Alt text](task1_output_images/chair.jpg)  | ![Alt text](sample_input_images/chair(1).jpg)   |![Alt text](task1_output_images/chair(1).jpg)   |![Alt text](sample_input_images/flower_vase.jpg)   |![Alt text](task1_output_images/flower_vase.jpg)
| ![Alt text](sample_input_images/lamp.jpg)   |![Alt text](task1_output_images/lamp.jpg)  | ![Alt text](sample_input_images/laptop.jpg)   |![Alt text](task1_output_images/laptop.jpg)   |![Alt text](sample_input_images/office_chair.jpg)   |![Alt text](task1_output_images/office_chair.jpg)
| ![Alt text](sample_input_images/sofa.jpg)   |![Alt text](task1_output_images/sofa.jpg)   |![Alt text](sample_input_images/table.jpg)   |![Alt text](task1_output_images/table.jpg)

## Task 2
### Problem Statement

Change the pose of the object preserving the scene.

### Method 1
1. Inpaint the Image to remove the object
2. Synthesitze novel object view from the polar and azimuth angles
3. Push the object back into the image by anchoring the object center with the bounding box center.


pip install --no-build-isolation -e GroundingDINO
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118




