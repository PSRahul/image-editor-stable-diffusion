# Object Pose Editor

## Assumptions 

1. Assume every image contains only one instance of the queried class. If more than one instance of the object exists, only one of it will be chosen.
2. The class of the queried object needs to be a common object class that is part of the widely available large scale vision dataset. Otherwise, the pretrained models cannot be used in this approach to solve the problem.

## Task 1

## Problem Statement

Mask the queried object with red colour

## Possible Solutions

### Method 1 

Is there a model that can directly identify the object based on the text prompt?

### Method 2

1. Get all the segmentation mask and bounding boxes in the image
2/ Crop the bounding box of each object instance
3. Use a Multi-model model such as [CLIP](https://openai.com/research/clip) that can classify each bounding box instance to text prompt class by similarity metrics from a unified text-image embedding.
4. Mask the assigned Bounding box with red color

## Solution Approach

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
## Problem Statement

Change the pose of the queried object preserving the scene.

## Solution Approach

Task 2.1  Inpaint the Image to remove the object and complete teh scene  
Task 2.2  Synthesize novel object view from the polar and azimuth angles  
Task 2.3 Overlay the object back into the image by anchoring the novel view center with the bounding box center from Task 1.


### Task 2.1 - Inpaint the Image to remove the object and complete teh scene  

For inpainting the scene, I used the [Stable Diffusion Model](https://huggingface.co/runwayml/stable-diffusion-inpainting) from Runway ML and with the Inpainting checkpoint. 

| Input Image | Inpainted Image | 
| :---:   | :---: | 
| ![Alt text](sample_input_images/chair.jpg) | ![Alt text](task_2_inter_output_images/chair.jpg) |

Prompt used is "background"

Unfortunately, this prompt and model did not work for the other images in the sample set.

### Task 2.2 - Synthesize novel object view from the polar and azimuth angles 

For this task, I intended to use the [Zero-1-to-3](https://github.com/cvlab-columbia/zero123) model from Columbia University and Toyota Research Institute.



pip install --no-build-isolation -e GroundingDINO
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118




