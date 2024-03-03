import argparse

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from lang_sam import LangSAM


def main():

    parser = argparse.ArgumentParser(description='Fill the class object pixels with red.')
    parser.add_argument('--image', help='Path to the input image containing the target object')
    parser.add_argument('--class_name', help='Class name of the target object to be masked')
    parser.add_argument('--output', help='Path to save the output image with the red mask')

    args = parser.parse_args()


    model = LangSAM()
    image_pil = Image.open(args.image)
    masks, boxes, phrases, logits = model.predict(image_pil.convert("RGB"), args.class_name)
    masks=masks.numpy()[0]

    image_array=np.array(image_pil)/255.0
    masked_image_array=np.copy(image_array)
    red_image_array=np.ones_like(masked_image_array,dtype=np.float32)
    red_image_array[:, :, 1:] = 0

    masked_image_array=image_array*(1-masks[:, :, np.newaxis])+red_image_array*masks[:, :, np.newaxis]

    plt.imsave(args.output,masked_image_array)
    plt.imsave(args.output.split(".jpg")[0]+"_mask.jpg",masks)


    
if __name__ == "__main__":

    main()