from PIL import Image
from lang_sam import LangSAM
import matplotlib.pyplot as plt
import numpy as np
import hydra

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config):
    model = LangSAM()
    image_pil = Image.open(config.image)
    text_prompt = config.class_name
    masks, boxes, phrases, logits = model.predict(image_pil.convert("RGB"), text_prompt)
    masks=masks.numpy()[0]

    image_array=np.array(image_pil)/255.0
    masked_image_array=np.copy(image_array)
    red_image_array=np.ones_like(masked_image_array,dtype=np.float32)
    red_image_array[:, :, 1:] = 0

    masked_image_array=image_array*(1-masks[:, :, np.newaxis])+red_image_array*masks[:, :, np.newaxis]


    plt.imshow(masked_image_array)
    plt.imsave(config.output,masked_image_array)



if __name__ == "__main__":
    main()