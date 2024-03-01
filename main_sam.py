import cv2
import hydra
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig, OmegaConf
from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)
    # sorted_anns = sorted_anns[0]
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones(
        (
            sorted_anns[0]["segmentation"].shape[0],
            sorted_anns[0]["segmentation"].shape[1],
            4,
        )
    )
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann["segmentation"]
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config):
    pass
    sam = sam_model_registry["vit_h"](
        checkpoint=config.model_checkpoint.segment_anything
    )
    mask_generator = SamAutomaticMaskGenerator(
        sam, pred_iou_thresh=0.9, min_mask_region_area=100
    )
    image = cv2.imread(config.input.image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    masks = mask_generator.generate(image)
    print(len(masks))

    plt.figure(figsize=(20, 20))
    plt.imshow(image)
    show_anns(masks)
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
