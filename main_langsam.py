from PIL import Image
from lang_sam import LangSAM
import matplotlib.pyplot as plt
 
model = LangSAM()
image_pil = Image.open("sample_input_images/chair.jpg").convert("RGB")
text_prompt = "chair"
masks, boxes, phrases, logits = model.predict(image_pil, text_prompt)
masks=masks.numpy()[0] *1
plt.imshow(masks)
plt.show()
print(len(masks))