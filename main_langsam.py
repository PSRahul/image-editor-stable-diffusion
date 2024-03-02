from PIL import Image
from lang_sam import LangSAM

model = LangSAM()
image_pil = Image.open("sample_input_images/chair.jpg").convert("RGB")
text_prompt = "chair"
masks, boxes, phrases, logits = model.predict(image_pil, text_prompt)
print(len(masks))