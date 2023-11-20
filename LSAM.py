from PIL import Image
from lang_sam import LangSAM
import supervision as sv
import numpy as np

model = LangSAM()
image_pil = Image.open("cube_pics/1.jpg").convert("RGB")
image_pil = image_pil.resize((300, 300))
text_prompt = "Rubiks Cube"
# sv.plot_image(np.array(image_pil), size=(16,16))
masks, boxes, phrases, logits = model.predict(image_pil, text_prompt)


# sv.plot_images_grid(
#         images=masks[:16],
#     grid_size=(8, 2),
#     size=(16, 16)
# )
for i in range(image_pil.size[0]):
    for j in range(image_pil.size[1]):
        if not masks[0][j][i] == 1:
            image_pil.putpixel((i,j), (0,0,0))

sv.plot_image(np.array(image_pil), size=(16,16))



