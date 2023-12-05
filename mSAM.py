from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2
import time
from torchvision.ops import box_convert

model = load_model("GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", "GroundingDINO/weights/groundingdino_swint_ogc.pth", device='cpu')
img_num = 12
IMAGE_PATH = f"cube_pics/{img_num}.jpg"
TEXT_PROMPT = "Rubik's Cube"
BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25

image_source, image = load_image(IMAGE_PATH)

start = time.time()
boxes, logits, phrases = predict(
    model=model,
    image=image,
    caption=TEXT_PROMPT,
    box_threshold=BOX_TRESHOLD,
    text_threshold=TEXT_TRESHOLD,
    device='cpu'
)
print("DINO INF: ", time.time() - start)

xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()[0]
print("xywh", xyxy)

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

model_type = "vit_t"
sam_checkpoint = "./MobileSAM/weights/mobile_sam.pt"
# model_type = "vit_b"
# sam_checkpoint = "./sam_vit_b_01ec64.pth"

mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
# mobile_sam.to(device=device)
mobile_sam.eval()

image_bgr = cv2.imread(IMAGE_PATH)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
image_pil = Image.open(IMAGE_PATH).convert("RGB")

predictor = SamPredictor(mobile_sam)
predictor.set_image(image_rgb)

xyxy[0] *= image_pil.size[0]
xyxy[1] *= image_pil.size[1]
xyxy[2] *= image_pil.size[0]
xyxy[3] *= image_pil.size[1]
box = xyxy
print(box)

start = time.time()
masks, _, _ = predictor.predict(box=box, multimask_output=False)
print("mSAM INF:", time.time() - start)

for i in range(image_pil.size[0]):
    for j in range(image_pil.size[1]):
        if not masks[0][j][i] == 1:
            image_pil.putpixel((i,j), (0,0,0))

def show_mask(mask, ax):
    color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
cv2.imwrite("annotated_image.jpg", annotated_frame)


f, axarr = plt.subplots(2+len(masks),1)

image_pil.save(f"isolated_cubes/isolated_cube{img_num}.jpg")

axarr[0].imshow(image_pil)
axarr[1].imshow(annotated_frame)

for i, mask in enumerate(masks):
    show_mask(mask, axarr[i+2])

print("masks number: ", len(masks))

plt.show()
