from pathlib import Path
import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

YOLO_CAM_ROOT = Path("C:\\Research\\repos\\YOLO-26-CAM")
sys.path.insert(0, str(YOLO_CAM_ROOT))

from yolo_cam.eigen_cam import EigenCAM
from yolo_cam.utils.image import show_cam_on_image

#load the model
model = YOLO("yolo11n.pt").cuda()  # Load the YOLO model and move it to GPU

#define the target layers for CAM
target_layers = [model.model.model[-4]]

#initialize the EigenCAM object
cam = EigenCAM(model, target_layers, task="od")

#load an image
#input_image_path = Path('data/proper_patched/images/cd6af452-e54a1e36.jpg')
input_image_path = Path('data/clean/images/cd6af452-e54a1e36.jpg')
img = cv2.imread(str(input_image_path))

#resize the image to 128x128
img = cv2.resize(img, (128, 128))

#check the height, width and the channels of the input image
height, width, channels = img.shape
print(height, width, channels)

img_copy = img.copy()

#Normalize one image to [0, 1] range for visualization
img_normalized = np.float32(img_copy) / 255.0

grayscale_cam = cam(img_normalized, eigen_smooth=True)[0]

# Convert the CAM output to a pure heatmap image without overlaying the input image
heatmap = cv2.normalize(grayscale_cam, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
cam_image = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

height, width, channels = cam_image.shape
print(height, width, channels)

#save the generated heatmap
output_dir = Path('heatmaps/clean')
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / f'{input_image_path.stem}.png'

save_image = np.array(cam_image)
if save_image.dtype != np.uint8:
    save_image = (save_image * 255).astype(np.uint8)
if save_image.ndim == 3 and save_image.shape[2] == 3:
    save_image = cv2.cvtColor(save_image, cv2.COLOR_RGB2BGR)

cv2.imwrite(str(output_path), save_image)
print(f'Saved heatmap to {output_path}')

#Display results
plt.imshow(cv2.cvtColor(cam_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()