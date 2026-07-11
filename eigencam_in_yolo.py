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
img = cv2.imread('data\\clean\\images\\cd6af452-e54a1e36.jpg');

img_copy = img.copy()

#Normalize one image to [0, 1] range for visualization
img_normalized = np.float32(img_copy) / 255.0

grayscale_cam = cam(img_normalized, eigen_smooth=True)[0]
cam_image = show_cam_on_image(img_normalized, grayscale_cam, use_rgb=True)


#Display results
plt.imshow(cam_image)
plt.axis('off')
plt.show()