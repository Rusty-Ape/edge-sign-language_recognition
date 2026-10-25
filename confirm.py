import os
import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


di_base_options = python.BaseOptions(
    model_asset_path="hand_landmarker.task"
)

options = vision.HandLandmarkerOptions(
    base_options=di_base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1
)

landmarker = vision.HandLandmarker.create_from_options(options)

num=0

# image_path =r"C:\Users\Aise\Documents\Sign Langugae\Sign labels\Y\WhatsApp Image 2026-10-13 at 1.52.04 PM.jpeg"
folder = r"C:\Users\Aise\Documents\Sign Langugae\Sign labels\C"
for img_name in os.listdir(folder):
    image_path = os.path.join(folder,img_name)
    image = cv2.imread(image_path)
    if image is None:
        print("No image")

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = landmarker.detect(mp_image)

    if not result.hand_landmarks:
        print("Not seen")
    else:
        points = np.array(
            [[lm.x, lm.y, lm.z]
                for lm in result.hand_landmarks[0]],
            dtype=np.float32
    )
        num+=1
        # print("The point is",points)
print(num)
