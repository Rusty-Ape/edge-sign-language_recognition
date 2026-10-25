import os
import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# DATASET_PATH = "C:\\Users\\Aise\\Documents\\Sign Langugae\\Sign labels"
DATASET_PATH = "C:\\Users\\Aise\\.cache\\kagglehub\\datasets\\grassknoted\\asl-alphabet\\versions\\1\\asl_alphabet_train\\asl_alphabet_train"

print("Dataset Path:", DATASET_PATH)

di_base_options = python.BaseOptions(
    model_asset_path="hand_landmarker.task"
)

options = vision.HandLandmarkerOptions(
    base_options=di_base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1
)

landmarker = vision.HandLandmarker.create_from_options(options)

classes = sorted([
    d for d in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, d))
])
class_to_idx = {c: i for i, c in enumerate(classes)}

print(classes)
print(class_to_idx)

def extract_features(image_path):

    image = cv2.imread(image_path)
    if image is None:
        return None

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = landmarker.detect(mp_image)

    if not result.hand_landmarks:
        print(f"Couldnt see Landmark for{image_path}")
        return None

    points = np.array(
        [[lm.x, lm.y, lm.z]
         for lm in result.hand_landmarks[0]],
        dtype=np.float32
    )



def run():

    X, y = [], []
    unseen = 0

    for cls in classes:

        folder = os.path.join(DATASET_PATH, cls)

        print(f"Processing {cls}")

        for img_name in os.listdir(folder):

            img_path = os.path.join(folder, img_name)

            features = extract_features(img_path)

            if features is None:
                unseen += 1 

            if features is not None:
                X.append(features)
                y.append(class_to_idx[cls])

    X = np.array(X, dtype=np.float32)
    # y = np.array(y, dtype=np.int64)

    print("Final dataset shape:", X)
    print(unseen)

run()