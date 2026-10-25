import os
import cv2
import numpy as np
import kagglehub

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

#################################################
# DOWNLOAD DATASET
#################################################

print("Downloading dataset...")

pathy = kagglehub.dataset_download(
    "grassknoted/asl-alphabet"
)

print(pathy)