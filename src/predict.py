import os
import keras
import numpy as np
import cv2
import segmentation_models as sm

from src.data_creator import DataCreator
from src.data_augmentation import (
    get_preprocessing,
    get_data_pred_aug,
    )

FRAMES_PRED_IN_PATH = os.environ.get("FRAMES_PRED_IN_PATH")
MASKS_PRED_OUT_PATH = os.environ.get("MASKS_PRED_OUT_PATH")

# Frames&masks input dimension
IN_HEIGHT = int(os.environ.get("IN_HEIGHT"))
IN_WIDTH = int(os.environ.get("IN_WIDTH"))
MODEL_PATH = os.environ.get("MODEL_PATH")
MODEL = os.environ.get("MODEL")
BACKBONE = os.environ.get("BACKBONE")
LR = 1e-3

# load file containing list of labels
LABELS_FILE = os.environ.get("LABELS_FILE")
with open(LABELS_FILE, 'r') as file:
    CLASSES = list(file)
if not CLASSES:
    raise Exception(f"Unable to load labels file {CLASSES}")

# LOAD TESTING SET
# Preprocessing: Scaling/Normalization
preprocess_input = sm.get_preprocessing(BACKBONE)
# Create set
pred_dataset = DataCreator(
    images_path=FRAMES_PRED_IN_PATH,
    masks_path=MASKS_PRED_OUT_PATH,
    classes=CLASSES,
    augmentation=get_data_pred_aug(in_h=IN_HEIGHT, in_w=IN_WIDTH),
    preprocessing=get_preprocessing(preprocess_input),
    prediction=True,
)

# SET PARAMS
n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES) + 1)  # case for binary and multiclass segmentation
activation = 'sigmoid' if n_classes == 1 else 'softmax'

# CREATE MODEL & LOAD WEIGHTS
model = sm.Unet(backbone_name=BACKBONE,
                encoder_weights='imagenet',
                classes=n_classes,
                decoder_block_type='transpose',
                activation=activation,
                #decoder_filters=(1024, 512, 256, 128, 64),
                encoder_freeze=False,
                )
# OPTIMIZER
optimizer = keras.optimizers.Adam(LR)
# LOSSES
focal_dice_loss = sm.losses.binary_focal_dice_loss if n_classes == 1 else sm.losses.categorical_focal_dice_loss
# METRICS
metrics = [sm.metrics.IOUScore(), sm.metrics.FScore()]
# COMPILE
model.compile(optimizer, focal_dice_loss, metrics)
# ------- Summary
model.summary()

# load best weights
model.load_weights(MODEL_PATH)

for i, fname in enumerate(os.listdir(FRAMES_PRED_IN_PATH)):
    image = pred_dataset[i]
    image = np.expand_dims(image, axis=0)
    # Prediction
    probs = model.predict(image)
    pred_mask = np.rint(probs).squeeze()

    cv2.imwrite(os.path.join(MASKS_PRED_OUT_PATH, fname), pred_mask * 255)


################ Continue with writing predicts back to the original size
print("DONE")


