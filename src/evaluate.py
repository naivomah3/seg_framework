import os
import keras
import segmentation_models as sm

from src.visualization import plot_history
from src.data_generator import DataGenerator
from src.data_creator import DataCreator
from src.data_augmentation import (
    get_preprocessing,
    get_data_test_aug,
    )

FRAMES_TEST_IN_PATH = os.environ.get("FRAMES_TEST_IN_PATH")
MASKS_TEST_IN_PATH = os.environ.get("MASKS_TEST_IN_PATH")
FRAMES_TEST_OUT_PATH = os.environ.get("FRAMES_TEST_IN_PATH")
MASKS_TEST_OUT_PATH = os.environ.get("MASKS_TEST_IN_PATH")

# Frames&masks input dimension
IN_HEIGHT = int(os.environ.get("IN_HEIGHT"))
IN_WIDTH = int(os.environ.get("IN_WIDTH"))
MODEL_PATH = os.environ.get("MODEL_PATH")
HIST_PATH = os.environ.get("HIST_PATH")
MODEL = os.environ.get("MODEL")
BACKBONE = os.environ.get("BACKBONE")
LR = 1e-3
PLOT_TRAIN = True if os.environ.get("PLOT_TRAIN") == 'True' else False
# load file containing list of labels
LABELS_FILE = os.environ.get("LABELS_FILE")
with open(LABELS_FILE, 'r') as file:
    CLASSES = list(file)
if not CLASSES:
    raise Exception(f"Unable to load labels file {CLASSES}")

# Plot training scores
if PLOT_TRAIN:
    plot_history(model_path=MODEL_PATH,
                 model_type=MODEL)
    exit()

# LOAD TESTING SET
# Preprocessing: Scaling/Normalization
preprocess_input = sm.get_preprocessing(BACKBONE)
# Create set
test_dataset = DataCreator(
    images_path=FRAMES_TEST_IN_PATH,
    masks_path=MASKS_TEST_IN_PATH,
    classes=CLASSES,
    augmentation=get_data_test_aug(in_h=IN_HEIGHT, in_w=IN_WIDTH),
    preprocessing=get_preprocessing(preprocess_input),
    prediction=False,
)
# Generate set
test_dataloader = DataGenerator(test_dataset, batch_size=1, shuffle=False)

# SET PARAMS
n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES) + 1)  # case for binary and multi label segmentation
activation = 'sigmoid' if n_classes == 1 else 'softmax'
# CREATE MODEL & LOAD WEIGHTS --- freeze only if found your labels within 'imagenet'
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
focal_dice_loss = sm.losses.DiceLoss() if n_classes == 1 else sm.losses.categorical_focal_dice_loss
# METRICS
metrics = [sm.metrics.IOUScore(), sm.metrics.FScore()]
# COMPILE
model.compile(optimizer, focal_dice_loss, metrics)
# ------- Summary
model.summary()
# WEIGHTS
model.load_weights(MODEL_PATH)

# EVALUATE WITH TESTING SET
scores = model.evaluate_generator(test_dataloader, verbose=1, workers=20)

print("Evaluation scores:")
print(f"Loss: {scores[0]:.5}")
for metric, value in zip(metrics, scores[1:]):
    print(f"mean {metric.__name__}: {value:.5}")

