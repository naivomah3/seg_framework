import os
from datetime import datetime
import keras
import segmentation_models as sm

from src.data_generator import DataGenerator
from src.data_creator import DataCreator
from src.utils import lr_decay, mk_dir
from src.visualization import plot_history
from src.callbacks import get_callbacks
from src.data_augmentation import (
    get_preprocessing,
    get_data_train_aug,
    get_data_valid_aug )

# Train set
FRAMES_TRAIN_PATH = os.environ.get("FRAMES_TRAIN_PATH")
MASKS_TRAIN_PATH = os.environ.get("MASKS_TRAIN_PATH")
# Validation set
FRAMES_VAL_PATH = os.environ.get("FRAMES_VAL_PATH")
MASKS_VAL_PATH = os.environ.get("MASKS_VAL_PATH")
# Where to save weights
MODEL_PATH = os.environ.get("MODEL_PATH")
# Backbone
MODEL = os.environ.get("MODEL")
BACKBONE = os.environ.get("BACKBONE")
# Batch-size
TRAIN_BATCH_SIZE = int(os.environ.get("TRAIN_BATCH_SIZE"))
VAL_BATCH_SIZE = int(os.environ.get("VAL_BATCH_SIZE"))
# Number of epochs & Learning rate
EPOCHS = int(os.environ.get("EPOCHS"))
LR = 1e-3
# Frames&masks input dimensions
IN_HEIGHT = int(os.environ.get("IN_HEIGHT"))
IN_WIDTH = int(os.environ.get("IN_WIDTH"))
# get problem name for naming history/model
P_NAME = os.environ.get("PROBLEM")
# Plotting history
PLOT_TRAIN = True if os.environ.get("PLOT_TRAIN") == 'True' else False
# load file containing list of classes
LABELS_FILE = os.environ.get("LABELS_FILE")
with open(LABELS_FILE, 'r') as file:
    CLASSES = list(file.read().splitlines())
    print(CLASSES)
if not CLASSES:
    raise Exception(f"Unable to load label file {CLASSES}")

# NETWORK & bunch of parameters
n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES) + 1)  # case for binary and multi label segmentation
activation = 'sigmoid' if n_classes == 1 else 'softmax'
model = sm.Unet(backbone_name=BACKBONE,
                encoder_weights='imagenet',
                classes=n_classes,
                decoder_block_type='transpose',
                activation=activation,
                encoder_freeze=False,
                #decoder_filters=(1024, 512, 256, 128, 64), # case if we want the model overfits :)
                )

# Preprocessing: Scaling/Normalization
preprocess_input = sm.get_preprocessing(BACKBONE)

# DATASET: Training set
train_set_create = DataCreator(
    images_path=FRAMES_TRAIN_PATH,
    masks_path=MASKS_TRAIN_PATH,
    classes=CLASSES,
    #augmentation=get_data_train_aug(in_h=IN_HEIGHT, in_w=IN_WIDTH), # uncomment if augmentation is needed
    preprocessing=get_preprocessing(preprocess_input),
    prediction=False,
)
# DATASET: Validation set
valid_set_create = DataCreator(
    images_path=FRAMES_VAL_PATH,
    masks_path=MASKS_VAL_PATH,
    classes=CLASSES,
    #augmentation=get_data_valid_aug(in_h=IN_HEIGHT, in_w=IN_WIDTH), # uncomment if augmentation is needed
    preprocessing=get_preprocessing(preprocess_input),
    prediction=False,
)
# GENERATOR: Train & Validation generator
train_set_gen = DataGenerator(train_set_create, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
valid_set_gen = DataGenerator(valid_set_create, batch_size=VAL_BATCH_SIZE, shuffle=False)

# OPTIMIZER
optimizer = keras.optimizers.Adam()
# LOSSES
focal_dice_loss = sm.losses.DiceLoss() # if n_classes == 1 else sm.losses.DiceLoss()
# METRICS
metrics = [sm.metrics.IOUScore(), sm.metrics.FScore(), sm.metrics.Recall(), sm.metrics.Precision()]
# COMPILE
model.compile(optimizer, focal_dice_loss, metrics)
# ----
model.summary()
# CALLBACKS
model_name = f"{P_NAME}_{MODEL}_{IN_HEIGHT}_{IN_WIDTH}_{BACKBONE}_{datetime.now().strftime('%d_%m_%y_%H_%M_%p')}"
callbacks = get_callbacks(model_path=MODEL_PATH, model_name=model_name)

# TRAIN
model.fit_generator(train_set_gen,
                    steps_per_epoch=len(train_set_gen),
                    epochs=EPOCHS,
                    callbacks=callbacks,
                    validation_data=valid_set_gen,
                    validation_steps=len(valid_set_gen),
                    use_multiprocessing=True,
                    workers=3,
                    max_queue_size=20,
                    )

# Plot training scores only
if PLOT_TRAIN:
    plot_history(model_path=os.path.join(MODEL_PATH, model_name, f"{model_name}.csv"),
                 model_type=MODEL)
    exit()
