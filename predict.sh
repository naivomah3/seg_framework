#======================================================
# PREDICTION
#======================================================
# ------------------------------
# Load environment variables
# ------------------------------
# Input & Output
export FRAMES_PRED_IN_PATH=input/prediction/images
export MASKS_PRED_OUT_PATH=input/prediction/mask
export IN_HEIGHT=576
export IN_WIDTH=560
export LABELS_FILE=labels.txt

export MODEL=unet
export BACKBONE=densenet121

export LABELS_FILE=labels.txt
export MODEL_PATH=models/models/vessels_seg_unet_576_576_densenet121_14_02_20_00_00_AM/vessels_seg_unet_576_576_densenet121_14_02_20_00_00_AM.h5
# ------------------------------
# Define script loader
# ------------------------------
python -m src.predict
#======================================================
