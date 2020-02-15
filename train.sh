#======================================================
# TRAINING
#======================================================
# ---------------------------
# Load environment variables
# ---------------------------
# Train set
export FRAMES_TRAIN_PATH=input/training/images
export MASKS_TRAIN_PATH=input/training/mask
# Valid set
export FRAMES_VAL_PATH=input/validation/images
export MASKS_VAL_PATH=input/validation/mask

# Others
export NO_CLASSES=1
export TRAIN_BATCH_SIZE=2
export VAL_BATCH_SIZE=4
export TRAIN_STEPS_PER_EPOCH=1100  # train_len(ex. 800 images) = batch_size(20) * steps_per_epoch(40)
export VAL_STEPS_PER_EPOCH=50     # val_len(ex. 200) = batch_size(5) * steps_per_epoch(40)
export EPOCHS=100
export IN_HEIGHT=576
export IN_WIDTH=576 #
export MODEL_PATH=models/
export PROBLEM=vessels_seg
# Define labels within this file without taking into account background
export LABELS_FILE=labels.txt
# If show training history
export PLOT_TRAIN=False

# ---------------------------
# Models
# ---------------------------
# linknet: sm.Linknet,
# unet:    sm.Unet,
# pspnet:  sm.PSPNet,
# fpn:     sm.FPN
export MODEL=unet
# ---------------------------
# Backbones
# ---------------------------
# VGG:          'vgg16' 'vgg19'
# ResNet:	      'resnet18' 'resnet34' 'resnet50' 'resnet101' 'resnet152'
# SE-ResNet:	  'seresnet18' 'seresnet34' 'seresnet50' 'seresnet101' 'seresnet152'
# ResNeXt:	    'resnext50' 'resnext101'
# SE-ResNeXt:	  'seresnext50' 'seresnext101'
# SENet154: 	  'senet154'
# DenseNet:	    'densenet121' 'densenet169' 'densenet201'
# Inception:    'inceptionv3' 'inceptionresnetv2'
# MobileNet:	  'mobilenet' 'mobilenetv2'
# EfficientNet:	'efficientnetb0' 'efficientnetb1' 'efficientnetb2' 'efficientnetb3' 'efficientnetb4' 'efficientnetb5' efficientnetb6' efficientnetb7'
#export BACKBONE=densenet121

# --------------------
# Define script loader
# --------------------
#python -m src.train
export BACKBONE=densenet121 && python -m src.train
export BACKBONE=seresnext50 && python -m src.train
#======================================================