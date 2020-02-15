#======================================================
# EVALUATION
#======================================================

# ---------------------------
# Load environment variables
# ---------------------------
export MODEL_PATH=models/models/vessels_seg_unet_576_576_densenet121_14_02_20_00_00_AM/vessels_seg_unet_576_576_densenet121_14_02_20_00_00_AM.h5

# Input
export FRAMES_TEST_IN_PATH=input/testing/images_in
export MASKS_TEST_IN_PATH=input/testing/mask_in
# Output
export FRAMES_TEST_OUT_PATH=input/testing/images_out
export MASKS_TEST_OUT_PATH=input/testing/mask_out
# Others
export NO_CLASSES=1
export IN_HEIGHT=576
export IN_WIDTH=576 # 560

export LABELS_FILE=labels.txt
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
#export BACKBONE=resnet34

# ---------------------------
# Define script loader
# ---------------------------
export BACKBONE=densenet121 && python -m src.evaluate
export BACKBONE=seresnext50 && python -m src.evaluate
#======================================================