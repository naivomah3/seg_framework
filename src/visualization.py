import os
import numpy as np
import pandas as pd
from datetime import date
import matplotlib.pyplot as plt

# Avoid overcoming 1
def round_clip_0_1(x, **kwargs):
    return x.round().clip(0, 1)

# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()

# helper function for data visualization
def denormalize(x):
    """Scale image to range 0..1 for correct plot"""
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)
    x = (x - x_min) / (x_max - x_min)
    x = x.clip(0, 1)
    return x

# Plot training history from file
def plot_history(model_path=None, model_type=None):

    if not os.path.isfile(model_path):
        raise Exception("Provide path to store the model")

    # Load history from file
    hist_df = pd.read_csv(model_path)
    fig = plt.figure(figsize=(14, 20))
    axs = fig.add_subplot(2, 2, 1)
    axs.set_title(f"{model_type} - IoU")
    axs.plot(hist_df['iou_score'], 'b', label='Training IoU')
    axs.plot(hist_df['val_iou_score'], 'r', label='Validation IoU')
    axs.legend()

    axs = fig.add_subplot(2, 2, 2)
    axs.set_title(f"{model_type} - F1-score")
    axs.plot(hist_df['f1-score'], 'b', label='Training F1-score')
    axs.plot(hist_df['val_f1-score'], 'r', label='Validation F1-score')
    axs.legend()

    axs = fig.add_subplot(2, 2, 3)
    axs.set_title(f"{model_type.upper()} - Loss")
    axs.plot(hist_df['loss'], 'b', label='Training diceloss')
    axs.plot(hist_df['val_loss'], 'r', label='Validation diceloss')
    axs.legend()

    plt.show()
