import os
import keras

from src.utils import mk_dir

def get_callbacks(model_path=None, model_name=None,):

    if not mk_dir(model_path, model_name):
        raise Exception("Provide path to store the model")

    callbacks = [
        keras.callbacks.ModelCheckpoint(filepath=os.path.join(model_path, model_name, f"{model_name}.h5"),
                                        monitor='val_iou_score',
                                        save_weights_only=True,
                                        save_best_only=True,
                                        mode='max',
                                        period=2,
                                        verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                          factor=0.9,
                                          patience=5,
                                          min_lr=1e-5,
                                          verbose=1),
        keras.callbacks.CSVLogger(os.path.join(model_path, model_name, f"{model_name}.csv"),
                                  append=True),
        keras.callbacks.EarlyStopping(monitor='val_loss',
                                      mode='auto',
                                      patience=20,
                                      verbose=1),
        # keras.callbacks.LearningRateScheduler(lr_decay)
    ]

    return callbacks