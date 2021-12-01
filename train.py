from data_preprocess import DataLoader
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import MeanIoU, Accuracy


import numpy as np
import os


def train(model, img_size, data_dir, ckpt_dir, batch_size, epochs,
          band_means, band_stds, band_indices):

    train_input_paths = np.load(os.path.join(data_dir, "train_input_paths.npy"))
    valid_input_paths = np.load(os.path.join(data_dir, "valid_input_paths.npy"))

    train_loader = DataLoader(batch_size,
                              img_size,
                              train_input_paths,
                              band_means,
                              band_stds,
                              band_indices)
    valid_loader = DataLoader(batch_size,
                              img_size,
                              valid_input_paths,
                              band_means,
                              band_stds,
                              band_indices)

    model.compile(optimizer=Adam(learning_rate=0.005),
                  loss=CategoricalCrossentropy(),
                  metrics=[MeanIoU(num_classes=12), Accuracy()])

    if not os.path.isdir(ckpt_dir):
        os.mkdir(ckpt_dir)

    filepath = os.path.join(ckpt_dir, "model_{epoch:02d}_{val_accuracy:.2f}.h5")
    checkpoint = ModelCheckpoint(filepath, save_freq='epoch')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)

    model.fit(train_loader,
              epochs=epochs,
              validation_data=valid_loader,
              callbacks=[checkpoint, reduce_lr],
              shuffle=False)

