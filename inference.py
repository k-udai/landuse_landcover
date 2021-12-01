from tensorflow.keras.models import load_model
from data_preprocess import one_hot_encode, load_band_stats
from rasterio import plot

import matplotlib.pyplot as plt
import numpy as np
import os
import rasterio


def visualize(model_path, im_path, data_dir):
    """
    :param data_dir: data_dir where band_stats.json is located
    :param model_path: h5 model path
    :param im_path: path of image for prediction
    :return:
    """
    im = rasterio.open(im_path).read()
    tci = np.stack((im[3, :, :], im[2, :, :], im[1, :, :]), axis=0)
    band_means, band_stds = load_band_stats(os.path.join(data_dir, 'band_stats.json'))
    im_norm = np.transpose((im - band_means) / band_stds, (1, 2, 0)).reshape(1, 256, 256, 13)
    model = load_model(model_path)
    pred = np.argmax(model.predict(im_norm), axis=-1).reshape(256, 256)
    act = np.argmax(one_hot_encode(rasterio.open(im_path.replace("s2_", "lc_")).read()), axis=-1)

    f, axarr = plt.subplots(1, 3, figsize=(20, 5))
    plot.show(tci / 2 ** 16, adjust=True, ax=axarr[0])
    plot.show(pred, ax=axarr[1])
    plot.show(act, ax=axarr[2])
    
