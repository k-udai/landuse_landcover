from data_preprocess import DataLoader
from metrics import ImageMetrics
from tensorflow.keras.models import load_model

import numpy as np
import os


def evaluate(model_path, img_size, data_dir, batch_size, band_means,
             band_stds, band_indices, metric, num_classes):

    model = load_model(model_path)
    test_input_img_paths = np.load(os.path.join(data_dir, "test_input_paths.npy"))[:2000]
    test_loader = DataLoader(batch_size,
                             img_size,
                             test_input_img_paths,
                             band_means,
                             band_stds,
                             band_indices)
    if metric == "mean_iou":
        iou_list = []
        for j, batch in enumerate(test_loader):
            pred = model.predict(batch[0])
            act = batch[1]
            batch_iou_score = ImageMetrics.semantic_seg_metric(act, pred, metric='iou',
                                                               level='batch', classes='all')
            iou_list.append(batch_iou_score)

        return np.mean(iou_list)

