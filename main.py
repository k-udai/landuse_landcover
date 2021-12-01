from eval import evaluate
from data_download import data_download
from data_preprocess import get_inp_img_paths, create_splits, save_band_statistics, DataLoader, load_band_stats
from models import unet_model
from train import train

import glob
import os


# Press the green button in the gutter to run the script.
def main(data_dir, ckpt_dir, download=False, preprocess=False, mode="evaluation", img_size=(256, 256),
         band_indices=None, num_classes=12, batch_size=32, epochs=20,
         eval_metric="mean_iou"):
    if band_indices is None:
        band_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    if download:
        data_download(data_dir)
    if preprocess:
        input_img_paths = get_inp_img_paths(data_dir, seasons=None)
        create_splits(input_img_paths, data_dir)
        save_band_statistics(input_img_paths, data_dir)
    mean, std = load_band_stats(os.path.join(data_dir, 'band_stats.json'))
    if mode == "training":
        model = unet_model(img_size+(len(band_indices),), num_classes)
        train(model, img_size, data_dir, ckpt_dir, batch_size, epochs,
              mean, std, band_indices)
        return "Model Trained"
    elif mode == "evaluation":
        model_path = sorted(glob.glob(os.path.join(ckpt_dir, "*")))[-1]
        e = evaluate(model_path, img_size, data_dir, batch_size, mean, std,
                 band_indices, eval_metric, num_classes)
        return str(e)

base_dir = os.path.realpath(__file__)
data_dir = os.path.join(base_dir, 'data')
ckpt_dir = os.path.join(base_dir, 'model_checkpoints')
#data_dir = '/home/ubuntu/mount/external_drive_1/SEN12MS/data'
#ckpt_dir = '/home/ubuntu/SEN12MS/saved_models/'
print(main(data_dir, ckpt_dir, mode="evaluation"))
