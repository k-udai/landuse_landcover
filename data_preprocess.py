from sklearn.model_selection import train_test_split

import glob
import json
import numpy as np
import os
import rasterio
import tensorflow as tf


def get_inp_img_paths(data_dir, seasons=None):
    """
    :str data_dir: SEN12MS data directory
    :list(str) seasons: list paths of files for given seasons
    :return: list of input files' paths
    """
    if seasons is None:
        seasons = ['ROIs2017_winter', 'ROIs1868_summer',
                   'ROIs1970_fall', 'ROIs1158_spring']
    input_img_paths = []
    for season in seasons:
        season_dir = os.path.join(data_dir, season)
        season_img_folders = glob.glob(os.path.join(season_dir, "s2*"))
        season_img_paths = []
        for folder in season_img_folders:
            season_img_paths.extend(glob.glob(os.path.join(folder, "*")))
        input_img_paths.extend(season_img_paths)
    return input_img_paths


def create_splits(input_img_paths, out_dir, test_size=0.1, val_size=0.1):
    """
    :list input_img_paths: input files paths
    :str out_dir: output directory where split files will be written
    :float test_size: from 0 to 1
    :float val_size: from 0 to 1
    :return: writes train, valid and test split in output dir
    """
    train_input_paths, val_test_input_paths = train_test_split(input_img_paths,
                                                               test_size=test_size + val_size,
                                                               random_state=67)
    valid_input_paths, test_input_paths = train_test_split(val_test_input_paths,
                                                           test_size=test_size / (test_size + val_size),
                                                           random_state=53)
    np.save(os.path.join(out_dir, "train_input_paths.npy"), np.array(train_input_paths))
    np.save(os.path.join(out_dir, "valid_input_paths.npy"), np.array(valid_input_paths))
    np.save(os.path.join(out_dir, "test_input_paths.npy"), np.array(test_input_paths))


def save_band_statistics(input_img_paths, out_dir):
    """
    :list input_img_paths: input files paths
    :str out_dir: output dir where band statistics file will be written
    :return: calculate band-wise mean and  standard deviation for given
             input paths and writes it as json in output dir
    """
    # band-wise mean calculation
    sum_arr = np.zeros((13, 256, 256), dtype='float32')
    for path in input_img_paths:
        im_arr = rasterio.open(path).read().astype('float32')
        sum_arr = sum_arr + im_arr
    mean_arr = np.sum(sum_arr, axis=(1, 2)) / (len(input_img_paths) * 256 * 256)
    mean_arr1 = mean_arr.reshape(13, 1, 1)

    # band-wise standard deviation calculation
    diff_sq_arr = np.zeros((13, 256, 256), dtype='float32')
    for path in input_img_paths:
        im_arr = rasterio.open(path).read().astype('float32')
        diff_sq_arr = diff_sq_arr + np.square(im_arr - mean_arr1)
    var_arr = np.sum(diff_sq_arr, axis=(1, 2)) / (len(input_img_paths) * 256 * 256)
    std_arr = np.sqrt(var_arr)

    bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07',
             'B08', 'B8A', 'B09', 'B10', 'B11', 'B12']

    band_stats = {'mean':
        {
            band: mean_value
            for band, mean_value in zip(
            bands, mean_arr)
        },
        'standard_deviation':
            {
                band: std_value
                for band, std_value in zip(
                bands, std_arr)
            }
    }
    with open(os.path.join(out_dir, "band_stats.json"), 'w') as fp:
        json.dump(band_stats, fp)


def image_slice(raster_path, out_dir, patch_size):
    height, width = rasterio.open(raster_path).shape
    im_name = raster_path.split("/")[-1][:-4]
    im_ext = raster_path.split("/")[-1][-3:].upper()
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    for i in range(0, height, patch_size[0]):
        for j in range(0, width, patch_size[1]):
            if im_ext in ['JPG', 'JPEG']:
                out_format = 'JPEG'
            elif im_ext == 'PNG':
                out_format = 'PNG'
            elif im_ext == 'TIF':
                out_format = 'GTIFF'
            else:
                out_format = im_ext
            com_string = "gdal_translate -of " + out_format + " -srcwin " + str(i) + ", " + str(j) + ", " + \
                         str(patch_size[0]) + ", " + str(patch_size[1]) + " " + raster_path + " " + \
                         os.path.join(out_dir, im_name + "_" + str(int(i / patch_size[0])) + "_" + str(
                             int(j / patch_size[1])) + "." + im_ext.lower())
            os.system(com_string + " > /dev/null")


def one_hot_encode(im_arr):
    """
    input: 3D numpy array
    return: one hot encoded float32 tensor with 12 predefined classes
    10->Dense Forest:0
    20->Open Forest:1
    30->Natural Herbaceou:2
    40->Shrublands:3
    36->Herbaceous Croplands:4
    9->Urban and Built-Up Lands:5
    25->Forest/Cropland Mosaics:6
    35->Natural Herbaceous/Croplands Mosaics:7
    2->Permanent Snow and Ice:8
    1->Barren:9
    3->Water Bodies:10
    else->unknown:11
    """
    lccs_lu = im_arr[2, :, :]
    class_map = np.where(lccs_lu == 10, 0,
                np.where(lccs_lu == 20, 1,
                np.where(lccs_lu == 30, 2,
                np.where(lccs_lu == 40, 3,
                np.where(lccs_lu == 36, 4,
                np.where(lccs_lu == 9, 5,
                np.where(lccs_lu == 25, 6,
                np.where(lccs_lu == 35, 7,
                np.where(lccs_lu == 2, 8,
                np.where(lccs_lu == 1, 9,
                np.where(lccs_lu == 3, 10, 11)))))))))))

    return tf.one_hot(class_map, depth=12)


class DataLoader(tf.keras.utils.Sequence):
    """
    Helper to iterate over the data (as Numpy arrays).
    """

    def __init__(self, batch_size, img_size, input_img_paths, band_means, band_stds, band_indices):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.band_means = band_means
        self.band_stds = band_stds
        self.band_indices = band_indices

    def __len__(self):
        return len(self.input_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i: i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size + (len(self.band_indices),), dtype="float32")
        y = np.zeros((self.batch_size,) + self.img_size + (12,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img = np.transpose((rasterio.open(path).read().astype('float32') - self.band_means) / self.band_stds,
                               (1, 2, 0))
            tgt = one_hot_encode(rasterio.open(path.replace("s2_", "lc_")).read())
            x[j] = np.stack((img[:,:,i] for i in self.band_indices), axis=-1)
            y[j] = tgt
        return x, y

    def on_epoch_end(self):
        np.random.shuffle(self.input_img_paths)


def load_band_stats(path):
    with open(path, 'r') as j:
        band_stats = json.loads(j.read())
    bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12']
    mean_dict = band_stats['mean']
    std_dict = band_stats['standard_deviation']
    band_means = np.array([mean_dict.get(band) for band in bands]).reshape(13, 1, 1)
    band_stds = np.array([std_dict.get(band) for band in bands]).reshape(13, 1, 1)
    return band_means, band_stds
