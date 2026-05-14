from pathlib import Path
from typing import List, Optional

import rasterio
from torch.utils.data import Dataset
import torch
import numpy as np
from torch.utils.data.dataset import T_co
import glob
import warnings
from .utils import get_means_stds_missing_values, get_indices_of_degree_features
import torchvision.transforms.functional as TF
import h5py
from datetime import date, datetime


class FireSpreadDataset(Dataset):
    LEGACY_NUM_BANDS = 23
    WSTS_REDUCED_NUM_BANDS = 18
    FORECAST_RAW_INDICES = tuple(range(17, 22))
    FORECAST_START_DATE = date(2015, 7, 1)
    ACTIVE_FIRE_RAW_INDEX = 22
    CLOUD_TIME_RAW_INDEX = 23
    CLOUD_ANY_RAW_INDEX = 24
    CLOUD_ALL_RAW_INDEX = 25
    CLOUD_FRACTION_RAW_INDEX = 26
    CLOUD_COUNT_RAW_INDEX = 27
    OBS_COUNT_RAW_INDEX = 28
    ACTIVE_FIRE_ENCODED_INDEX = 25
    VALID_LOSS_WEIGHT_MODES = {
        "none",
        "cloud_hard",
        "cloud_soft",
        "cloud_soft_positive_preserve",
    }

    def __init__(self, data_dir: str, included_fire_years: List[int], n_leading_observations: int,
                 crop_side_length: int, load_from_hdf5: bool, is_train: bool, remove_duplicate_features: bool,
                 stats_years: List[int], n_leading_observations_test_adjustment: Optional[int] = None, 
                 features_to_keep: Optional[List[int]] = None, return_doy: bool = False, is_pad: Optional[bool] = False,
                 pad_size: int = 224, stats_path: Optional[str] = None, loss_weight_mode: str = "none"):
        """_summary_

        Args:
            data_dir (str): _description_ Root directory of the dataset, should contain several folders, each corresponding to a different fire.
            included_fire_years (List[int]): _description_ Years in dataset_root that should be used in this instance of the dataset.
            n_leading_observations (int): _description_ Number of days to use as input observation. 
            crop_side_length (int): _description_ The side length of the random square crops that are computed during training and validation.
            load_from_hdf5 (bool): _description_ If True, load data from HDF5 files instead of TIF. 
            is_train (bool): _description_ Whether this dataset is used for training or not. If True, apply geometric data augmentations. If False, only apply center crop to get the required dimensions.
            remove_duplicate_features (bool): _description_ Remove duplicate static features from all time steps but the last one. Requires flattening the temporal dimension, since after removal, the number of features is not the same across time steps anymore.
            stats_years (List[int]): _description_ Which years to use for computing the mean and standard deviation of each feature. This is important for the test set, which should be standardized using the same statistics as the training set.
            n_leading_observations_test_adjustment (Optional[int], optional): _description_. Adjust the test set to look like it would with n_leading_observations set to this value. 
        In practice, this means that if n_leading_observations is smaller than this value, some samples are skipped. Defaults to None. If None, nothing is skipped. This is especially used for the train and val set. 
            features_to_keep (Optional[List[int]], optional): _description_. List of processed feature indices from 0 to 42, indicating which features to keep. Defaults to None, which means using all features.
            return_doy (bool, optional): _description_. Return the day of the year per time step, as an additional feature. Defaults to False.
            is_pad (book, optional): _description_. Whether to zero-pad images for SwinUnet/TransUnet.
            pad_size (int, optional): Target side length for zero-padding when ``is_pad`` is enabled.
        Raises:
            ValueError: _description_ Raised if input values are not in the expected ranges.
        """
        super().__init__()

        self.stats_years = stats_years
        self.stats_path = stats_path
        self.return_doy = return_doy
        self.features_to_keep = features_to_keep
        self.remove_duplicate_features = remove_duplicate_features
        self.is_train = is_train
        self.load_from_hdf5 = load_from_hdf5
        self.crop_side_length = crop_side_length
        self.n_leading_observations = n_leading_observations
        self.n_leading_observations_test_adjustment = n_leading_observations_test_adjustment
        self.included_fire_years = included_fire_years
        self.data_dir = data_dir
        self.is_pad = is_pad
        self.pad_size = pad_size
        self.loss_weight_mode = loss_weight_mode

        self.validate_inputs()

        # Compute how many samples to skip in the test set, to make it look like it would with n_leading_observations set to this value.
        if self.n_leading_observations_test_adjustment is None:
            self.skip_initial_samples = 0
        else:
            self.skip_initial_samples = self.n_leading_observations_test_adjustment - self.n_leading_observations
            if self.skip_initial_samples < 0:
                raise ValueError(f"n_leading_observations_test_adjustment must be greater than or equal to n_leading_observations, but got {self.n_leading_observations_test_adjustment=} and {self.n_leading_observations=}")

        # Create an inventory of all images in the dataset, and how many data points each fire contains. Since we have multiple data points per fire,
        # we need to know how many data points each fire contains, to be able to map a dataset index to a specific fire.
        self.imgs_per_fire = self.read_list_of_images()
        self.datapoints_per_fire = self.compute_datapoints_per_fire()
        self.length = sum([sum(self.datapoints_per_fire[fire_year].values())
                          for fire_year in self.datapoints_per_fire])

        # Used in preprocessing and normalization. Better to define it once than build/call for every data point
        # The one-hot matrix is used for one-hot encoding of land cover classes
        self.one_hot_matrix = torch.eye(17)
        self.means, self.stds, _ = get_means_stds_missing_values(self.stats_years, self.stats_path)
        self.means = self.means[None, :, None, None]
        self.stds = self.stds[None, :, None, None]
        self.indices_of_degree_features = get_indices_of_degree_features()

    def find_image_index_from_dataset_index(self, target_id) -> (int, str, int):
        """_summary_ Given the index of a data point in the dataset, find the corresponding fire that contains it, 
        and its index within that fire.

        Args:
            target_id (_type_): _description_ Dataset index of the data point.

        Raises:
            RuntimeError: _description_ Raised if the dataset index is out of range.

        Returns:
            (int, str, int): _description_ Year, name of fire, index of data point within fire.
        """

        # Handle negative indexing, e.g. -1 should be the last item in the dataset
        if target_id < 0:
            target_id = self.length + target_id
        if target_id >= self.length:
            raise RuntimeError(
                f"Tried to access item {target_id}, but maximum index is {self.length - 1}.")

        # The index is relative to the length of the full dataset. However, we need to make sure that we know which
        # specific fire the queried index belongs to. We know how many data points each fire contains from
        # self.datapoints_per_fire.
        first_id_in_current_fire = 0
        found_fire_year = None
        found_fire_name = None
        for fire_year in self.datapoints_per_fire:  
            # Corrects an error in dataset loading that also impacted the published paper's results.
            # The nested loop runs through all years (outer loop) and through all fires in these years (inner loop). If the inner loop found the searched-after fire, it would break. However, the outer loop would continue to the next year and update found_fire_year and found_fire_name before breaking in that second year.
            if found_fire_year is None: 
                for fire_name, datapoints_in_fire in self.datapoints_per_fire[fire_year].items():
                    if target_id - first_id_in_current_fire < datapoints_in_fire:
                        found_fire_year = fire_year
                        found_fire_name = fire_name
                        break
                    else:
                        first_id_in_current_fire += datapoints_in_fire
        in_fire_index = target_id - first_id_in_current_fire

        return found_fire_year, found_fire_name, in_fire_index

    def load_imgs(self, found_fire_year, found_fire_name, in_fire_index):
        """_summary_ Load the images corresponding to the specified data point from disk.

        Args:
            found_fire_year (_type_): _description_ Year of the fire that contains the data point.
            found_fire_name (_type_): _description_ Name of the fire that contains the data point.
            in_fire_index (_type_): _description_ Index of the data point within the fire.

        Returns:
            _type_: _description_ (x,y) or (x,y,doy) tuple, depending on whether return_doy is True or False. 
            x is a tensor of shape (n_leading_observations, n_features, height, width), containing the input data. 
            y is a tensor of shape (height, width) containing the binary next day's active fire mask.
            doy is a tensor of shape (n_leading_observations) containing the day of the year for each observation.
        """

        in_fire_index += self.skip_initial_samples
        end_index = (in_fire_index + self.n_leading_observations + 1)

        if self.load_from_hdf5:
            hdf5_path = self.imgs_per_fire[found_fire_year][found_fire_name][0]
            with h5py.File(hdf5_path, 'r') as f:
                imgs = f["data"][in_fire_index:end_index]
                img_dates = self.decode_img_dates(f["data"].attrs.get("img_dates", []))
                if self.return_doy:
                    doys = self.img_dates_to_doys(img_dates[in_fire_index:(end_index-1)])
                    doys = torch.Tensor(doys)
            x, y = np.split(imgs, [-1], axis=0)
            target_img = y[0]
            x = self.canonicalize_raw_features(x, img_dates=img_dates[in_fire_index:(end_index-1)])
            target_img = self.canonicalize_raw_features(target_img, img_dates=img_dates[end_index-1:end_index])
            # Last image's active fire mask is used as label, rest is input data
            y = target_img[self.get_active_fire_raw_index(target_img.shape[0]), ...]
        else:
            imgs_to_load = self.imgs_per_fire[found_fire_year][found_fire_name][in_fire_index:end_index]
            imgs = []
            img_dates = []
            for img_path in imgs_to_load:
                with rasterio.open(img_path, 'r') as ds:
                    imgs.append(ds.read())
                img_dates.append(img_path.split("/")[-1].split("_")[0].replace(".tif", ""))
            x = np.stack(imgs[:-1], axis=0)
            target_img = imgs[-1]
            x = self.canonicalize_raw_features(x, img_dates=img_dates[:-1])
            target_img = self.canonicalize_raw_features(target_img, img_dates=img_dates[-1:])
            y = target_img[self.get_active_fire_raw_index(target_img.shape[0]), ...]

        loss_weight = self.compute_loss_weight(target_img, y)

        if self.return_doy and self.uses_loss_weights():
            return x, y, doys, loss_weight
        if self.return_doy:
            return x, y, doys
        if self.uses_loss_weights():
            return x, y, loss_weight
        return x, y

    def __getitem__(self, index):

        found_fire_year, found_fire_name, in_fire_index = self.find_image_index_from_dataset_index(
            index)
        loaded_imgs = self.load_imgs(
            found_fire_year, found_fire_name, in_fire_index)

        if self.return_doy:
            if self.uses_loss_weights():
                x, y, doys, loss_weight = loaded_imgs
            else:
                x, y, doys = loaded_imgs
                loss_weight = None
        else:
            if self.uses_loss_weights():
                x, y, loss_weight = loaded_imgs
            else:
                x, y = loaded_imgs
                loss_weight = None

        x, y, loss_weight = self.preprocess_and_augment(x, y, loss_weight)

        # Remove duplicate static features, which can greatly reduce the number of features, since we use 
        # one-hot encoded landcover types. The result would have different amounts of feature channels per 
        # time step, therefore, we flatten the temporal dimension.
        if self.remove_duplicate_features and self.n_leading_observations > 1:
            x = self.flatten_and_remove_duplicate_features_(x)
        # Discard features that we don't want to use
        elif self.features_to_keep is not None:
            if len(x.shape) != 4:
                raise NotImplementedError(f"Removing features is only implemented for 4D tensors, but got {x.shape=}.")
            x = x[:, self.features_to_keep, ...]

        if self.return_doy and self.uses_loss_weights():
            return x, y, doys, loss_weight
        if self.return_doy:
            return x, y, doys
        if self.uses_loss_weights():
            return x, y, loss_weight
        return x, y

    def __len__(self):
        return self.length

    def validate_inputs(self):
        if self.n_leading_observations < 1:
            raise ValueError("Need at least one day of observations.")
        if self.return_doy and not self.load_from_hdf5:
            raise NotImplementedError(
                "Returning day of year is only implemented for hdf5 files.")
        if self.loss_weight_mode not in self.VALID_LOSS_WEIGHT_MODES:
            raise ValueError(
                f"Invalid loss_weight_mode {self.loss_weight_mode!r}. "
                f"Expected one of {sorted(self.VALID_LOSS_WEIGHT_MODES)}."
            )
        if self.n_leading_observations_test_adjustment is not None:
            if self.n_leading_observations_test_adjustment < self.n_leading_observations:
                raise ValueError(
                    "n_leading_observations_test_adjustment must be greater than or equal to n_leading_observations.")
            if self.n_leading_observations_test_adjustment < 1:
                raise ValueError(
                    "n_leading_observations_test_adjustment must be greater than or equal to 1. Value 1 is used for having a single observation as input.")

    def read_list_of_images(self):
        """_summary_ Create an inventory of all images in the dataset.

        Returns:
            _type_: _description_ Returns a dictionary mapping integer years to dictionaries. 
            These dictionaries map names of fires that happened within the respective year to either
            a) the corresponding list of image files (in case hdf5 files are not used) or
            b) the individual hdf5 file for each fire.
        """
        imgs_per_fire = {}
        for fire_year in self.included_fire_years:
            imgs_per_fire[fire_year] = {}

            if not self.load_from_hdf5:
                fires_in_year = glob.glob(f"{self.data_dir}/{fire_year}/*/")
                fires_in_year.sort()
                for fire_dir_path in fires_in_year:
                    fire_name = fire_dir_path.split("/")[-2]
                    fire_img_paths = glob.glob(f"{fire_dir_path}/*.tif")
                    fire_img_paths.sort()
                    
                    imgs_per_fire[fire_year][fire_name] = fire_img_paths

                    if len(fire_img_paths) == 0:
                        warnings.warn(f"In dataset preparation: Fire {fire_year}: {fire_name} contains no images.",
                                      RuntimeWarning)
            else:
                fires_in_year = glob.glob(
                    f"{self.data_dir}/{fire_year}/*.hdf5")
                fires_in_year.sort()
                for fire_hdf5 in fires_in_year:
                    fire_name = Path(fire_hdf5).stem
                    imgs_per_fire[fire_year][fire_name] = [fire_hdf5]

        return imgs_per_fire

    def compute_datapoints_per_fire(self):
        """_summary_ Compute how many data points each fire contains. This is important for mapping a dataset index to a specific fire.

        Returns:
            _type_: _description_ Returns a dictionary mapping integer years to dictionaries. 
            The dictionaries map the fire name to the number of data points in that fire.
        """
        datapoints_per_fire = {}
        for fire_year in self.imgs_per_fire:
            datapoints_per_fire[fire_year] = {}
            for fire_name, fire_imgs in self.imgs_per_fire[fire_year].items():
                if not self.load_from_hdf5:
                    n_fire_imgs = len(fire_imgs) - self.skip_initial_samples
                else:
                    # Catch error case that there's no file
                    if not fire_imgs:
                        n_fire_imgs = 0
                    else:
                        with h5py.File(fire_imgs[0], 'r') as f:
                            n_fire_imgs = len(f["data"]) - self.skip_initial_samples
                # If we have two days of observations, and a lead of one day,
                # we can only predict the second day's fire mask, based on the first day's observation
                datapoints_in_fire = n_fire_imgs - self.n_leading_observations
                if datapoints_in_fire <= 0:
                    warnings.warn(
                        f"In dataset preparation: Fire {fire_year}: {fire_name} does not contribute data points. It contains "
                        f"{len(fire_imgs)} images, which is too few for a lead of {self.n_leading_observations} observations.",
                        RuntimeWarning)
                    datapoints_per_fire[fire_year][fire_name] = 0
                else:
                    datapoints_per_fire[fire_year][fire_name] = datapoints_in_fire
        return datapoints_per_fire

    def standardize_features(self, x):
        """_summary_ Standardizes the input data, using the mean and standard deviation of each feature. 
        Some features are excluded from this, which are the degree features (e.g. wind direction), and the land cover class.
        The binary active fire mask is also excluded, since it's added after standardization.

        Args:
            x (_type_): _description_ Input data, of shape (time_steps, features, height, width)

        Returns:
            _type_: _description_ Standardized input data, of shape (time_steps, features, height, width)
        """

        x = (x - self.means) / self.stds

        return x

    def preprocess_and_augment(self, x, y, loss_weight=None):
        """_summary_ Preprocesses and augments the input data. 
        This includes: 
        1. Slight preprocessing of active fire features, if loading from TIF files.
        2. Geometric data augmentation.
        3. Expanding degree features into sin/cos pairs, to preserve circular information.
        4. Standardization of features. 
        5. Addition of the binary active fire mask, as an addition to the fire mask that indicates the time of detection. 
        6. One-hot encoding of land cover classes.

        Args:
            x (_type_): _description_ Input data, of shape (time_steps, features, height, width)
            y (_type_): _description_ Target data, next day's binary active fire mask, of shape (height, width)

        Returns:
            _type_: _description_
        """

        x, y = torch.Tensor(x), torch.Tensor(y)
        if loss_weight is not None:
            loss_weight = torch.Tensor(loss_weight)

        # Preprocessing that has been done in HDF files already
        if not self.load_from_hdf5:

            # Active fire masks have nans where no detections occur. In general, we want to replace NaNs with
            # the mean of the respective feature. Since the NaNs here don't represent missing values, we replace
            # them with 0 instead.
            active_fire_idx = self.get_active_fire_raw_index(x.shape[1])
            x[:, active_fire_idx, ...] = torch.nan_to_num(x[:, active_fire_idx, ...], nan=0)
            y = torch.nan_to_num(y, nan=0.0)

            # Turn event detection times from HHMM to hour of day.
            x[:, active_fire_idx, ...] = torch.floor_divide(x[:, active_fire_idx, ...], 100)
            if self.has_cloud_bands(x.shape[1]):
                x[:, self.CLOUD_TIME_RAW_INDEX, ...] = torch.nan_to_num(
                    x[:, self.CLOUD_TIME_RAW_INDEX, ...], nan=0
                )
                x[:, self.CLOUD_TIME_RAW_INDEX, ...] = torch.floor_divide(
                    x[:, self.CLOUD_TIME_RAW_INDEX, ...], 100
                )
                x[:, self.CLOUD_ANY_RAW_INDEX, ...] = torch.nan_to_num(
                    x[:, self.CLOUD_ANY_RAW_INDEX, ...], nan=0
                )

        y = (y > 0).long()

        # Augmentation has to come before normalization, because we have to correct the angle features when we change
        # the orientation of the image.
        if self.is_train:
            x, y, loss_weight = self.augment(x, y, loss_weight)
        else:
            x, y, loss_weight = self.center_crop_x32(x, y, loss_weight)
        
        # If using a model that expects images of larger size, use zero-padding 
        if self.is_pad:
            x, y, loss_weight = self.zero_pad_to_size(x, y, loss_weight, desired_size=self.pad_size)
        
        # Compute binary mask of active fire pixels before normalization changes what 0 means. 
        active_fire_idx = self.get_active_fire_raw_index(x.shape[1])
        binary_af_mask = (x[:, active_fire_idx:active_fire_idx + 1, ...] > 0).float()

        x = self.standardize_features(x)

        # Some features take values in [0,360] degrees. Expanding them to sin/cos pairs preserves
        # circular information while keeping values near 0 and 360 close in feature space.
        x = self.encode_degree_features(x)

        # Adds the binary fire mask immediately after active fire time, keeping
        # processed IDs 41 and 42 stable when cloud bands are appended.
        binary_insert_idx = self.get_binary_active_fire_insert_index(x.shape[1])
        x = torch.cat([x[:, :binary_insert_idx, ...], binary_af_mask, x[:, binary_insert_idx:, ...]], axis=1)

        # Replace NaN values with 0, thereby essentially setting them to the mean of the respective feature.
        x = torch.nan_to_num(x, nan=0.0)

        # Create land cover class one-hot encoding, put it where the land cover integer was
        new_shape = (x.shape[0], x.shape[2], x.shape[3],
                     self.one_hot_matrix.shape[0])
        # -1 because land cover classes start at 1
        landcover_classes_flattened = x[:, 16, ...].long().flatten() - 1
        landcover_encoding = self.one_hot_matrix[landcover_classes_flattened].reshape(
            new_shape).permute(0, 3, 1, 2)
        x = torch.concatenate(
            [x[:, :16, ...], landcover_encoding, x[:, 17:, ...]], dim=1)

        return x, y, loss_weight

    def augment(self, x, y, loss_weight=None):
        """_summary_ Applies geometric transformations: 
          1. random square cropping, preferring images with a) fire pixels in the output and b) (with much less weight) fire pixels in the input
          2. rotate by multiples of 90°
          3. flip horizontally and vertically
        Adjustment of angles is done as in https://github.com/google-research/google-research/blob/master/simulation_research/next_day_wildfire_spread/image_utils.py

        Args:
            x (_type_): _description_ Input data, of shape (time_steps, features, height, width)
            y (_type_): _description_ Target data, next day's binary active fire mask, of shape (height, width)

        Returns:
            _type_: _description_
        """
    
        # Need square crop to prevent rotation from creating/destroying data at the borders, due to uneven side lengths.
        # Try several crops, prefer the ones with most fire pixels in output, followed by most fire_pixels in input
        best_n_fire_pixels = -1
        best_crop = (None, None)

        for i in range(10):
            top = np.random.randint(0, x.shape[-2] - self.crop_side_length)
            left = np.random.randint(0, x.shape[-1] - self.crop_side_length)
            x_crop = TF.crop(
                x, top, left, self.crop_side_length, self.crop_side_length)
            y_crop = TF.crop(
                y, top, left, self.crop_side_length, self.crop_side_length)
            if loss_weight is not None:
                loss_weight_crop = TF.crop(
                    loss_weight, top, left, self.crop_side_length, self.crop_side_length)
            else:
                loss_weight_crop = None

            # We really care about having fire pixels in the target. But if we don't find any there,
            # we care about fire pixels in the input, to learn to predict that no new observations will be made,
            # even though previous days had active fires.
            active_fire_idx = self.get_active_fire_raw_index(x_crop.shape[1])
            n_fire_pixels = x_crop[:, active_fire_idx, ...].mean() + \
                1000 * y_crop.float().mean()
            if n_fire_pixels > best_n_fire_pixels:
                best_n_fire_pixels = n_fire_pixels
                best_crop = (x_crop, y_crop, loss_weight_crop)

        x, y, loss_weight = best_crop

        hflip = bool(np.random.random() > 0.5)
        vflip = bool(np.random.random() > 0.5)
        rotate = int(np.floor(np.random.random() * 4))
        if hflip:
            x = TF.hflip(x)
            y = TF.hflip(y)
            if loss_weight is not None:
                loss_weight = TF.hflip(loss_weight)
            # Adjust angles
            x[:, self.indices_of_degree_features, ...] = 360 - \
                x[:, self.indices_of_degree_features, ...]

        if vflip:
            x = TF.vflip(x)
            y = TF.vflip(y)
            if loss_weight is not None:
                loss_weight = TF.vflip(loss_weight)
            # Adjust angles
            x[:, self.indices_of_degree_features, ...] = (
                180 - x[:, self.indices_of_degree_features, ...]) % 360

        if rotate != 0:
            angle = rotate * 90
            x = TF.rotate(x, angle)
            y = torch.unsqueeze(y, 0)
            y = TF.rotate(y, angle)
            y = torch.squeeze(y, 0)
            if loss_weight is not None:
                loss_weight = torch.unsqueeze(loss_weight, 0)
                loss_weight = TF.rotate(loss_weight, angle)
                loss_weight = torch.squeeze(loss_weight, 0)

            # Adjust angles
            x[:, self.indices_of_degree_features, ...] = (x[:, self.indices_of_degree_features,
                                                          ...] - 90 * rotate) % 360
        return x, y, loss_weight

    def center_crop_x32(self, x, y, loss_weight=None):
        """_summary_ Crops the center of the image to side lengths that are a multiple of 32, 
        which the ResNet U-net architecture requires. Only used for computing the test performance.

        Args:
            x (_type_): _description_
            y (_type_): _description_

        Returns:
            _type_: _description_
        """
        T, C, H, W = x.shape
        H_new, W_new = self.crop_side_length, self.crop_side_length
        #H_new = H//32 * 32
        #W_new = W//32 * 32

        x = TF.center_crop(x, (H_new, W_new))
        y = TF.center_crop(y, (H_new, W_new))
        if loss_weight is not None:
            loss_weight = TF.center_crop(loss_weight, (H_new, W_new))
        
        return x, y, loss_weight

    def zero_pad_to_size(self, x, y, loss_weight=None, desired_size=224):
        """Zero-pads the input images to ensure they fit the desired size."""
        T, C, H, W = x.shape
        if H < desired_size or W < desired_size:
            pad_height = max(0, desired_size - H)
            pad_width = max(0, desired_size - W)

            padding = (pad_width // 2, pad_width - pad_width // 2,  
                    pad_height // 2, pad_height - pad_height // 2)  
            
            x = torch.nn.functional.pad(x, padding)
            y = torch.nn.functional.pad(y, padding)
            if loss_weight is not None:
                loss_weight = torch.nn.functional.pad(loss_weight, padding)

        return x, y, loss_weight
    
    def flatten_and_remove_duplicate_features_(self, x):
        """_summary_ For a simple U-Net, static and forecast features can be removed everywhere but in the last time step
        to reduce the number of features. Since that would result in different numbers of channels for different
        time steps, we flatten the temporal dimension. 
        Also discards features that we don't want to use. 

        Args:
            x (_type_): _description_ Input tensor data of shape (n_leading_observations, n_features, height, width)

        Returns:
            _type_: _description_
        """
        static_feature_ids, dynamic_feature_ids = self.get_static_and_dynamic_features_to_keep(self.features_to_keep)
        dynamic_feature_ids = torch.tensor(dynamic_feature_ids).int()

        x_dynamic_only = x[:-1, dynamic_feature_ids, :, :].flatten(start_dim=0, end_dim=1)
        x_last_day = x[-1, self.features_to_keep, ...].squeeze(0)

        return torch.cat([x_dynamic_only, x_last_day], axis=0)

    @staticmethod
    def get_static_and_dynamic_feature_ids():
        """_summary_ Returns the indices of static and dynamic features.
        Static features include topographical features and one-hot encoded land cover classes.

        Returns:
            _type_: _description_ Tuple of lists of integers, first list contains static feature indices, second list contains dynamic feature indices.
        """
        static_feature_ids = [13, 14, 15, 16] + list(range(18, 35))
        dynamic_feature_ids = list(range(13)) + [17] + list(range(35, 43))
        return static_feature_ids, dynamic_feature_ids

    @staticmethod
    def get_static_and_dynamic_features_to_keep(features_to_keep:Optional[List[int]]):
        """_summary_ Returns the indices of static and dynamic features that should be kept, based on the input list of feature indices to keep.

        Args:
            features_to_keep (Optional[List[int]]): _description_

        Returns:
            _type_: _description_
        """
        static_features_to_keep, dynamic_features_to_keep = FireSpreadDataset.get_static_and_dynamic_feature_ids()
        
        if type(features_to_keep) == list:
            max_feature_id = max(features_to_keep) if features_to_keep else 42
            if max_feature_id >= 43:
                dynamic_features_to_keep = list(range(13)) + [17] + list(range(35, max_feature_id + 1))
            dynamic_features_to_keep = list(set(dynamic_features_to_keep) & set(features_to_keep))
            dynamic_features_to_keep.sort()

        if type(features_to_keep) == list:
            static_features_to_keep = list(set(static_features_to_keep) & set(features_to_keep))
            static_features_to_keep.sort()

        return static_features_to_keep, dynamic_features_to_keep

    @staticmethod
    def get_n_features(n_observations:int, features_to_keep:Optional[List[int]], deduplicate_static_features:bool):
        """_summary_ Computes the number of features that the dataset will have after preprocessing, 
        considering the number of input observations, which features to keep or discard, and whether to deduplicate static features.

        Args:
            n_observations (int): _description_
            features_to_keep (Optional[List[int]]): _description_
            deduplicate_static_features (bool): _description_

        Returns:
            _type_: _description_ If deduplicate_static_features is True, returns the total number of features, flattened across all time steps. 
            Otherwise, returns the number of features per time step.
        """
        static_features_to_keep, dynamic_features_to_keep = FireSpreadDataset.get_static_and_dynamic_features_to_keep(features_to_keep)

        n_static_features = len(static_features_to_keep)
        n_dynamic_features = len(dynamic_features_to_keep)
        n_all_features = n_static_features + n_dynamic_features

        # If we deduplicate static features, we remove them from all time steps but the last one.
        # The last day then gets dynamic and static features. All other days only get dynamic features. 
        n_features = (int(deduplicate_static_features)*n_dynamic_features)*(n_observations-1) + n_all_features

        return n_features


    @staticmethod
    def img_dates_to_doys(img_dates):
        """_summary_ Converts a list of date strings to day of year values.

        Args:
            img_dates (_type_): _description_ List of date strings

        Returns:
            _type_: _description_ List of day of year values
        """
        date_format = "%Y-%m-%d"
        # In old preprocessing, the dates still had a TIF file extension, which is also removed here.
        return [datetime.strptime(img_date.replace(".tif", ""), date_format).timetuple().tm_yday for img_date in img_dates]

    @staticmethod
    def map_channel_index_to_features():
        """_summary_ Maps the channel index to the feature name.

        Returns:
            _type_: _description_
        """
        return {
            0: 'VIIRS band M11',
            1: 'VIIRS band I2',
            2: 'VIIRS band I1',
            3: 'NDVI',
            4: 'EVI2',
            5: 'total precipitation',
            6: 'wind speed',
            7: 'wind direction sin',
            8: 'wind direction cos',
            9: 'minimum temperature',
            10: 'maximum temperature',
            11: 'energy release component',
            12: 'specific humidity',
            13: 'slope',
            14: 'aspect sin',
            15: 'aspect cos',
            16: 'elevation',
            17: 'pdsi',
            18: 'Landcover_Type1',
            19: 'Landcover_Type2',
            20: 'Landcover_Type3',
            21: 'Landcover_Type4',
            22: 'Landcover_Type5',
            23: 'Landcover_Type6',
            24: 'Landcover_Type7',
            25: 'Landcover_Type8',
            26: 'Landcover_Type9',
            27: 'Landcover_Type10',
            28: 'Landcover_Type11',
            29: 'Landcover_Type12',
            30: 'Landcover_Type13',
            31: 'Landcover_Type14',
            32: 'Landcover_Type15',
            33: 'Landcover_Type16',
            34: 'Landcover_Type17',
            35: 'forecast total_precipitation',
            36: 'forecast wind speed',
            37: 'forecast wind direction sin',
            38: 'forecast wind direction cos',
            39: 'forecast temperature',
            40: 'forecast specific humidity',
            41: 'active fire',
            42: 'binary active fire mask',
            43: 'cloud_time',
            44: 'cloud_any',
            45: 'cloud_all',
            46: 'cloud_fraction',
            47: 'cloud_count',
            48: 'obs_count',
        }

    @staticmethod
    def encode_degree_features(x: torch.Tensor) -> torch.Tensor:
        """Expand raw degree channels into sin/cos pairs.

        Args:
            x: Raw preprocessed tensor of shape (time_steps, channels, height, width).

        Returns:
            Tensor where each degree feature has been replaced by a sin/cos pair
            in deterministic order.
        """
        angles_rad = torch.deg2rad(x[:, [7, 13, 19], ...])
        encoded_angle_features = torch.stack(
            [
                torch.sin(angles_rad[:, 0, ...]),
                torch.cos(angles_rad[:, 0, ...]),
                torch.sin(angles_rad[:, 1, ...]),
                torch.cos(angles_rad[:, 1, ...]),
                torch.sin(angles_rad[:, 2, ...]),
                torch.cos(angles_rad[:, 2, ...]),
            ],
            dim=1,
        )

        return torch.cat(
            [
                x[:, :7, ...],
                encoded_angle_features[:, :2, ...],
                x[:, 8:13, ...],
                encoded_angle_features[:, 2:4, ...],
                x[:, 14:19, ...],
                encoded_angle_features[:, 4:, ...],
                x[:, 20:, ...],
            ],
            dim=1,
        )

    @classmethod
    def has_cloud_bands(cls, num_raw_bands: int) -> bool:
        return num_raw_bands >= cls.OBS_COUNT_RAW_INDEX + 1

    @classmethod
    def get_active_fire_raw_index(cls, num_raw_bands: int) -> int:
        if num_raw_bands > cls.ACTIVE_FIRE_RAW_INDEX:
            return cls.ACTIVE_FIRE_RAW_INDEX
        return num_raw_bands - 1

    @classmethod
    def canonicalize_raw_features(cls, x: np.ndarray, img_dates=None) -> np.ndarray:
        """Map raw tensors onto the canonical 23-band layout and mark unavailable forecasts.

        WSTS_v2 stores 2012 through pre-July-2015 forecast channels as zero-filled
        23-band tensors. Those zeros are missing-data placeholders, not measured
        forecasts, so convert them to NaN before standardization. Older HDF5 exports
        may still use an 18-band layout with forecast channels physically absent.
        """
        if x.shape[-3] == cls.WSTS_REDUCED_NUM_BANDS:
            canonical_shape = list(x.shape)
            canonical_shape[-3] = cls.LEGACY_NUM_BANDS
            dtype = x.dtype if np.issubdtype(x.dtype, np.floating) else np.float32
            canonical = np.full(canonical_shape, np.nan, dtype=dtype)
            canonical[..., :17, :, :] = x[..., :17, :, :]
            canonical[..., cls.ACTIVE_FIRE_RAW_INDEX, :, :] = x[..., -1, :, :]
            x = canonical

        if x.shape[-3] == cls.LEGACY_NUM_BANDS and img_dates is not None:
            x = cls.mark_unavailable_forecasts(x, img_dates)

        return x

    @classmethod
    def mark_unavailable_forecasts(cls, x: np.ndarray, img_dates) -> np.ndarray:
        parsed_dates = cls.parse_img_dates(img_dates)
        if not parsed_dates:
            return x
        if x.ndim == 3:
            parsed_dates = parsed_dates[:1]
            if parsed_dates and parsed_dates[0] < cls.FORECAST_START_DATE:
                x = x.astype(np.float32, copy=True)
                x[list(cls.FORECAST_RAW_INDICES), ...] = np.nan
            return x

        if x.ndim != 4:
            return x

        missing_date_indices = [
            idx for idx, parsed_date in enumerate(parsed_dates[:x.shape[0]])
            if parsed_date < cls.FORECAST_START_DATE
        ]
        if not missing_date_indices:
            return x

        x = x.astype(np.float32, copy=True)
        x[np.ix_(missing_date_indices, list(cls.FORECAST_RAW_INDICES))] = np.nan
        return x

    @classmethod
    def parse_img_dates(cls, img_dates):
        parsed_dates = []
        for raw_date in img_dates:
            if isinstance(raw_date, bytes):
                raw_date = raw_date.decode("utf-8")
            try:
                parsed_dates.append(date.fromisoformat(str(raw_date)[:10]))
            except ValueError:
                continue
        return parsed_dates

    @classmethod
    def decode_img_dates(cls, raw_dates):
        decoded = []
        for raw_date in raw_dates:
            if isinstance(raw_date, bytes):
                raw_date = raw_date.decode("utf-8")
            decoded.append(str(raw_date)[:10])
        return decoded

    @classmethod
    def get_binary_active_fire_insert_index(cls, num_encoded_bands: int) -> int:
        if num_encoded_bands > cls.ACTIVE_FIRE_ENCODED_INDEX:
            return cls.ACTIVE_FIRE_ENCODED_INDEX + 1
        return num_encoded_bands

    def uses_loss_weights(self) -> bool:
        return self.loss_weight_mode != "none"

    def compute_loss_weight(self, target_img: np.ndarray, y: np.ndarray) -> Optional[np.ndarray]:
        if not self.uses_loss_weights():
            return None
        if not self.has_cloud_bands(target_img.shape[0]):
            raise ValueError(
                f"loss_weight_mode={self.loss_weight_mode!r} requires cloud-augmented "
                f"data with at least {self.OBS_COUNT_RAW_INDEX + 1} bands, got {target_img.shape[0]}."
            )

        y_positive = np.nan_to_num(y, nan=0.0) > 0
        if self.loss_weight_mode == "cloud_hard":
            cloud_all = np.nan_to_num(target_img[self.CLOUD_ALL_RAW_INDEX], nan=0.0)
            all_cloud_negative = (~y_positive) & (cloud_all >= 1.0)
            return np.where(all_cloud_negative, 0.0, 1.0).astype(np.float32)

        cloud_count = np.nan_to_num(target_img[self.CLOUD_COUNT_RAW_INDEX], nan=0.0)
        obs_count = np.nan_to_num(target_img[self.OBS_COUNT_RAW_INDEX], nan=0.0)
        clear_count = np.maximum(obs_count - cloud_count, 0.0)
        soft_weight = np.ones_like(obs_count, dtype=np.float32)
        observed = obs_count > 0
        soft_weight[observed] = np.clip(
            clear_count[observed] / obs_count[observed],
            0.0,
            1.0,
        ).astype(np.float32)

        if self.loss_weight_mode == "cloud_soft_positive_preserve":
            soft_weight[y_positive] = 1.0
        return soft_weight.astype(np.float32)

    def get_generator_for_hdf5(self):
        """_summary_ Creates a generator that is used to turn the dataset into HDF5 files. It applies a few 
        preprocessing steps to the active fire features that need to be applied anyway, to save some computation.

        Yields:
            _type_: _description_ Generator that yields tuples of (year, fire_name, img_dates, lnglat, img_array) 
            where img_array contains all images available for the respective fire, preprocessed such 
            that event-time detection channels are converted to hours. lnglat contains
            longitude and latitude of the center of the image.
        """

        for year, fires_in_year in self.imgs_per_fire.items():
            for fire_name, img_files in fires_in_year.items():
                imgs = []
                lnglat = None
                for img_path in img_files:
                    with rasterio.open(img_path, 'r') as ds:
                        imgs.append(ds.read())
                        if lnglat is None:
                            lnglat = ds.lnglat()
                x = np.stack(imgs, axis=0)

                # Get dates from filenames
                img_dates = [img_path.split("/")[-1].split("_")[0].replace(".tif", "")
                             for img_path in img_files]

                # Active fire masks have nans where no detections occur. In general, we want to replace NaNs with
                # the mean of the respective feature. Since the NaNs here don't represent missing values, we replace
                # them with 0 instead.
                active_fire_idx = self.get_active_fire_raw_index(x.shape[1])
                x[:, active_fire_idx, ...] = np.nan_to_num(x[:, active_fire_idx, ...], nan=0)

                # Turn event detection times from HHMM to hour of day.
                x[:, active_fire_idx, ...] = np.floor_divide(x[:, active_fire_idx, ...], 100)
                if self.has_cloud_bands(x.shape[1]):
                    x[:, self.CLOUD_TIME_RAW_INDEX, ...] = np.nan_to_num(
                        x[:, self.CLOUD_TIME_RAW_INDEX, ...], nan=0
                    )
                    x[:, self.CLOUD_TIME_RAW_INDEX, ...] = np.floor_divide(
                        x[:, self.CLOUD_TIME_RAW_INDEX, ...], 100
                    )
                    x[:, self.CLOUD_ANY_RAW_INDEX, ...] = np.nan_to_num(
                        x[:, self.CLOUD_ANY_RAW_INDEX, ...], nan=0
                    )
                x = self.canonicalize_raw_features(x, img_dates=img_dates)
                yield year, fire_name, img_dates, lnglat, x
