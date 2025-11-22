from pathlib import Path

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import Subset, DataLoader
import glob
from .FireSpreadDataset import FireSpreadDataset
from typing import List, Optional, Union


class FireSpreadDataModule(LightningDataModule):

    def __init__(self, data_dir: str, batch_size: int, n_leading_observations: int, n_leading_observations_test_adjustment: int,
                 crop_side_length: int,
                 load_from_hdf5: bool, num_workers: int, remove_duplicate_features: bool, 
                 is_pad: Optional[bool] = False,
                 features_to_keep: Union[Optional[List[int]], str] = None, return_doy: bool = False,
                 data_fold_id: int = 0, non_outlier_indices_path: Optional[str] = None, filter_ignition_train: Optional[bool] = False, filter_ignition_val_test: Optional[bool] = False,
                 ignition_only_train: Optional[bool] = False, ignition_only_val_test: Optional[bool] = False, additional_data: Optional[bool] = False, *args, **kwargs):
        """_summary_ Data module for loading the WildfireSpreadTS dataset.

        Args:
            data_dir (str): _description_ Path to the directory containing the data.
            batch_size (int): _description_ Batch size for training and validation set. Test set uses batch size 1, because images of different sizes can not be batched together.
            n_leading_observations (int): _description_ Number of days to use as input observation. 
            n_leading_observations_test_adjustment (int): _description_ When increasing the number of leading observations, the number of samples per fire is reduced.
              This parameter allows to adjust the number of samples in the test set to be the same across several different values of n_leading_observations, 
              by skipping some initial fires. For example, if this is set to 5, and n_leading_observations is set to 1, the first four samples that would be 
              in the test set are skipped. This way, the test set is the same as it would be for n_leading_observations=5, thereby retaining comparability 
              of the test set.
            crop_side_length (int): _description_ The side length of the random square crops that are computed during training and validation.
            load_from_hdf5 (bool): _description_ If True, load data from HDF5 files instead of TIF. 
            num_workers (int): _description_ Number of workers for the dataloader.
            remove_duplicate_features (bool): _description_ Remove duplicate static features from all time steps but the last one. Requires flattening the temporal dimension, since after removal, the number of features is not the same across time steps anymore.
            features_to_keep (Union[Optional[List[int]], str], optional): _description_. List of feature indices from 0 to 39, indicating which features to keep. Defaults to None, which means using all features.
            return_doy (bool, optional): _description_. Return the day of the year per time step, as an additional feature. Defaults to False.
            data_fold_id (int, optional): _description_. Which data fold to use, i.e. splitting years into train/val/test set. Defaults to 0.
        """
        super().__init__()

        self.n_leading_observations_test_adjustment = n_leading_observations_test_adjustment
        self.data_fold_id = data_fold_id
        self.return_doy = return_doy
        # wandb apparently can't pass None values via the command line without turning them into a string, so we need this workaround
        self.features_to_keep = features_to_keep if type(
            features_to_keep) != str else None
        self.remove_duplicate_features = remove_duplicate_features
        self.num_workers = num_workers
        self.load_from_hdf5 = load_from_hdf5
        self.crop_side_length = crop_side_length
        self.n_leading_observations = n_leading_observations
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_dataset, self.val_dataset, self.test_dataset = None, None, None
        self.is_pad=is_pad
        self.non_outlier_indices_path = non_outlier_indices_path
        self.filter_ignition_train = filter_ignition_train
        self.filter_ignition_val_test = filter_ignition_val_test
        self.ignition_only_train = ignition_only_train
        self.ignition_only_val_test = ignition_only_val_test
        self.additional_data = additional_data


    def keep_ignition(self, dataset):
        ignition_indices = []
        total_samples = len(dataset)
        kept = 0
        
        for idx in range(total_samples):
            sample = dataset[idx]
            inputs = sample[0]  # Shape: [1, 7, 128, 128]
            x_af = inputs[:, -1, :, :]  # Active fire mask
            
            # Check current fire presence
            if torch.sum(x_af == 1) < 1:  # Original filtering condition
                ignition_indices.append(idx)
                kept += 1
    
        # Print detailed statistics
        print(f"Total samples: {total_samples}")
        print(f"Kept samples (ignition): {kept} ({kept/total_samples:.2%})")
        print(f"Discarded samples: {total_samples - kept} ({(total_samples - kept)/total_samples:.2%})")
        return Subset(dataset, ignition_indices)

    def filter_dataset(self, dataset):
        valid_indices = []
        total_samples = len(dataset)
        kept = 0
        for idx in range(total_samples):
            sample = dataset[idx]
            inputs = sample[0]  # Shape: [1, 7, 128, 128] if T=1; but [5*N, 128, 128] if T=5, where N is the number of features
            if len(inputs.shape) == 3:
                x_af = inputs[-1, :, :]
            else:
                x_af = inputs[:, -1, :, :]  # Active fire mask
            
            # Check current fire presence
            if torch.sum(x_af == 1) > 1:  # Original filtering condition
                valid_indices.append(idx)
                kept += 1
        
        # Print detailed statistics
        print(f"Total samples: {total_samples}")
        print(f"Kept samples (current fire): {kept} ({kept/total_samples:.2%})")
        print(f"Discarded samples: {total_samples - kept} ({(total_samples - kept)/total_samples:.2%})")
        
        return Subset(dataset, valid_indices)
        
    def setup(self, stage):
        train_years, val_years, test_years = self.split_fires(
            self.data_fold_id, self.additional_data)
        self.train_dataset = FireSpreadDataset(data_dir=self.data_dir, included_fire_years=train_years,
                                               n_leading_observations=self.n_leading_observations,
                                               n_leading_observations_test_adjustment=None,
                                               crop_side_length=self.crop_side_length,
                                               load_from_hdf5=self.load_from_hdf5, is_train=True,
                                               remove_duplicate_features=self.remove_duplicate_features,
                                               features_to_keep=self.features_to_keep, return_doy=self.return_doy,
                                               stats_years=train_years, is_pad=self.is_pad)
        
        if self.non_outlier_indices_path is not None:
            non_outlier_indices = np.load(self.non_outlier_indices_path).tolist()
            print(f"Subsetting train_loader using {self.non_outlier_indices_path}")
            self.train_dataset = Subset(self.train_dataset, non_outlier_indices)

        if self.filter_ignition_train:
            self.train_dataset = self.filter_dataset(self.train_dataset)

        if self.ignition_only_train:
            self.train_dataset = self.keep_ignition(self.train_dataset)

        
        self.val_dataset = FireSpreadDataset(data_dir=self.data_dir, included_fire_years=val_years,
                                             n_leading_observations=self.n_leading_observations,
                                             n_leading_observations_test_adjustment=None,
                                             crop_side_length=self.crop_side_length,
                                             load_from_hdf5=self.load_from_hdf5, is_train=True,
                                             remove_duplicate_features=self.remove_duplicate_features,
                                             features_to_keep=self.features_to_keep, return_doy=self.return_doy,
                                             stats_years=train_years, is_pad=self.is_pad)
        self.test_dataset = FireSpreadDataset(data_dir=self.data_dir, included_fire_years=test_years,
                                              n_leading_observations=self.n_leading_observations,
                                              n_leading_observations_test_adjustment=self.n_leading_observations_test_adjustment,
                                              crop_side_length=self.crop_side_length,
                                              load_from_hdf5=self.load_from_hdf5, is_train=False,
                                              remove_duplicate_features=self.remove_duplicate_features,
                                              features_to_keep=self.features_to_keep, return_doy=self.return_doy,
                                              stats_years=train_years, is_pad=self.is_pad)

        if self.filter_ignition_val_test:
            self.val_dataset = self.filter_dataset(self.val_dataset)
            self.test_dataset = self.filter_dataset(self.test_dataset)
            
        if self.ignition_only_val_test:
            self.val_dataset = self.keep_ignition(self.val_dataset)
            self.test_dataset = self.keep_ignition(self.test_dataset)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, shuffle=False, num_workers=self.num_workers, pin_memory=True)

    def predict_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

    @staticmethod
    def split_fires(data_fold_id, additional_data):
        """_summary_ Split the years into train/val/test set.

        Args:
            data_fold_id (_type_): _description_ Index of the respective split to choose, see method body for details.

        Returns:
            _type_: _description_
        """
        if not additional_data:

            folds = [(2018, 2019, 2020, 2021),
                 (2018, 2019, 2021, 2020),
                 (2018, 2020, 2019, 2021),
                 (2018, 2020, 2021, 2019),
                 (2018, 2021, 2019, 2020),
                 (2018, 2021, 2020, 2019),
                 (2019, 2020, 2018, 2021),
                 (2019, 2020, 2021, 2018),
                 (2019, 2021, 2018, 2020),
                 (2019, 2021, 2020, 2018),
                 (2020, 2021, 2018, 2019),
                 (2020, 2021, 2019, 2018)]
            train_years = list(folds[data_fold_id][:2])
            val_years = list(folds[data_fold_id][2:3])
            test_years = list(folds[data_fold_id][3:4])
        
        else:
            folds = [(2016, 2017, 2020, 2021, 2018, 2019, 2022, 2023),
                 (2018, 2019, 2022, 2023, 2020, 2021, 2016, 2017),
                 (2016, 2017, 2020, 2021, 2022, 2023, 2018, 2019),
                 (2018, 2019, 2022, 2023, 2016, 2017, 2020, 2021)]
            train_years = list(folds[data_fold_id][:4])
            val_years = list(folds[data_fold_id][4:6])
            test_years = list(folds[data_fold_id][6:8])

        print(
            f"Using the following dataset split:\nTrain years: {train_years}, Val years: {val_years}, Test years: {test_years}")

        return train_years, val_years, test_years

