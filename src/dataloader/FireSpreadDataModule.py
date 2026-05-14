from pathlib import Path
import pickle

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, Subset, DataLoader
import glob
from .FireSpreadDataset import FireSpreadDataset
from typing import List, Optional, Union


class FireSpreadDataModule(LightningDataModule):
    WSTS_PLUS_TEST_YEARS = [2018, 2019, 2020, 2021]
    WSTS_PLUS_PLUS_TEST_YEARS = [2018, 2019, 2020, 2021]
    WSTS_STAR_TEST_YEAR_GROUPS = [
        [2012, 2013, 2014],
        [2015, 2016, 2017],
        [2018, 2019, 2020],
        [2021, 2022, 2023],
    ]
    WSTS_2018_2021_TEST_YEARS = [2018, 2019, 2020, 2021]

    def __init__(self, data_dir: str, batch_size: int, n_leading_observations: int, n_leading_observations_test_adjustment: int,
                 crop_side_length: int,
                 load_from_hdf5: bool, num_workers: int, remove_duplicate_features: bool, 
                 is_pad: Optional[bool] = False, pad_size: int = 224,
                 features_to_keep: Union[Optional[List[int]], str] = None, return_doy: bool = False,
                 stats_path: Optional[str] = None,
                 data_fold_id: int = 0, non_outlier_indices_path: Optional[str] = None, filter_ignition_train: Optional[bool] = False, filter_ignition_val_test: Optional[bool] = False,
                 ignition_only_train: Optional[bool] = False, ignition_only_val_test: Optional[bool] = False, additional_data: Optional[bool] = False,
                 train_mode: str = "default_fold", event_split_indices_path: Optional[str] = None,
                 loss_weight_mode: str = "none",
                 *args, **kwargs):
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
            features_to_keep (Union[Optional[List[int]], str], optional): _description_. List of processed feature indices from 0 to 42, indicating which features to keep. Defaults to None, which means using all features.
            return_doy (bool, optional): _description_. Return the day of the year per time step, as an additional feature. Defaults to False.
            data_fold_id (int, optional): _description_. Which data fold to use, i.e. splitting years into train/val/test set. Defaults to 0.
        """
        super().__init__()

        self.n_leading_observations_test_adjustment = n_leading_observations_test_adjustment
        self.data_fold_id = data_fold_id
        self.return_doy = return_doy
        self.stats_path = stats_path
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
        self.pad_size = pad_size
        self.non_outlier_indices_path = non_outlier_indices_path
        self.filter_ignition_train = filter_ignition_train
        self.filter_ignition_val_test = filter_ignition_val_test
        self.ignition_only_train = ignition_only_train
        self.ignition_only_val_test = ignition_only_val_test
        self.additional_data = additional_data
        self.train_mode = train_mode
        self.event_split_indices_path = event_split_indices_path
        self.loss_weight_mode = loss_weight_mode
        self._event_split_data = None
        self.determined_train_years = None

        valid_modes = {"default_fold", *self.get_event_fold_modes()}
        if self.train_mode not in valid_modes:
            raise ValueError(f"Invalid train_mode: {self.train_mode}. Expected one of {sorted(valid_modes)}.")
        if self.train_mode != "default_fold" and not self.event_split_indices_path:
            raise ValueError("event_split_indices_path is required for WSTS-based train modes.")
        if self.train_mode == "wsts_star" and self.loss_weight_mode != "none":
            raise ValueError(
                "wsts_star uses mixed-era WSTS_v2 data and requires loss_weight_mode='none' "
                "unless every selected HDF5 file has cloud bands."
            )

    @staticmethod
    def _get_mapping_value(mapping, key, default=None):
        if mapping is None:
            return default
        if key in mapping:
            return mapping[key]
        return mapping.get(str(key), default)

    @classmethod
    def get_wsts_plus_fold_definitions(cls):
        return [
            {"test_years": [2018], "train_years": [2016, 2017, 2019, 2020, 2021, 2022, 2023]},
            {"test_years": [2019], "train_years": [2016, 2017, 2018, 2020, 2021, 2022, 2023]},
            {"test_years": [2020], "train_years": [2016, 2017, 2018, 2019, 2021, 2022, 2023]},
            {"test_years": [2021], "train_years": [2016, 2017, 2018, 2019, 2020, 2022, 2023]},
        ]

    @classmethod
    def get_wsts_plus_plus_fold_definitions(cls):
        all_years = list(range(2012, 2024))
        return [
            {
                "test_years": [test_year],
                "train_years": [year for year in all_years if year != test_year],
            }
            for test_year in cls.WSTS_PLUS_PLUS_TEST_YEARS
        ]

    @classmethod
    def get_wsts_star_fold_definitions(cls):
        all_years = list(range(2012, 2024))
        return [
            {
                "test_years": list(test_years),
                "train_years": [year for year in all_years if year not in test_years],
            }
            for test_years in cls.WSTS_STAR_TEST_YEAR_GROUPS
        ]

    @classmethod
    def get_wsts_2018_2021_fold_definitions(cls):
        return [
            {"test_years": [2018], "train_years": [2019, 2020, 2021]},
            {"test_years": [2019], "train_years": [2018, 2020, 2021]},
            {"test_years": [2020], "train_years": [2018, 2019, 2021]},
            {"test_years": [2021], "train_years": [2018, 2019, 2020]},
        ]

    @classmethod
    def get_event_fold_definitions_by_mode(cls):
        return {
            "wsts_plus": cls.get_wsts_plus_fold_definitions(),
            "wsts_plus_plus": cls.get_wsts_plus_plus_fold_definitions(),
            "wsts_star": cls.get_wsts_star_fold_definitions(),
            "wsts_2018_2021": cls.get_wsts_2018_2021_fold_definitions(),
        }

    @classmethod
    def get_event_fold_modes(cls):
        return set(cls.get_event_fold_definitions_by_mode().keys())

    @classmethod
    def get_event_fold_definition(cls, train_mode: str, data_fold_id: int):
        if train_mode not in cls.get_event_fold_modes():
            raise ValueError(f"Unsupported event fold mode: {train_mode}")
        folds = cls.get_event_fold_definitions_by_mode()[train_mode]
        if not (0 <= int(data_fold_id) < len(folds)):
            raise ValueError(
                f"Invalid {train_mode} data_fold_id {data_fold_id}. Expected one of 0-{len(folds) - 1}."
            )
        fold = folds[int(data_fold_id)]
        test_years = cls.get_fold_test_years(fold)
        return {
            "data_fold_id": int(data_fold_id),
            "test_year": int(test_years[0]) if len(test_years) == 1 else None,
            "test_years": test_years,
            "train_years": list(fold["train_years"]),
        }

    @classmethod
    def get_event_fold_definitions(cls, train_mode: str):
        if train_mode not in cls.get_event_fold_modes():
            raise ValueError(f"Unsupported event fold mode: {train_mode}")
        return cls.get_event_fold_definitions_by_mode()[train_mode]

    @staticmethod
    def get_fold_test_years(fold):
        if "test_years" in fold:
            return [int(year) for year in fold["test_years"]]
        return [int(fold["test_year"])]

    def _load_event_split_indices(self):
        if self._event_split_data is not None:
            return
        event_split_path = Path(self.event_split_indices_path)
        if not event_split_path.exists():
            raise FileNotFoundError(f"Event split indices file not found: {event_split_path}")
        with event_split_path.open("rb") as f:
            self._event_split_data = pickle.load(f)
        print(f"Loaded event split indices from {event_split_path}")

    def _dataset_kwargs(self, years, is_train, n_leading_observations_test_adjustment):
        return {
            "data_dir": self.data_dir,
            "included_fire_years": years,
            "n_leading_observations": self.n_leading_observations,
            "n_leading_observations_test_adjustment": n_leading_observations_test_adjustment,
            "crop_side_length": self.crop_side_length,
            "load_from_hdf5": self.load_from_hdf5,
            "is_train": is_train,
            "remove_duplicate_features": self.remove_duplicate_features,
            "features_to_keep": self.features_to_keep,
            "return_doy": self.return_doy,
            "stats_years": self.determined_train_years,
            "is_pad": self.is_pad,
            "pad_size": self.pad_size,
            "stats_path": self.stats_path,
            "loss_weight_mode": self.loss_weight_mode,
        }
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
        
    def _apply_optional_filters(self):
        
        if self.non_outlier_indices_path is not None:
            non_outlier_indices = np.load(self.non_outlier_indices_path).tolist()
            print(f"Subsetting train_loader using {self.non_outlier_indices_path}")
            self.train_dataset = Subset(self.train_dataset, non_outlier_indices)

        if self.filter_ignition_train:
            self.train_dataset = self.filter_dataset(self.train_dataset)

        if self.ignition_only_train:
            self.train_dataset = self.keep_ignition(self.train_dataset)

        if self.filter_ignition_val_test:
            self.val_dataset = self.filter_dataset(self.val_dataset)
            self.test_dataset = self.filter_dataset(self.test_dataset)
            
        if self.ignition_only_val_test:
            self.val_dataset = self.keep_ignition(self.val_dataset)
            self.test_dataset = self.keep_ignition(self.test_dataset)

    def setup(self, stage):
        if self.train_mode == "default_fold":
            train_years, val_years, test_years = self.split_fires(
                self.data_fold_id, self.additional_data)
            self.determined_train_years = train_years
            self.train_dataset = FireSpreadDataset(
                **self._dataset_kwargs(train_years, is_train=True, n_leading_observations_test_adjustment=None)
            )
            self.val_dataset = FireSpreadDataset(
                **self._dataset_kwargs(val_years, is_train=True, n_leading_observations_test_adjustment=None)
            )
            self.test_dataset = FireSpreadDataset(
                **self._dataset_kwargs(test_years, is_train=False, n_leading_observations_test_adjustment=self.n_leading_observations_test_adjustment)
            )
        else:
            self._load_event_split_indices()
            split_mode = self._event_split_data.get("mode")
            if split_mode not in {None, self.train_mode}:
                raise ValueError(
                    f"{self.train_mode} expects a split file with mode {self.train_mode!r} or None, got {split_mode!r}."
                )

            split_n_leading_observations = self._event_split_data.get("n_leading_observations")
            if (
                split_n_leading_observations is not None
                and int(split_n_leading_observations) != int(self.n_leading_observations)
            ):
                raise ValueError(
                    "Split file was generated for a different n_leading_observations. "
                    f"Split file has n_leading_observations={split_n_leading_observations}, "
                    f"but datamodule requested {self.n_leading_observations}."
                )

            fold_definition = self.get_event_fold_definition(self.train_mode, self.data_fold_id)
            fold_payload = self._get_mapping_value(self._event_split_data["folds"], self.data_fold_id)
            if fold_payload is None:
                raise ValueError(
                    f"Split file does not contain fold {self.data_fold_id}. "
                    f"Available folds: {list(self._event_split_data['folds'].keys())}"
                )

            expected_test_years = fold_definition["test_years"]
            expected_train_years = fold_definition["train_years"]
            actual_test_years = self.get_fold_test_years(fold_payload)
            actual_train_years = [int(year) for year in fold_payload["train_years"]]
            if actual_test_years != expected_test_years or actual_train_years != expected_train_years:
                raise ValueError(
                    f"Split file fold does not match expected {self.train_mode} definition. "
                    f"Expected test_years={expected_test_years}, train_years={expected_train_years}; "
                    f"got test_years={actual_test_years}, train_years={actual_train_years}."
                )

            self.determined_train_years = expected_train_years
            fold_train_indices = fold_payload["train_indices_per_year"]
            fold_val_indices = fold_payload["val_indices_per_year"]
            train_subsets = []
            val_subsets = []
            train_counts_by_year = {}
            val_counts_by_year = {}

            for train_iter_year in expected_train_years:
                year_train_ds = FireSpreadDataset(
                    **self._dataset_kwargs([train_iter_year], is_train=True, n_leading_observations_test_adjustment=None)
                )
                year_val_ds = FireSpreadDataset(
                    **self._dataset_kwargs([train_iter_year], is_train=True, n_leading_observations_test_adjustment=None)
                )
                train_indices = self._get_mapping_value(fold_train_indices, train_iter_year, [])
                val_indices = self._get_mapping_value(fold_val_indices, train_iter_year, [])

                if train_indices:
                    train_subsets.append(Subset(year_train_ds, train_indices))
                    train_counts_by_year[int(train_iter_year)] = len(train_indices)
                else:
                    print(f"Warning: No train indices for year {train_iter_year} in fold {self.data_fold_id}.")

                if val_indices:
                    val_subsets.append(Subset(year_val_ds, val_indices))
                    val_counts_by_year[int(train_iter_year)] = len(val_indices)
                else:
                    print(f"Warning: No val indices for year {train_iter_year} in fold {self.data_fold_id}.")

            self.train_dataset = ConcatDataset(train_subsets) if train_subsets else Subset(
                FireSpreadDataset(**self._dataset_kwargs([], is_train=True, n_leading_observations_test_adjustment=None)), []
            )
            self.val_dataset = ConcatDataset(val_subsets) if val_subsets else Subset(
                FireSpreadDataset(**self._dataset_kwargs([], is_train=False, n_leading_observations_test_adjustment=None)), []
            )
            self.test_dataset = FireSpreadDataset(
                **self._dataset_kwargs(expected_test_years, is_train=False, n_leading_observations_test_adjustment=self.n_leading_observations_test_adjustment)
            )

            print(
                f"Mode: {self.train_mode}. Fold {self.data_fold_id}. "
                f"Train years: {expected_train_years}. Test years: {expected_test_years}. "
                f"Train samples by year: {train_counts_by_year}. "
                f"Val samples by year: {val_counts_by_year}. "
                f"Total sizes -> train: {len(self.train_dataset)}, val: {len(self.val_dataset)}, test: {len(self.test_dataset)}."
            )

        self._apply_optional_filters()

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
