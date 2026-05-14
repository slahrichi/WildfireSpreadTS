import argparse
import glob
import os
import pickle
from pathlib import Path

import h5py
import numpy as np


FOLD_DEFINITIONS = {
    "wsts_plus": [
        {"test_years": [2018], "train_years": [2016, 2017, 2019, 2020, 2021, 2022, 2023]},
        {"test_years": [2019], "train_years": [2016, 2017, 2018, 2020, 2021, 2022, 2023]},
        {"test_years": [2020], "train_years": [2016, 2017, 2018, 2019, 2021, 2022, 2023]},
        {"test_years": [2021], "train_years": [2016, 2017, 2018, 2019, 2020, 2022, 2023]},
    ],
    "wsts_plus_plus": [
        {"test_years": [2018], "train_years": [2012, 2013, 2014, 2015, 2016, 2017, 2019, 2020, 2021, 2022, 2023]},
        {"test_years": [2019], "train_years": [2012, 2013, 2014, 2015, 2016, 2017, 2018, 2020, 2021, 2022, 2023]},
        {"test_years": [2020], "train_years": [2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023]},
        {"test_years": [2021], "train_years": [2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2022, 2023]},
    ],
    "wsts_star": [
        {"test_years": [2012, 2013, 2014], "train_years": [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]},
        {"test_years": [2015, 2016, 2017], "train_years": [2012, 2013, 2014, 2018, 2019, 2020, 2021, 2022, 2023]},
        {"test_years": [2018, 2019, 2020], "train_years": [2012, 2013, 2014, 2015, 2016, 2017, 2021, 2022, 2023]},
        {"test_years": [2021, 2022, 2023], "train_years": [2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]},
    ],
    "wsts_2018_2021": [
        {"test_years": [2018], "train_years": [2019, 2020, 2021]},
        {"test_years": [2019], "train_years": [2018, 2020, 2021]},
        {"test_years": [2020], "train_years": [2018, 2019, 2021]},
        {"test_years": [2021], "train_years": [2018, 2019, 2020]},
    ],
}


def get_fold_test_years(fold_definition):
    if "test_years" in fold_definition:
        return [int(year) for year in fold_definition["test_years"]]
    return [int(fold_definition["test_year"])]


def datapoints_per_year(data_dir, year, n_leading_observations):
    year_paths = sorted(glob.glob(f"{data_dir}/{year}/*.hdf5"))
    if not year_paths:
        raise FileNotFoundError(f"No HDF5 files found for year {year} in {data_dir}")

    total_datapoints = 0
    per_fire = []
    for path in year_paths:
        with h5py.File(path, "r") as handle:
            n_images = len(handle["data"])
        datapoints = max(0, n_images - n_leading_observations)
        fire_name = Path(path).stem
        per_fire.append(
            {
                "fire_name": fire_name,
                "n_images": int(n_images),
                "n_datapoints": int(datapoints),
            }
        )
        total_datapoints += datapoints
    return int(total_datapoints), per_fire


def split_indices_for_year(num_datapoints, val_fraction, seed):
    if num_datapoints <= 0:
        return [], []

    val_count = int(round(num_datapoints * val_fraction))
    if val_fraction > 0:
        val_count = max(1, min(val_count, num_datapoints - 1))
    else:
        val_count = 0

    rng = np.random.default_rng(seed)
    permutation = rng.permutation(num_datapoints)
    val_indices = np.sort(permutation[:val_count]).tolist()
    train_indices = np.sort(permutation[val_count:]).tolist()
    return train_indices, val_indices


def main():
    parser = argparse.ArgumentParser(description="Generate event split indices for WSTS event-based folds.")
    parser.add_argument("--data_dir", required=True, help="Directory containing year/*.hdf5 files.")
    parser.add_argument("--output_path", required=True, help="Where to write the pickle file.")
    parser.add_argument(
        "--train_mode",
        required=True,
        choices=sorted(FOLD_DEFINITIONS.keys()),
        help="Event split mode definition to generate.",
    )
    parser.add_argument("--n_leading_observations", type=int, default=1, help="n_leading_observations used by the dataset.")
    parser.add_argument("--val_fraction", type=float, default=0.1, help="Fraction of each training year reserved for validation.")
    parser.add_argument("--val_seed", type=int, default=0, help="Base seed for validation subset selection.")
    args = parser.parse_args()

    if not (0.0 <= args.val_fraction < 1.0):
        raise ValueError(f"val_fraction must be in [0, 1), got {args.val_fraction}")
    if args.n_leading_observations < 1:
        raise ValueError("n_leading_observations must be >= 1")

    folds = {}
    year_inventory = {}
    required_years = sorted(
        {
            year
            for fold in FOLD_DEFINITIONS[args.train_mode]
            for year in get_fold_test_years(fold)
        }.union(
            {
                year
                for fold in FOLD_DEFINITIONS[args.train_mode]
                for year in fold["train_years"]
            }
        )
    )

    for year in required_years:
        total_datapoints, per_fire = datapoints_per_year(
            args.data_dir, year, args.n_leading_observations
        )
        year_inventory[int(year)] = {
            "num_datapoints": int(total_datapoints),
            "num_fires": len(per_fire),
            "per_fire": per_fire,
        }

    for fold_id, fold_definition in enumerate(FOLD_DEFINITIONS[args.train_mode]):
        train_indices_per_year = {}
        val_indices_per_year = {}
        for year in fold_definition["train_years"]:
            num_datapoints = year_inventory[int(year)]["num_datapoints"]
            train_indices, val_indices = split_indices_for_year(
                num_datapoints=num_datapoints,
                val_fraction=args.val_fraction,
                seed=args.val_seed + int(year),
            )
            train_indices_per_year[int(year)] = train_indices
            val_indices_per_year[int(year)] = val_indices

        folds[int(fold_id)] = {
            "test_years": get_fold_test_years(fold_definition),
            "train_years": [int(year) for year in fold_definition["train_years"]],
            "train_indices_per_year": train_indices_per_year,
            "val_indices_per_year": val_indices_per_year,
        }

    payload = {
        "mode": args.train_mode,
        "data_dir": args.data_dir,
        "n_leading_observations": int(args.n_leading_observations),
        "val_fraction": float(args.val_fraction),
        "val_seed": int(args.val_seed),
        "fold_test_years": [get_fold_test_years(fold) for fold in FOLD_DEFINITIONS[args.train_mode]],
        "years": required_years,
        "year_inventory": year_inventory,
        "folds": folds,
    }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump(payload, handle)

    print(f"Wrote split file to {output_path}")
    print(f"Mode: {args.train_mode}")
    print(f"n_leading_observations: {args.n_leading_observations}")
    print(f"Validation fraction: {args.val_fraction}")
    for fold_id, fold in folds.items():
        print(
            f"Fold {fold_id}: test_years={fold['test_years']}, "
            f"train_years={fold['train_years']}"
        )
        for year in fold["train_years"]:
            print(
                f"  year {year}: train={len(fold['train_indices_per_year'][year])}, "
                f"val={len(fold['val_indices_per_year'][year])}"
            )


if __name__ == "__main__":
    main()
