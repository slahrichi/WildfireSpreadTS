import argparse
import ast
import json
import os
from glob import glob
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import h5py
import numpy as np


LEGACY_NUM_BANDS = 23
CLOUD_NUM_BANDS = 29
ACTIVE_FIRE_CANONICAL_INDEX = 22
CLOUD_TIME_CANONICAL_INDEX = 23
LEGACY_CANONICAL_RAW_IDS = tuple(range(LEGACY_NUM_BANDS))
CLOUD_CANONICAL_RAW_IDS = tuple(range(CLOUD_NUM_BANDS))
WSTS_2012_2015_CANONICAL_RAW_IDS = tuple(list(range(17)) + [22])

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


def normalize_combo(years: Sequence[int]) -> Tuple[int, ...]:
    return tuple(sorted(set(int(year) for year in years)))


def parse_year_combos_literal(raw_value: str) -> List[Tuple[int, ...]]:
    try:
        parsed = ast.literal_eval(raw_value)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(
            "--year-combos must be a valid Python/JSON-style list of year lists."
        ) from exc

    if not isinstance(parsed, (list, tuple)):
        raise ValueError("--year-combos must evaluate to a list or tuple of year lists.")

    combos: List[Tuple[int, ...]] = []
    for combo in parsed:
        if not isinstance(combo, (list, tuple)):
            raise ValueError("Each item in --year-combos must be a list or tuple of years.")
        combos.append(normalize_combo(combo))

    return combos


def discover_years(data_dir: str) -> List[int]:
    years = []
    for entry in sorted(os.listdir(data_dir)):
        path = os.path.join(data_dir, entry)
        if os.path.isdir(path) and entry.isdigit():
            years.append(int(entry))
    return years


def build_year_combos(args: argparse.Namespace) -> List[Tuple[int, ...]]:
    combos: List[Tuple[int, ...]] = []

    if args.train_mode:
        for fold in FOLD_DEFINITIONS[args.train_mode]:
            combos.append(normalize_combo(fold["train_years"]))

    if args.years:
        combos.extend(normalize_combo(years) for years in args.years)
    if args.year_combos:
        combos.extend(parse_year_combos_literal(args.year_combos))

    if args.exclude_years:
        all_years = normalize_combo(args.all_years or discover_years(args.data_dir))
        if not all_years:
            raise ValueError("No year folders were found to build exclusion-based combos.")
        for excluded in args.exclude_years:
            excluded_set = set(int(year) for year in excluded)
            combo = tuple(year for year in all_years if year not in excluded_set)
            if not combo:
                raise ValueError(
                    f"Excluding {sorted(excluded_set)} leaves an empty year combination."
                )
            combos.append(combo)

    if not combos:
        raise ValueError(
            "Provide --train_mode, --years, --year-combos, or --exclude-years."
        )

    return sorted(set(combos))


def get_hdf5_files_for_years(
    data_dir: str, years: Sequence[int], limit_files_per_year: int = None
) -> List[str]:
    files: List[str] = []
    for year in years:
        year_dir = os.path.join(data_dir, str(year))
        year_files = sorted(glob(os.path.join(year_dir, "*.hdf5")))
        if not year_files:
            raise FileNotFoundError(f"No HDF5 files found for year {year} in {data_dir}")
        if limit_files_per_year is not None:
            year_files = year_files[:limit_files_per_year]
        files.extend(year_files)
    return files


def get_canonical_raw_feature_ids(num_bands: int) -> Tuple[int, ...]:
    if num_bands == len(LEGACY_CANONICAL_RAW_IDS):
        return LEGACY_CANONICAL_RAW_IDS
    if num_bands == len(CLOUD_CANONICAL_RAW_IDS):
        return CLOUD_CANONICAL_RAW_IDS
    if num_bands == len(WSTS_2012_2015_CANONICAL_RAW_IDS):
        return WSTS_2012_2015_CANONICAL_RAW_IDS
    raise ValueError(
        f"Unsupported number of raw bands: {num_bands}. Expected "
        f"{len(LEGACY_CANONICAL_RAW_IDS)}, {len(CLOUD_CANONICAL_RAW_IDS)}, "
        f"or {len(WSTS_2012_2015_CANONICAL_RAW_IDS)}."
    )


def process_single_file(file_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        with h5py.File(file_path, "r") as h5_file:
            data = h5_file["data"][()]
    except Exception as exc:
        raise RuntimeError(f"Failed to read {file_path}: {exc}") from exc

    if data.ndim != 4:
        raise ValueError(f"{file_path} has unexpected data shape {data.shape}; expected (T, C, H, W).")

    _, num_bands, _, _ = data.shape
    canonical_raw_feature_ids = get_canonical_raw_feature_ids(num_bands)

    valid_mask = ~np.isnan(data)
    # Match project event-time semantics: zeros in active fire and cloud_time
    # mean "no event recorded", not a measured midnight overpass.
    for raw_idx, canonical_idx in enumerate(canonical_raw_feature_ids):
        if canonical_idx in {ACTIVE_FIRE_CANONICAL_INDEX, CLOUD_TIME_CANONICAL_INDEX}:
            valid_mask[:, raw_idx, :, :] &= data[:, raw_idx, :, :] != 0

    safe_data = np.where(valid_mask, data, 0.0)
    canonical_num_bands = max(canonical_raw_feature_ids) + 1
    band_sum = np.zeros(canonical_num_bands, dtype=np.float64)
    band_sum_sq = np.zeros(canonical_num_bands, dtype=np.float64)
    valid_count = np.zeros(canonical_num_bands, dtype=np.int64)
    total_count = np.zeros(canonical_num_bands, dtype=np.int64)

    raw_band_sum = np.sum(safe_data, axis=(0, 2, 3), dtype=np.float64)
    raw_band_sum_sq = np.sum(safe_data * safe_data, axis=(0, 2, 3), dtype=np.float64)
    raw_valid_count = np.sum(valid_mask, axis=(0, 2, 3), dtype=np.int64)
    raw_total_count = np.full(num_bands, data.shape[0] * data.shape[2] * data.shape[3], dtype=np.int64)

    for raw_idx, canonical_idx in enumerate(canonical_raw_feature_ids):
        band_sum[canonical_idx] += raw_band_sum[raw_idx]
        band_sum_sq[canonical_idx] += raw_band_sum_sq[raw_idx]
        valid_count[canonical_idx] += raw_valid_count[raw_idx]
        total_count[canonical_idx] += raw_total_count[raw_idx]

    return band_sum, band_sum_sq, valid_count, total_count


def compute_stats_for_files(hdf5_files: Sequence[str], num_workers: int) -> Dict[str, np.ndarray]:
    if not hdf5_files:
        raise ValueError("No HDF5 files found for the requested year combination.")

    if num_workers <= 1:
        results = [process_single_file(path) for path in hdf5_files]
    else:
        with Pool(processes=num_workers) as pool:
            results = pool.map(process_single_file, hdf5_files)

    canonical_num_bands = max(result[0].shape[0] for result in results)
    total_sum = np.zeros(canonical_num_bands, dtype=np.float64)
    total_sum_sq = np.zeros(canonical_num_bands, dtype=np.float64)
    valid_count = np.zeros(canonical_num_bands, dtype=np.int64)
    total_count = np.zeros(canonical_num_bands, dtype=np.int64)

    for band_sum, band_sum_sq, band_valid_count, band_total_count in results:
        if band_sum.shape[0] < canonical_num_bands:
            pad_width = canonical_num_bands - band_sum.shape[0]
            band_sum = np.pad(band_sum, (0, pad_width))
            band_sum_sq = np.pad(band_sum_sq, (0, pad_width))
            band_valid_count = np.pad(band_valid_count, (0, pad_width))
            band_total_count = np.pad(band_total_count, (0, pad_width))
        total_sum += band_sum
        total_sum_sq += band_sum_sq
        valid_count += band_valid_count
        total_count += band_total_count

    means = np.zeros(canonical_num_bands, dtype=np.float64)
    stds = np.ones(canonical_num_bands, dtype=np.float64)
    missing_values = np.ones(canonical_num_bands, dtype=np.float64)

    present_mask = valid_count > 0
    means[present_mask] = total_sum[present_mask] / valid_count[present_mask]
    variances = np.zeros(canonical_num_bands, dtype=np.float64)
    variances[present_mask] = np.maximum(
        (total_sum_sq[present_mask] / valid_count[present_mask]) - np.square(means[present_mask]),
        0.0,
    )
    stds = np.sqrt(variances)
    stds[~present_mask] = 1.0
    total_present_mask = total_count > 0
    missing_values[total_present_mask] = 1.0 - (
        valid_count[total_present_mask] / total_count[total_present_mask].astype(np.float64)
    )

    return {
        "means": means.astype(np.float32),
        "stds": stds.astype(np.float32),
        "missing_values": missing_values.astype(np.float32),
    }


def load_existing_stats(path: str, overwrite: bool) -> Dict[Tuple[int, ...], Dict[str, np.ndarray]]:
    if overwrite or not os.path.exists(path):
        return {}

    if path.endswith(".npy"):
        loaded = np.load(path, allow_pickle=True).item()
        return {normalize_combo(key): value for key, value in loaded.items()}

    with open(path, "r", encoding="utf-8") as json_file:
        loaded = json.load(json_file)
    normalized = {}
    for key, stats in loaded.items():
        combo = normalize_combo(int(year) for year in key.split(","))
        normalized[combo] = {
            metric_name: np.array(metric_value, dtype=np.float32)
            for metric_name, metric_value in stats.items()
        }
    return normalized


def stats_to_jsonable(
    stats_dict: Dict[Tuple[int, ...], Dict[str, np.ndarray]]
) -> Dict[str, Dict[str, List[float]]]:
    json_ready: Dict[str, Dict[str, List[float]]] = {}
    for years, stats in stats_dict.items():
        key = ",".join(str(year) for year in years)
        json_ready[key] = {
            metric_name: metric_value.tolist()
            for metric_name, metric_value in stats.items()
        }
    return json_ready


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute per-band means, stds, and missing-value rates for one or more "
            "combinations of wildfire HDF5 years."
        )
    )
    parser.add_argument(
        "--data_dir",
        required=True,
        help="Directory containing year subdirectories of .hdf5 files.",
    )
    parser.add_argument(
        "--train_mode",
        choices=sorted(FOLD_DEFINITIONS.keys()),
        default=None,
        help="Generate stats for all training-year combinations implied by this mode.",
    )
    parser.add_argument(
        "--years",
        nargs="+",
        action="append",
        type=int,
        help=(
            "One year combination to process. Repeat this flag for multiple combos, "
            "for example: --years 2018 2019 2020 --years 2019 2020 2021"
        ),
    )
    parser.add_argument(
        "--year-combos",
        type=str,
        default=None,
        help=(
            "Multiple year combinations as a Python/JSON-style list of lists, "
            "for example: '[[2018, 2019, 2020], [2019, 2020, 2021]]'"
        ),
    )
    parser.add_argument(
        "--exclude-years",
        nargs="+",
        action="append",
        type=int,
        help=(
            "Define a combo by excluding years from the discovered year list. Repeat "
            "for multiple combos."
        ),
    )
    parser.add_argument(
        "--all-years",
        nargs="+",
        type=int,
        help=(
            "Optional explicit year universe used with --exclude-years. Defaults to "
            "discovering all numeric year folders in --data_dir."
        ),
    )
    parser.add_argument(
        "--output_path",
        required=True,
        help="Path to the JSON stats file to create or update.",
    )
    parser.add_argument(
        "--output_npy",
        default=None,
        help="Optional .npy export path with the same stats content.",
    )
    parser.add_argument(
        "--overwrite_output",
        action="store_true",
        help="Ignore any existing output file and write only newly computed combos.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of worker processes for file-level parallelism.",
    )
    parser.add_argument(
        "--limit_files_per_year",
        type=int,
        default=None,
        help="Optional cap for quick smoke tests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    year_combos = build_year_combos(args)

    print(f"Will compute {len(year_combos)} year combination(s): {year_combos}")
    combined_stats = load_existing_stats(args.output_path, args.overwrite_output)

    for combo in year_combos:
        files = get_hdf5_files_for_years(
            args.data_dir,
            combo,
            limit_files_per_year=args.limit_files_per_year,
        )
        print(f"Years {combo}: found {len(files)} HDF5 files")
        combined_stats[combo] = compute_stats_for_files(files, num_workers=args.num_workers)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as json_file:
        json.dump(stats_to_jsonable(combined_stats), json_file)
    print(f"Saved JSON stats export to {output_path}")

    if args.output_npy:
        np.save(args.output_npy, combined_stats, allow_pickle=True)
        print(f"Saved numpy stats dictionary to {args.output_npy}")


if __name__ == "__main__":
    main()
