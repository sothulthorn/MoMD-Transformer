"""
Preprocessing script for MoMD Transformer datasets.

Converts raw dataset files into standardized .npy arrays:
  - vibration.npy  (num_samples, signal_length)
  - current.npy    (num_samples, signal_length)
  - labels.npy     (num_samples,)

Supported datasets:
  1. PU (Paderborn University) bearing dataset
     - Raw format: .mat files (MATLAB v5)
     - Source: https://mb.uni-paderborn.de/en/kat/research/bearing-datacenter
     - Reference: Lessmeier et al., PHM Society European Conference, 2016

  2. PMSM stator fault dataset
     - Raw format: .tdms files (NI LabVIEW)
     - Source: https://data.mendeley.com/datasets/rgn5brrgrn/5
     - Reference: Jung et al., Data Brief 47 (2023) 108952

Usage:
  python preprocess.py --dataset pu   --raw_dir ./data/pu_raw   --output_dir ./data/pu
  python preprocess.py --dataset pmsm --raw_dir ./data/pmsm_raw --output_dir ./data/pmsm
  python preprocess.py --dataset pu   --raw_dir ./data/pu_raw   --inspect
"""

import argparse
import os
import numpy as np


# ==============================================================================
# PU Bearing Dataset
# ==============================================================================
# Bearing codes and labels (Table 2 in paper)
PU_BEARING_LABELS = {
    "K001": 0, "K002": 0, "K003": 0,     # Normal
    "KA04": 1, "KA15": 1, "KA16": 1,     # Outer Ring (OR) Damage
    "KI04": 2, "KI14": 2, "KI16": 2,     # Inner Ring (IR) Damage
}
PU_LABEL_NAMES = {0: "Normal", 1: "OR Damage", 2: "IR Damage"}
PU_SAMPLES_PER_BEARING = 800  # Table 2
PU_SAMPLING_RATE = 64000      # 64 kHz


def inspect_pu_mat(filepath):
    """Inspect the structure of a PU dataset .mat file."""
    from scipy.io import loadmat

    print(f"\nInspecting: {filepath}")
    mat = loadmat(filepath, squeeze_me=True)

    # Filter out MATLAB metadata
    keys = [k for k in mat.keys() if not k.startswith("__")]
    print(f"Top-level keys: {keys}")

    for key in keys:
        data = mat[key]
        print(f"\n  Key '{key}': type={type(data).__name__}, "
              f"shape={getattr(data, 'shape', 'N/A')}, "
              f"dtype={getattr(data, 'dtype', 'N/A')}")

        # Navigate the nested struct to find channel names
        try:
            record = np.atleast_1d(data)[0]
            channels = record[2]
            print(f"  Found {len(channels)} measurement channels:")
            for ch in channels:
                name = str(ch[0])
                signal = ch[2]
                print(f"    '{name}': shape={signal.shape}, "
                      f"dtype={signal.dtype}, "
                      f"min={signal.min():.4f}, max={signal.max():.4f}")
        except (IndexError, TypeError) as e:
            print(f"  Could not parse nested structure: {e}")
            print(f"  Try manual inspection with: loadmat('{filepath}', squeeze_me=True)")


def extract_pu_signals(filepath):
    """
    Extract vibration and current signals from a PU .mat file.

    PU .mat structure (with squeeze_me=True):
      mat[filename_key] -> record -> record[2] -> array of channels
      Each channel: [0]=name, [2]=signal_data

    Channels:
      - 'vibration_1': vibration acceleration (64 kHz)
      - 'phase_current_1': motor phase current 1 (64 kHz)
      - 'phase_current_2': motor phase current 2 (64 kHz)

    Returns:
        vibration: 1D numpy array
        current: 1D numpy array
    """
    from scipy.io import loadmat

    mat = loadmat(filepath, squeeze_me=True)
    keys = [k for k in mat.keys() if not k.startswith("__")]
    key = keys[0]

    record = np.atleast_1d(mat[key])[0]
    channels = record[2]

    channel_dict = {}
    for ch in channels:
        name = str(ch[0])
        signal = np.asarray(ch[2], dtype=np.float64).flatten()
        channel_dict[name] = signal

    if "vibration_1" not in channel_dict:
        raise KeyError(
            f"'vibration_1' not found in {filepath}. "
            f"Available channels: {list(channel_dict.keys())}"
        )
    if "phase_current_1" not in channel_dict:
        raise KeyError(
            f"'phase_current_1' not found in {filepath}. "
            f"Available channels: {list(channel_dict.keys())}"
        )

    return channel_dict["vibration_1"], channel_dict["phase_current_1"]


def preprocess_pu(raw_dir, output_dir, signal_length=2048):
    """
    Preprocess PU bearing dataset.

    Expected raw directory structure:
      raw_dir/
        K001/
          N15_M07_F10_K001_1.mat
          N15_M07_F10_K001_2.mat
          ...
        K002/
          ...
        KA04/
          ...

    Each .mat file contains a 4-second recording at 64 kHz (256,000 samples).
    We segment each recording into non-overlapping windows of `signal_length`.
    """
    os.makedirs(output_dir, exist_ok=True)

    all_vib, all_cur, all_labels = [], [], []

    for bearing_code, label in PU_BEARING_LABELS.items():
        bearing_dir = os.path.join(raw_dir, bearing_code)
        if not os.path.isdir(bearing_dir):
            print(f"  [SKIP] {bearing_dir} not found")
            continue

        mat_files = sorted(
            f for f in os.listdir(bearing_dir) if f.endswith(".mat")
        )
        print(f"  {bearing_code} ({PU_LABEL_NAMES[label]}): "
              f"{len(mat_files)} .mat files")

        bearing_vib, bearing_cur = [], []
        for mf in mat_files:
            filepath = os.path.join(bearing_dir, mf)
            try:
                vib, cur = extract_pu_signals(filepath)
                bearing_vib.append(vib)
                bearing_cur.append(cur)
            except Exception as e:
                print(f"    [WARN] Skipping {mf}: {e}")
                continue

        if not bearing_vib:
            print(f"    [WARN] No valid files for {bearing_code}")
            continue

        # Concatenate all recordings for this bearing
        vib_concat = np.concatenate(bearing_vib)
        cur_concat = np.concatenate(bearing_cur)

        # Segment into fixed-length windows
        usable_len = min(len(vib_concat), len(cur_concat))
        num_windows = usable_len // signal_length
        num_windows = min(num_windows, PU_SAMPLES_PER_BEARING)

        for i in range(num_windows):
            start = i * signal_length
            all_vib.append(vib_concat[start:start + signal_length])
            all_cur.append(cur_concat[start:start + signal_length])
            all_labels.append(label)

        print(f"    -> {num_windows} samples extracted")

    _save_dataset(all_vib, all_cur, all_labels, output_dir, "PU")


# ==============================================================================
# PMSM Stator Fault Dataset
# ==============================================================================
# Stator codes and labels (Table 4 in paper)
PMSM_STATOR_FILES = {
    # label, expected_samples, vibration_filename, current_filename
    0: {
        "name": "Normal",
        "expected_samples": 1200,
        "entries": [
            ("1000W_0_00_vibration_interturn.tdms",
             "1000W_0_00_current_interturn.tdms"),
        ],
    },
    1: {
        "name": "Inter-turn Fault",
        "expected_samples": 1200,
        "entries": [
            ("1000W_6_48_vibration_interturn.tdms",
             "1000W_6_48_current_interturn.tdms"),
            ("1000W_21_69_vibration_interturn.tdms",
             "1000W_21_69_current_interturn.tdms"),
        ],
    },
    2: {
        "name": "Inter-coil Fault",
        "expected_samples": 1200,
        "entries": [
            ("1000W_2_00_vibration_intercoil.tdms",
             "1000W_2_00_current_intercoil.tdms"),
            ("1000W_7_56_vibration_intercoil.tdms",
             "1000W_7_56_current_intercoil.tdms"),
        ],
    },
}


def inspect_pmsm_tdms(filepath):
    """Inspect the structure of a PMSM dataset .tdms file."""
    from nptdms import TdmsFile

    print(f"\nInspecting: {filepath}")
    tdms = TdmsFile.read(filepath)

    for group in tdms.groups():
        print(f"  Group: '{group.name}'")
        for channel in group.channels():
            data = channel[:]
            print(f"    Channel: '{channel.name}', "
                  f"length={len(data)}, dtype={data.dtype}, "
                  f"min={data.min():.4f}, max={data.max():.4f}")


def extract_pmsm_vibration(filepath):
    """
    Extract vibration signal from a PMSM .tdms file.

    Group: 'Log', Channel: 'cDAQ1Mod1/ai3' (z-direction acceleration), 25.6 kHz.
    """
    from nptdms import TdmsFile

    tdms = TdmsFile.read(filepath)
    group = tdms["Log"]
    return np.asarray(group["cDAQ1Mod1/ai3"][:], dtype=np.float64)


def extract_pmsm_current(filepath):
    """
    Extract U-phase current from a PMSM .tdms file.

    Group: 'Log', Channel: 'cDAQ1Mod2/ai0' (U-phase current), 100 kHz.
    """
    from nptdms import TdmsFile

    tdms = TdmsFile.read(filepath)
    group = tdms["Log"]
    return np.asarray(group["cDAQ1Mod2/ai0"][:], dtype=np.float64)


def resample_signal(signal, orig_rate, target_rate):
    """Resample a signal from orig_rate to target_rate using scipy."""
    from scipy.signal import resample

    duration = len(signal) / orig_rate
    target_len = int(duration * target_rate)
    return resample(signal, target_len)


def preprocess_pmsm(raw_dir, output_dir, signal_length=2048):
    """
    Preprocess PMSM stator fault dataset.

    Expected raw directory structure:
      raw_dir/
        1000W_0_00_vibration_interturn.tdms
        1000W_0_00_current_interturn.tdms
        1000W_6_48_vibration_interturn.tdms
        1000W_6_48_current_interturn.tdms
        ...

    Vibration: 25.6 kHz, Current: 100 kHz.
    Each modality is segmented at its native sampling rate into windows
    of `signal_length`, then paired by index.
    """
    os.makedirs(output_dir, exist_ok=True)

    all_vib, all_cur, all_labels = [], [], []

    for label, info in PMSM_STATOR_FILES.items():
        class_name = info["name"]
        max_samples = info["expected_samples"]

        print(f"  Class {label} ({class_name}): "
              f"target {max_samples} samples")

        class_vib, class_cur = [], []

        for vib_fname, cur_fname in info["entries"]:
            vib_path = os.path.join(raw_dir, vib_fname)
            cur_path = os.path.join(raw_dir, cur_fname)

            if not os.path.exists(vib_path):
                print(f"    [SKIP] {vib_fname} not found")
                continue
            if not os.path.exists(cur_path):
                print(f"    [SKIP] {cur_fname} not found")
                continue

            print(f"    Loading {vib_fname}...")
            vib_signal = extract_pmsm_vibration(vib_path)

            print(f"    Loading {cur_fname}...")
            cur_signal = extract_pmsm_current(cur_path)

            class_vib.append(vib_signal)
            class_cur.append(cur_signal)

        if not class_vib:
            print(f"    [WARN] No valid files for class {label}")
            continue

        vib_concat = np.concatenate(class_vib)
        cur_concat = np.concatenate(class_cur)

        # Segment each modality independently at native rate, pair by index
        num_vib_windows = len(vib_concat) // signal_length
        num_cur_windows = len(cur_concat) // signal_length
        num_windows = min(num_vib_windows, num_cur_windows, max_samples)

        print(f"    Available windows: vib={num_vib_windows}, "
              f"cur={num_cur_windows}, using={num_windows}")

        for i in range(num_windows):
            start = i * signal_length
            all_vib.append(vib_concat[start:start + signal_length])
            all_cur.append(cur_concat[start:start + signal_length])
            all_labels.append(label)

        print(f"    -> {num_windows} samples extracted")

    _save_dataset(all_vib, all_cur, all_labels, output_dir, "PMSM")


# ==============================================================================
# Common utilities
# ==============================================================================

def _save_dataset(all_vib, all_cur, all_labels, output_dir, dataset_name):
    """Z-score normalize, save arrays, and print summary."""
    if not all_labels:
        print(f"\n[ERROR] No data extracted for {dataset_name}. "
              "Check your raw_dir path and file structure.")
        return

    all_vib = np.array(all_vib, dtype=np.float32)
    all_cur = np.array(all_cur, dtype=np.float32)
    all_labels = np.array(all_labels, dtype=np.int64)

    # Z-score normalization per modality
    all_vib = (all_vib - all_vib.mean()) / (all_vib.std() + 1e-8)
    all_cur = (all_cur - all_cur.mean()) / (all_cur.std() + 1e-8)

    np.save(os.path.join(output_dir, "vibration.npy"), all_vib)
    np.save(os.path.join(output_dir, "current.npy"), all_cur)
    np.save(os.path.join(output_dir, "labels.npy"), all_labels)

    print(f"\n{dataset_name} dataset saved to {output_dir}/")
    print(f"  vibration.npy: {all_vib.shape}")
    print(f"  current.npy:   {all_cur.shape}")
    print(f"  labels.npy:    {all_labels.shape}")
    for label in np.unique(all_labels):
        count = (all_labels == label).sum()
        print(f"  Class {label}: {count} samples")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess raw datasets for MoMD Transformer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Inspect a PU .mat file to verify structure
  python preprocess.py --dataset pu --raw_dir ./data/pu_raw --inspect

  # Preprocess PU bearing dataset
  python preprocess.py --dataset pu --raw_dir ./data/pu_raw --output_dir ./data/pu

  # Inspect a PMSM .tdms file
  python preprocess.py --dataset pmsm --raw_dir ./data/pmsm_raw --inspect

  # Preprocess PMSM stator dataset
  python preprocess.py --dataset pmsm --raw_dir ./data/pmsm_raw --output_dir ./data/pmsm

Expected raw data layout:
  PU dataset (download from Paderborn University Bearing DataCenter):
    raw_dir/
      K001/N15_M07_F10_K001_1.mat, ...
      K002/...
      K003/...
      KA04/...  KA15/...  KA16/...
      KI04/...  KI14/...  KI16/...

  PMSM dataset (download from Mendeley Data: doi.org/10.17632/rgn5brrgrn.5):
    raw_dir/
      1000W_0_00_vibration_interturn.tdms
      1000W_0_00_current_interturn.tdms
      1000W_6_48_vibration_interturn.tdms
      1000W_6_48_current_interturn.tdms
      1000W_21_69_vibration_interturn.tdms
      1000W_21_69_current_interturn.tdms
      1000W_2_00_vibration_intercoil.tdms
      1000W_2_00_current_intercoil.tdms
      1000W_7_56_vibration_intercoil.tdms
      1000W_7_56_current_intercoil.tdms
        """,
    )
    parser.add_argument(
        "--dataset", type=str, required=True, choices=["pu", "pmsm"],
        help="Which dataset to preprocess",
    )
    parser.add_argument(
        "--raw_dir", type=str, required=True,
        help="Path to raw dataset files",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory for .npy files (default: ./data/{dataset})",
    )
    parser.add_argument(
        "--signal_length", type=int, default=2048,
        help="Length of each signal window (default: 2048)",
    )
    parser.add_argument(
        "--inspect", action="store_true",
        help="Inspect raw file structure instead of preprocessing",
    )
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = f"./data/{args.dataset}"

    if args.inspect:
        # Inspect mode: show file structure of first found file
        if args.dataset == "pu":
            for bearing in PU_BEARING_LABELS:
                bdir = os.path.join(args.raw_dir, bearing)
                if os.path.isdir(bdir):
                    mats = [f for f in os.listdir(bdir) if f.endswith(".mat")]
                    if mats:
                        inspect_pu_mat(os.path.join(bdir, sorted(mats)[0]))
                        return
            print("No .mat files found. Check --raw_dir path.")
        else:
            tdms_files = [
                f for f in os.listdir(args.raw_dir) if f.endswith(".tdms")
            ]
            if tdms_files:
                # Inspect one vibration and one current file
                vib_files = [f for f in tdms_files if "vibration" in f]
                cur_files = [f for f in tdms_files if "current" in f]
                if vib_files:
                    inspect_pmsm_tdms(
                        os.path.join(args.raw_dir, sorted(vib_files)[0])
                    )
                if cur_files:
                    inspect_pmsm_tdms(
                        os.path.join(args.raw_dir, sorted(cur_files)[0])
                    )
                if not vib_files and not cur_files:
                    inspect_pmsm_tdms(
                        os.path.join(args.raw_dir, sorted(tdms_files)[0])
                    )
            else:
                print("No .tdms files found. Check --raw_dir path.")
        return

    # Preprocess mode
    print(f"Preprocessing {args.dataset.upper()} dataset")
    print(f"  Raw dir:    {args.raw_dir}")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Signal len: {args.signal_length}")
    print()

    if args.dataset == "pu":
        preprocess_pu(args.raw_dir, args.output_dir, args.signal_length)
    else:
        preprocess_pmsm(args.raw_dir, args.output_dir, args.signal_length)


if __name__ == "__main__":
    main()
