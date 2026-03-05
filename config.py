"""
Configuration for MoMD Transformer.
Based on: "MoMD Transformer: adaptive multi-modal fault diagnosis via knowledge
           transfer with vibration-current signals"
           (Information Fusion, 2026)
"""

# ==============================================================================
# Model Hyperparameters (Table 1)
# ==============================================================================
SIGNAL_LENGTH = 2048
SEGMENT_LENGTH = 64
EMBED_DIM = 128
MLP_DIM = 512
NUM_HEADS = 8
BLOCK_DEPTH = 3
DROPOUT = 0.2

# ==============================================================================
# Training Hyperparameters
# ==============================================================================
OPTIMIZER = "Adam"
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 64
NUM_EPOCHS = 200
NUM_WORKERS = 4
NUM_REPEATS = 10
SEED = 42

# Loss weights (Eq. 21: L_all = L_D + lambda1 * L_gkt + lambda2 * L_msm)
LAMBDA_GKT = 1.0
LAMBDA_MSM = 1.0

# Masked Signal Modeling
MASK_RATIO = 0.15

# ==============================================================================
# PU Bearing Dataset Configuration (Table 2)
# ==============================================================================
PU_CONFIG = {
    "name": "PU",
    "num_classes": 3,
    "signal_length": 2048,
    "data_dir": "./data/pu",
    "split_ratio": (0.6, 0.2, 0.2),  # train:val:test = 3:1:1
    "label_names": {
        0: "Normal",
        1: "OR Damage",
        2: "IR Damage",
    },
    # Bearing codes per class (Table 2)
    "bearings": {
        "Normal": ["K001", "K002", "K003"],
        "OR Damage": ["KA04", "KA15", "KA16"],
        "IR Damage": ["KI04", "KI14", "KI16"],
    },
    "samples_per_bearing": 800,
}

# ==============================================================================
# PMSM Stator Dataset Configuration (Table 4)
# ==============================================================================
PMSM_CONFIG = {
    "name": "PMSM",
    "num_classes": 3,
    "signal_length": 2048,
    "data_dir": "./data/pmsm",
    "split_ratio": (0.6, 0.2, 0.2),  # train:val:test = 3:1:1
    "label_names": {
        0: "Normal",
        1: "Inter-turn Fault",
        2: "Inter-coil Fault",
    },
    "stators": {
        "Normal": [
            "1000W_0_00_vibration_interturn.tdms",
            "1000W_0_00_current_interturn.tdms"
        ],
        "Inter-turn Fault": [
            "1000W_6_48_vibration_interturn.tdms",
            "1000W_6_48_current_interturn.tdms",
            "1000W_21_69_vibration_interturn.tdms",
            "1000W_21_69_current_interturn.tdms",
        ],
        "Inter-coil Fault": [
            "1000W_2_00_vibration_intercoil.tdms",
            "1000W_2_00_current_intercoil.tdms",
            "1000W_7_56_vibration_intercoil.tdms",
            "1000W_7_56_current_intercoil.tdms",
        ],
    },
}

# ==============================================================================
# Output
# ==============================================================================
OUTPUT_DIR = "./results"
