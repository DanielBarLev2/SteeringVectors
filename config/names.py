import torch
import os

# Device
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Models
LLAMA_2_7B = "meta-llama/Llama-2-7b-chat-hf"

# Directories
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

NORM_VEC_PATH = PROJECT_ROOT + '/normalized_vectors'
HALLUCINATION_NV_PATH = NORM_VEC_PATH + '/hallucination'
REFUSAL_NV_PATH = NORM_VEC_PATH + '/refusal'
SURVIVAL_NV_PATH = NORM_VEC_PATH + '/survival-instinct'
SYCOPHANCY_NV_PATH = NORM_VEC_PATH + '/sycophancy'

# Dataset
DATASET_DIR = PROJECT_ROOT + '/dataset'
ADVANCE_AI_RISK = DATASET_DIR + "/advanced-ai-risk" + "/human_generated_evals"

LAYER_IDX = 14
ALPHA = 1
BATCH_SIZE = 4
