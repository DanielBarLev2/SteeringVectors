import torch
import os

# Device
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Models
LLAMA_2_7B = "meta-llama/Llama-2-7b-chat-hf"

# Directories
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

# - steering vector types
NORM_VEC_PATH = PROJECT_ROOT + '/normalized_vectors'
HALLUCINATION_NV_PATH = NORM_VEC_PATH + '/hallucination'
REFUSAL_NV_PATH = NORM_VEC_PATH + '/refusal'
SURVIVAL_NV_PATH = NORM_VEC_PATH + '/survival-instinct'
SYCOPHANCY_NV_PATH = NORM_VEC_PATH + '/sycophancy'

# - Dataset
ADVANCE_AI_RISK = PROJECT_ROOT + '/dataset' + "/advanced-ai-risk" + "/human_generated_evals"
SURVIVAL_JS_PATH = ADVANCE_AI_RISK + '/survival-instinct.jsonl'


LAYER_IDX = 13
ALPHA = 1
BATCH_SIZE = 1
MAX_NEW_TOKENS = 64
