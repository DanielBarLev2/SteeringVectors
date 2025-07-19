import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# MODELS
LLAMA_2_7B = "meta-llama/Llama-2-7b-chat-hf"
GEMMA_2_2B = "google/gemma-2b"

# DATA
ALPACA_PATH = "tatsu-lab/alpaca"
MAL_PATH  = "walledai/MaliciousInstruct"
ADV_PATH = "walledai/AdvBench"
TDC_PATH = "walledai/TDC23-RedTeaming"

LAYER_IDX = 14
DATASET_SIZE = 128
BATCH_SIZE = 8 # for hooking