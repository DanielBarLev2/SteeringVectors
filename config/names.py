import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# MODELS
LLAMA_2_7B = "meta-llama/Llama-2-7b-chat-hf"
GAMMA_2_2B = "gemma/gemma-2-2b"

# DATA
ALPACA_PATH = "tatsu-lab/alpaca"
MAL_PATH  = "walledai/MaliciousInstruct"
ADV_PATH = "walledai/AdvBench"
TDC_PATH = "walledai/TDC23-RedTeaming"
