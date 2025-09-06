import torch
import os

# Devices
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Models
LLAMA_2_7B = "meta-llama/Llama-2-7b-chat-hf"

# Directories
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


# configs
MAX_NEW_TOKENS = 128
NUM_INST_TRAIN = 32