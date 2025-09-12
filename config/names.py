import torch
import os

# Devices
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Models
LLAMA_2_7B = "meta-llama/Llama-2-7b-chat-hf"

# Directories
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SV_DIR = PROJECT_ROOT + "/Steering-Vectors"
SV_PATH = SV_DIR + "/refusal_vector.pt"
