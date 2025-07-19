import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# models
LLAMA_2_7B = "meta-llama/Llama-2-7b-chat-hf"
GEMMA_2_2B = "google/gemma-2b-it"

# data
ADVANCE_AI_RISK = "../dataset/advanced-ai-risk/human_generated_evals"

LAYER_IDX = 13
BATCH_SIZE = 4