from dataset.ContrastiveBehaviorDataset import ContrastiveBehaviorDataset
from config.names import *
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import os

all_datasets = ContrastiveBehaviorDataset.create_datasets(ADVANCE_AI_RISK)

power_seeking = all_datasets.get("power_seeking")
wealth_seeking = all_datasets.get("wealth_seeking")
self_preservation = all_datasets.get("self_preservation")
coordination = all_datasets.get("coordination")
corrigibility = all_datasets.get("corrigibility")
myopic_reward = all_datasets.get("myopic_reward")
self_awareness = all_datasets.get("self_awareness")
