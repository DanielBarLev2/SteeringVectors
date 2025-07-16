from datasets import load_dataset
from config.names import ALPACA_PATH, MAL_PATH, ADV_PATH, TDC_PATH
import random


class ContrastivePromptDataset:
    def __init__(self, num_pairs, shuffle=False, seed=None):
        """

        :param num_pairs: Number of pairs to return.
        :param shuffle: Whether to shuffle the pairs before slicing.
        :param seed: For reproducibility when shuffling.
        """
        self.num_pairs = num_pairs
        self.shuffle = shuffle
        self.seed = seed

        alpaca = load_dataset(path=ALPACA_PATH, split="train")
        harmless = list(alpaca["instruction"])

        malicious = load_dataset(path=MAL_PATH, split="train")
        mal = list(malicious["prompt"])

        adv = load_dataset(path=ADV_PATH, split="train")
        adv_b = list(adv["prompt"])

        tdc = load_dataset(path=TDC_PATH, split="train")
        tdc_ = list(tdc["prompt"])

        # preserve order, drop exact duplicates
        combined_harmful = mal + adv_b + tdc_
        unique_harmful = list(dict.fromkeys(combined_harmful))

        if self.shuffle:
            if self.seed is not None:
                random.seed(self.seed)
            random.shuffle(harmless)
            random.shuffle(unique_harmful)

        if len(harmless) < num_pairs:
            raise ValueError(f"Only {len(harmless)} harmless prompts available, but num_pairs={num_pairs}")
        if len(unique_harmful) < num_pairs:
            raise ValueError(f"Only {len(unique_harmful)} harmful prompts available, but num_pairs={num_pairs}")

        self.harmless = harmless[:num_pairs]
        self.harmful = unique_harmful[:num_pairs]

    def __len__(self):
        return self.num_pairs

    def __getitem__(self, idx):
        return {
            "harmless": self.harmless[idx],
            "harmful": self.harmful[idx],
        }

    def get_pairs(self):
        """Return list of (harmless, harmful) tuples."""
        return list(zip(self.harmless, self.harmful))
