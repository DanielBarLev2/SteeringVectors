import os
import json
from glob import glob
from typing import List, Dict


class ContrastiveBehaviorDataset:
    """
    Builds contrastive refusal/compliance pairs from the Advanced‑AI‑Risk human evals.

    Each item is a dict with:
      • 'prompt': the base question (including the "(A)… / (B)…" choices)
      • 'refusal': prompt + the refusal option (the 'not matching' behavior)
      • 'compliance': prompt + the compliance option (the 'matching' behavior)
      • 'refusal_label': just the token string for refusal (e.g. " (A)" or " (B)")
      • 'compliance_label': just the token string for compliance
    """


    def __init__(self, name, path: str,
                 label_positive: str = 'answer_matching_behavior',
                 label_negative: str = 'answer_not_matching_behavior',):
        self.name = name
        self.examples: List[Dict[str, str]] = []

        if os.path.isdir(path):
            files = glob(os.path.join(path, '*.jsonl'))
        else:
            files = [path]

        for fn in files:
            with open(fn, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    q = data.get('question', '').strip()
                    pos = data.get(label_positive, '').strip()
                    neg = data.get(label_negative, '').strip()
                    if not q or not pos or not neg:
                        continue
                    self.examples.append({
                        'prompt': q,
                        'positive': q + pos,
                        'negative': q + neg,
                        'positive_label': pos,
                        'negative_label': neg,
                    })

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        return self.examples[idx]

    @classmethod
    def create_datasets(cls, base_dir: str) -> Dict[str, 'ContrastiveBehaviorDataset']:
        """
        Constructs all behavior datasets from known file mappings.

        Args:
            base_dir: path to 'advanced-ai-risk/human_generated_evals/'

        Returns:
            dict of behavior_name -> ContrastiveBehaviorDataset
        """
        behavior_config = {
            "power_seeking": "power-seeking-inclination.jsonl",
            "wealth_seeking": "wealth-seeking-inclination.jsonl",
            "self_preservation": "survival-instinct.jsonl",
            "coordination": "coordinate-other-ais.jsonl",
            "corrigibility": "corrigible-neutral-HHH.jsonl",
            "myopic_reward": "myopic-reward.jsonl",
            "self_awareness": "self-awareness-general-ai.jsonl",
        }

        datasets = {}
        for name, file in behavior_config.items():
            full_path = os.path.join(base_dir, file)
            if os.path.exists(full_path):
                datasets[name] = cls(
                    name=name,
                    path=full_path,
                    label_positive='answer_matching_behavior',
                    label_negative='answer_not_matching_behavior'
                )

        return datasets