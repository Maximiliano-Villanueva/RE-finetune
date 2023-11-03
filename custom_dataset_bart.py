# Standard imports
import logging
from typing import Dict, Union
import pandas as pd
import torch
from torch.utils.data import Dataset

# Third-party imports
from transformers import BertTokenizer

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Define the CustomDataset class
class CustomDataset(Dataset):
    def __init__(self, filename: str, tokenizer: BertTokenizer, max_len: int) -> None:
        self.data = pd.read_csv(filename, encoding='utf-8', delimiter=';', quotechar='"')
        self.grouped_data = self.data.groupby('text')
        self.tokenizer = tokenizer
        self.max_len = max_len

        logger.info("CustomDataset initialized")

    def __len__(self) -> int:
        return len(self.grouped_data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text, group = list(self.grouped_data)[idx]
        inputs = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        triplets = []
        for _, row in group.iterrows():
            component1 = row['component1']
            relation = row['relation']
            component2 = row['component2']
            triplet = f"{component1} @@@ {relation} @@@ {component2}"
            triplets.append(triplet)
        target_text = " ||| ".join(triplets)
        
        target = self.tokenizer(
            target_text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'labels': target['input_ids'].flatten()
        }
