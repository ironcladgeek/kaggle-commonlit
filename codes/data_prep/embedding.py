from sentence_transformers import SentenceTransformer
from typing import Union
import numpy as np
import torch


class TextEmbedding:
    def __init__(self, model_name:str):
        # model names: https://www.sbert.net/docs/pretrained_models.html#sentence-embedding-models
        self.model = SentenceTransformer(model_name)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def get_embeddings(self, sentences:Union[list, np.ndarray]) -> np.ndarray:
        return self.model.encode(sentences, device=self.device)


