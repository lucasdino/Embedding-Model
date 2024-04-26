import os
import torch
import torch.nn as nn

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'embeddings', 'v16384_d512_4_26_24.pth')

def get_embeddings(weights_path=MODEL_PATH):
    if os.path.exists(weights_path):
        pretrained_weights = torch.load(weights_path)
        vocab_size, embed_dim = pretrained_weights.shape
        embedding_layer = nn.Embedding(vocab_size, embed_dim)
        embedding_layer.weight.data.copy_(pretrained_weights)
        print(f"Embeddings loaded successfully from {weights_path}.")
        return embedding_layer
    else:
        raise FileNotFoundError("No pretrained weights found at the specified path.")