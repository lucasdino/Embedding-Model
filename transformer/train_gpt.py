# Import Dependencies
# -----------------------------------------------------------------------------------------------
import sys
import os

project_root = os.path.dirname(os.getcwd())
sys.path.append(project_root)

# Pytorch
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim

# Import our various classes
from gpt import GPTLanguageModel
from dataloader.dataloader import MyDataLoader
from tokenizer.tokenizer import MyTokenizer
from trainer import ModelTrainer



# Hyperparams and other necessary instantiation
# -----------------------------------------------------------------------------------------------
model_params = {
    'batch_size': 64,
    'block_size': 64,
    'n_embd': 512,
    'vocab_size': 16384,
    'n_head': 6,
    'n_layer': 6,
    'dropout': 0.2
}
train_params = {
    'max_iters': 1e7,
    'eval_interval': 500,
    'learning_rate': 3e-4,
    'eval_iters': 50
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

torch.manual_seed(1337)



# Train our model
# -----------------------------------------------------------------------------------------------
model = GPTLanguageModel(model_params, device)
trainer = ModelTrainer()

