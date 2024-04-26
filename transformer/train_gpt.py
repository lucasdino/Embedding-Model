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



# Hyperparams and other necessary instantiation
# -----------------------------------------------------------------------------------------------
params = {
    'batch_size': 64,
    'block_size': 64,
    'max_iters': 1e7,
    'eval_interval': 500,
    'learning_rate': 3e-4,
    'eval_iters': 50,
    'n_embd': 512,
    'vocab_size': 16384,
    'n_head': 6,
    'n_layer': 6,
    'dropout': 0.2
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

torch.manual_seed(1337)



# Define helper functions
# -----------------------------------------------------------------------------------------------



# Train our model
# -----------------------------------------------------------------------------------------------
model = GPTLanguageModel(params, device)


def train_gpt(tokenizer):

    # Set up our train and text splits
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n = int(0.9*len(data)) # first 90% will be train, rest val
    train_data = data[:n]
    val_data = data[n:]

    train_model(vocab_size, tokenizer, train_data, val_data, device, save_weights = True)


# Import data and run our tokenizer
with open('data/tinyshakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()
print(f"Device: {device}")

# Ideally we import our tokenizer but I don't want to deal with that right now.
tokenizer = BPETokenizer()
tokenizer.train(text, vocab_size)

if mode == "train":
    train_gpt(tokenizer)
    chat(vocab_size, tokenizer, device, "weights/gpt_model_weights.pth", 2000)
else:
    chat(vocab_size, tokenizer, device, "weights/gpt_model_weights.pth", 2000)





## ------------------------------------------------------------
## Train and generation functions
## ------------------------------------------------------------

def train_model(vocab_size, tokenizer, train_data, val_data, device, save_weights = True):
    """ Function to train our model based on data / params specified above """
    
    model = GPTLanguageModel(vocab_size, device)
    m = model.to(device)
    print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')
     
    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Train model / print loss
    for iter in range(max_iters):

        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = _estimate_loss(model, train_data, val_data, device)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # sample a batch of data
        xb, yb = _get_batch('train', train_data, val_data, device)

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


    if save_weights:
        # Path where you want to save the weights
        weights_path = os.path.join('weights', 'gpt_model_weights.pth')
        # Save the model state dictionary
        torch.save(m.state_dict(), weights_path)
    else:
        # Simply just generate from the model
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        print(tokenizer.decode(m.generate(context, max_new_tokens=500)[0].tolist()))


def chat(vocab_size, tokenizer, device, weights_path, gen_length):
    """ Call this to load in model weights and generate a sequence of tokens """
    model = GPTLanguageModel(vocab_size, device)
    model.load_state_dict(torch.load(weights_path))
    m = model.to(device)
    
    # Print number of parameters
    print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

    # Generate text
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(tokenizer.decode(m.generate(context, max_new_tokens=gen_length)[0].tolist()))





# data loading
def _get_batch(split, train_data, val_data, device):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def _estimate_loss(model, train_data, val_data, device):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = _get_batch(split, train_data, val_data, device)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
