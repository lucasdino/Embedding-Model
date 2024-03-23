import torch
from gpt import train_model, chat
from tokenizer.bpetokenizer import BPETokenizer

mode = "train"


# Hyperparams
vocab_size = 512
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(1337)


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