


class ModelTrainer():
    def __init__():
        pass




def _yield_CBOW_batch(text, n_window, batch_size=8, util_rate=0.5, spec_iter=None, vocab_size=tk_vocab_size, device=device, tokenizer=tokenizer):
    """
        Generator function that takes in text and returns a number of batches for each dataset based on the utilization rate specified. To be used in a CBOW model.

        Inputs:
            text:       (string) The text provided by the dataloader
            n_window:   (int) Size of our context window for the CBOW model. I.e., if n_window=4, then we will use the left 4 words and right 4 words to predict our target word
            batch_size: (int) Number of samples to return in each batch
            util_rate:  (float) Value from (0, 1] that specifies the % of possible batches that are generated before moving to next sample
            spec_iter:  (int) If set to none, we use util rate to determine num batches from this data. If set to an int, we use that number.
            vocab_size: (int) size of our tokenizer vocabulary
            device:     Pytorch device (e.g., cuda / cpu)
            tokenizer:  Our defined tokenizer (above). encode_as_ids(text) returns a 1-D python list of tokens

        Yields (generator function) batches of data in the form of GPU-mounted pytorch tensors until util_rate is tripped.
    """
    # Tokenize our data and determine number of batches we'll use
    tokens = torch.tensor(tokenizer.encode_as_ids(text), dtype=torch.int, device=device)
    len_tokens = len(tokens)
    num_possible_pairs = len_tokens - (2 * n_window)
    num_batches = int((num_possible_pairs * util_rate) // batch_size) if spec_iter==None else min(spec_iter, int(num_possible_pairs // batch_size))

    # Get random permutation of the possible 'center indices' --> these become our targets and surrounding 'n_window' size is our context
    center_indices = torch.arange(n_window, len_tokens - n_window, device=device).int()
    center_indices = center_indices[torch.randperm(center_indices.size(0))][:num_batches*batch_size]

    # Generate # of batches for this dataset
    for i in range(num_batches):
        batch_center_indices = center_indices[i*batch_size:(i+1)*batch_size]
        context_indices_list = []
        target_indices_list = []
        
        for center_idx in batch_center_indices:
            # Create a context window around the center word
            context_window = tokens[(center_idx - n_window):(center_idx + n_window + 1)]
            context_indices = torch.cat((context_window[:n_window], context_window[n_window+1:]))
            target_index = tokens[center_idx]
            
            context_indices_list.append(context_indices)
            target_indices_list.append(target_index)
        
        # Stack lists to create batch tensors
        context_tensor = torch.stack(context_indices_list).to(device)
        target_tensor = torch.stack(target_indices_list).to(device)
        
        yield context_tensor, target_tensor

for batch, (context, target) in enumerate(_yield_CBOW_batch(sample_traindata, n_window=4)):
    print(f"Batch {batch+1}")
    print(f"{'-'*60}")
    print(context)
    print(target)
    break


# Next - let's keep track of our loss data
def _estimate_loss(model, n_window, batching_fn, train_dl, test_dl, n_samples=10, iters_per_sample=10, device=device):
    """
        Function to estimate our loss (train and test) that we can call

        Inputs:
            model:             Pytorch sequential model
            n_window:          (int) Specify size of window to test over (for our CBOW / Skipgram models)
            batching_fn:       Function that returns our data as (context, targets)
            train_dl:          Pytorch dataloader for train data
            test_dl:           Pytorch dataloader for test data
            n_samples:         (int) Specify number of iterations to compute loss over
            iters_per_sample:  (int) Specify number of iterations per sample to compute loss over
            device:            Pytorch device (cuda / cpu)

        Returns the mean train loss and the test loss over n_iters samples
    """
    train_loss = torch.zeros(n_samples*iters_per_sample, device=device)
    test_loss = torch.zeros(n_samples*iters_per_sample, device=device)

    model.eval()
    with torch.no_grad():
        for i, sample in enumerate(train_dl):
            for (context, targets) in batching_fn(sample[0][0], n_window, spec_iter=iters_per_sample):
                loss = model(context, targets)
                train_loss[i-1] += loss
            if i == n_samples:
                break
        for i, sample in enumerate(test_dl):
            for (context, targets) in batching_fn(sample[0][0], n_window, spec_iter=iters_per_sample):
                loss = model(context, targets)
                test_loss[i-1] += loss
            if i == n_samples:
                break

    model.train()
    return train_loss.mean(), test_loss.mean()



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
