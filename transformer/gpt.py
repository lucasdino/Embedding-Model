import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from embedding_model.embeddings import get_embeddings


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, params):
        super().__init__()
        head_size = params['embed_dim'] // params['n_head']
        self.key = nn.Linear(params['embed_dim'], head_size, bias=False)
        self.query = nn.Linear(params['embed_dim'], head_size, bias=False)
        self.value = nn.Linear(params['embed_dim'], head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(params['context_size'], params['context_size'])))

        self.dropout = nn.Dropout(params['dropout'])


    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out



class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, params):
        super().__init__()
        head_size = params['embed_dim'] // params['n_head']
        self.heads = nn.ModuleList([Head(params) for _ in range(params['n_head'])])
        self.proj = nn.Linear(head_size * params['n_head'], params['embed_dim'])
        self.dropout = nn.Dropout(params['dropout'])


    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out



class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, params):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(params['embed_dim'], 4 * params['embed_dim']),
            nn.ReLU(),
            nn.Linear(4 * params['embed_dim'], params['embed_dim']),
            nn.Dropout(params['dropout']),
        )


    def forward(self, x):
        return self.net(x)



class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, params):
        super().__init__()
        self.sa = MultiHeadAttention(params)
        self.ffwd = FeedFoward(params)
        self.ln1 = nn.LayerNorm(params['embed_dim'])
        self.ln2 = nn.LayerNorm(params['embed_dim'])


    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x



class GPTLanguageModel(nn.Module):
    def __init__(self, params, device):
        super().__init__()
        self.params = params
        self.device = device
        self.token_embedding_table = get_embeddings()
        self.position_embedding_table = nn.Embedding(params['context_size'], params['embed_dim'])
        self.blocks = nn.Sequential(*[Block(params) for _ in range(params['n_layer'])])
        self.ln_f = nn.LayerNorm(params['embed_dim'])
        self.lm_head = nn.Linear(params['embed_dim'], params['vocab_size'])

        # Make token_embedding_table non-trainable and normalize it
        self.token_embedding_table.weight.requires_grad = False
        self.token_embedding_table.weight.data = self.token_embedding_table.weight.data / torch.linalg.norm(self.token_embedding_table.weight.data, dim=1, keepdim=True)

        self.apply(self._init_weights)


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding) and module is not self.token_embedding_table:
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)  # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device))  # (T, C)
        x = tok_emb + pos_emb  # (B, T, C)
        x = self.blocks(x)  # (B, T, C)
        x = self.ln_f(x)  # (B, T, C)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss


    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.params['context_size']:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
    

    def save_model(self, directory, filename):
        filepath = os.path.join(directory, filename)
        torch.save(self.state_dict(), filepath)
        print(f"Model successfully saved at: {filepath}")


    def load_model(self, filepath):
        self.load_state_dict(torch.load(filepath))
        print(f"Model weights successfully loaded from: {filepath}")