import torch
import torch.nn as nn
from torch.nn import functional as F




# Bigram model
class Head(nn.Module):
    # single self attention head
    def __init__(self, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.head_size = head_size
        self.n_embd = n_embd
        self.block_size = block_size
        self.dropout = dropout
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size,block_size)))

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C**-0.5 # normalize by sqrt of head size -> controls variance
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v # v is like an encoding of x. 
        return out
    
class MultiHeadAttention(nn.Module):
    """ Creates multiple heads in parallel and concatenates them """

    def __init__(self, n_head, head_size, n_embd, dropout, block_size) -> None:
        super().__init__()
        self.n_head = n_head
        self.head_size = head_size
        self.n_embd = n_embd
        self.dropout = dropout
        self.block_size = block_size
        self.heads = nn.ModuleList([Head(head_size=self.head_size, n_embd=self.n_embd, block_size=self.block_size, dropout=self.dropout) for _ in range(n_head)])
        self.proj = nn.Linear(n_head * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """ MLP layer """

    def __init__(self, n_embd, dropout):
        super().__init__()
        self.dropout = dropout
        self.n_embd = n_embd
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    """ Transformer block """

    def __init__(self, n_embd, n_head, dropout, block_size):
        super().__init__()
        self.n_embd = n_embd
        self.n_head = n_head
        self.dropout = dropout
        self.block_size = block_size
        head_size = self.n_embd // self.n_head
        self.sa = MultiHeadAttention(n_head=self.n_head, head_size=head_size, n_embd=self.n_embd, dropout=self.dropout, block_size = self.block_size)
        self.ffwd = FeedForward(self.n_embd, self.dropout)
        self.ln1 = nn.LayerNorm(self.n_embd)
        self.ln2 = nn.LayerNorm(self.n_embd)
    
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class SimpleGPTModel(nn.Module):
    def __init__(self, vocab_size, n_embd, block_size, n_layer, n_head, dropout, device):
        super().__init__()
        self.device = device
        self.dropout = dropout
        self.n_layer = n_layer
        self.block_size = block_size
        self.n_embd = n_embd
        self.vocab_size = vocab_size
        self.n_head = n_head
        self.token_embedding_table = nn.Embedding(self.vocab_size, self.n_embd)
        self.position_embedding_table = nn.Embedding(self.block_size, self.n_embd)
        self.blocks = nn.Sequential(*[Block(self.n_embd, self.n_head, self.dropout, self.block_size) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # n_embd, n_head, dropout, block_size
        self.lm_head = nn.Linear(n_embd, vocab_size)
    
    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx) # BxTxC
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device))
        x = tok_emb + pos_emb
        # x = self.sa_heads(x)
        # x = self.ffwd(x)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) # B x T x vocab_size

        if targets == None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:,-1,:]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

