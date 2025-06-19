import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.functional import scaled_dot_product_attention

class HugeMLP(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, activation="sigmoid"):
        super(HugeMLP, self).__init__()
        self.activation = activation
        self.nn = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, out_dim)
        )
        if activation == "sigmoid":
            self.activation = torch.nn.Sigmoid()
        elif activation == "tanh":
            self.activation = torch.nn.Tanh()
        elif activation == "relu":
            self.activation = torch.nn.ReLU()
        elif activation == "leaky_relu":
            self.activation = torch.nn.LeakyReLU()
        else:
            self.activation = None
    
    def forward(self, x):
        x = self.nn(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

class FFMLP(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, activation="sigmoid"):
        super(FFMLP, self).__init__()
        self.activation = activation
        self.nn = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, out_dim)
        )
        if activation == "sigmoid":
            self.activation = torch.nn.Sigmoid()
        elif activation == "tanh":
            self.activation = torch.nn.Tanh()
        elif activation == "relu":
            self.activation = torch.nn.ReLU()
        elif activation == "leaky_relu":
            self.activation = torch.nn.LeakyReLU()
        else:
            self.activation = None
    
    def forward(self, x):
        x = self.nn(x)
        if self.activation is not None:
            x = self.activation(x)
        return x
    
class MLPwithSkip(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, D, skips=[4], activation="sigmoid"):
        super(MLPwithSkip, self).__init__()
        self.D = D
        self.skips = skips
        self.networks = nn.ModuleList(
            [nn.Linear(in_dim, hidden_dim)] + \
            [nn.Linear(hidden_dim, hidden_dim) if i not in skips else nn.Linear(in_dim+hidden_dim, hidden_dim) for i in range(0, D-1)]
        )
        self.output_layer = nn.Linear(hidden_dim, out_dim)
        if activation == "sigmoid":
            self.activation = torch.nn.Sigmoid()
        elif activation == "tanh":
            self.activation = torch.nn.Tanh()
        elif activation == "relu":
            self.activation = torch.nn.ReLU()
        elif activation == "leaky_relu":
            self.activation = torch.nn.LeakyReLU()
        else:
            self.activation = None
    
    def forward(self, x):
        h = x
        for i,l in enumerate(self.networks):
            h = self.networks[i](h)
            h = F.leaky_relu(h)
            if i in self.skips:
                h = torch.cat((x, h), dim=-1)
        h = self.output_layer(h)
        if self.activation is not None:
            h = self.activation(h)
        return h
        
class TransformLayer(torch.nn.Module):
    def __init__(self):
        super(TransformLayer, self).__init__()
    
    def forward(self, x):
        # x is now in range [-1,1], convert it to [0,1]
        x = (x + 1) / 2
        return x


def _scaled_dot_product_attention(Q, K, V, mask=None):
    # Compute the dot products between Q and K, then scale by the square root of the key dimension
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

    # Apply mask if provided (useful for masked self-attention in transformers)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # Softmax to normalize scores, producing attention weights
    attention_weights = F.softmax(scores, dim=-1)
    
    # Compute the final output as weighted values
    output = torch.matmul(attention_weights, V)
    return output
    
    
class SelfAttention(nn.Module):
    def __init__(self, embed_size):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        # Define linear transformations for Q, K, V
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)

    def forward(self, x, mask=None):
        # Generate Q, K, V matrices
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        # Calculate attention using our scaled dot-product function
        out = _scaled_dot_product_attention(Q, K, V, mask)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert embed_size % num_heads == 0, "Embedding size must be divisible by number of heads"
        
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

        # Linear layers for Q, K, V for all heads
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)
        
        # Output linear layer
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, x, mask=None):
        N, seq_len, embed_size = x.shape
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Reshape Q, K, V to (N, num_heads, seq_len, head_dim)
        Q = Q.view(N, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(N, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(N, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Perform scaled dot-product attention and concatenate heads
        out, _ = scaled_dot_product_attention(Q, K, V, mask)
        out = out.transpose(1, 2).contiguous().view(N, seq_len, embed_size)

        # Final linear transformation
        return self.fc_out(out)
    
if __name__ == "__main__":
    # import time
    # attention = SelfAttention(embed_size=128).to('cuda:1')
    # start = time.time()
    # for i in range(50):
    #     with torch.no_grad():
    #         x = torch.randn(30000, 128).to('cuda:1')  # Example input
    #         y = attention(x)
    #         print(y[1])
    #         # swap x[0] and x[1]
    #         x = torch.cat((x[1:2], x[0:1], x[2:]), dim=0)
    #         y = attention(x)
    #         print(y[0])
    # end = time.time()
    # print(f"Time taken: {end - start:.4f} seconds")
    
    x = torch.randn(30000,128).to("cuda")
    mlp = HugeMLP(128,256,2,activation="sigmoid").to("cuda")
    y = mlp(x)
    print(y.shape)
    mlp = MLPwithSkip(128,256,2,D=8, skips=[4],activation='sigmoid').to("cuda")
    y = mlp(x)
    print(y.shape)