# trpca.py
import torch
from torch import nn
import torch.nn.functional as F
from typing import Dict
import math
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FFNExpert(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super(FFNExpert, self).__init__()
        self.input_projection = nn.Linear(d_model, hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, d_model)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.input_projection(x)
        x = self.activation(x)
        x = self.output_projection(x)
        return x

class Router(nn.Module):
    def __init__(self, d_model, num_experts):
        super(Router, self).__init__()
        self.layer = nn.Linear(d_model, num_experts)
    
    def forward(self, x):
        return F.softmax(self.layer(x), dim=-1)

class MoE(nn.Module):
    def __init__(self, d_model, num_experts, hidden_dim, top_k=2):
        super(MoE, self).__init__()
        self.experts = nn.ModuleList([FFNExpert(d_model, hidden_dim) for _ in range(num_experts)])
        self.router = Router(d_model, num_experts)
        self.top_k = top_k
    
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        # logging.info(f"MoE input shape: {x.shape}")
        routing_weights = self.router(x)  # Shape: [batch_size, seq_len, num_experts]
        topk_vals, topk_indices = torch.topk(routing_weights, self.top_k, dim=-1)  # Shapes: [batch_size, seq_len, top_k]
        topk_vals_normalized = topk_vals / topk_vals.sum(dim=-1, keepdim=True)  # Shape: [batch_size, seq_len, top_k]
        
        output = torch.zeros(batch_size, seq_len, d_model, device=x.device)  # Shape: [batch_size, seq_len, d_model]
        
        for i, expert in enumerate(self.experts):
            expert_mask = (topk_indices == i).float()  # Shape: [batch_size, seq_len, top_k]
            if expert_mask.any():
                inputs_to_expert = x.unsqueeze(2) * expert_mask.unsqueeze(-1)  # Shape: [batch_size, seq_len, top_k, d_model]
                inputs_to_expert = inputs_to_expert.view(-1, d_model)  # Flatten: [batch_size * seq_len * top_k, d_model]
                expert_output = expert(inputs_to_expert)  # Shape: [batch_size * seq_len * top_k, d_model]
                expert_output = expert_output.view(batch_size, seq_len, self.top_k, d_model)  # Reshape: [batch_size, seq_len, top_k, d_model]
                
                weights = topk_vals_normalized.unsqueeze(-1)  # Shape: [batch_size, seq_len, top_k, 1]
                weighted_output = expert_output * weights  # Shape: [batch_size, seq_len, top_k, d_model]
                
                output += weighted_output.sum(dim=2)  # Shape: [batch_size, seq_len, d_model]
        
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

class NormalizedTransformerBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_experts=4, top_k=2, dropout=0.1):
        super(NormalizedTransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=4, dropout=dropout, batch_first=True)
        self.moe = MoE(d_model=input_dim, num_experts=num_experts, hidden_dim=hidden_dim, top_k=top_k)
        self.alphaA = nn.Parameter(torch.tensor(1.0))
        self.alphaM = nn.Parameter(torch.tensor(1.0))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.normalize(x, p=2, dim=-1)

        hA, _ = self.attention(x, x, x)
        hA = F.normalize(hA, p=2, dim=-1)
        x = F.normalize(x + self.alphaA * (hA - x), p=2, dim=-1)

        x = x + self.alphaA * self.dropout(hA - x)
        hM = self.moe(x)

        hM = F.normalize(hM, p=2, dim=-1)
        x = F.normalize(x + self.alphaM * self.dropout(hM - x), p=2, dim=-1)


        return x

class NormalizedTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, projection_dim=4, num_experts=4, top_k=2, dropout=0.1):
        super(NormalizedTransformer, self).__init__()
        
        self.projection_dim = projection_dim
        
        self.pca_projection = nn.Linear(input_dim, hidden_dim)
        
        self.view_generator = nn.Sequential(
            nn.Linear(hidden_dim, projection_dim * hidden_dim),
            nn.LayerNorm(projection_dim * hidden_dim)
        )
        
        self.pos_encoder = PositionalEncoding(hidden_dim)
        
        self.transformer_blocks = nn.ModuleList(
            [NormalizedTransformerBlock(hidden_dim, hidden_dim * 2, num_experts=num_experts, top_k=top_k, dropout=dropout) 
             for _ in range(num_layers)]
        )
        self.regression_head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        batch_size = x.shape[0]
        
        x = self.pca_projection(x)
        x = F.normalize(x, p=2, dim=-1)

        x = self.view_generator(x)
        
        x = x.view(batch_size, self.projection_dim, -1)
        x = F.normalize(x, p=2, dim=-1)
        
        x = self.pos_encoder(x)
        for block in self.transformer_blocks:
            x = block(x)
        
        x = x.mean(dim=1)
        
        output = self.regression_head(x)
        outputs = {'regression_output': output}
        return outputs
        
# import torch
# from torch import nn
# import torch.nn.functional as F
# from typing import Dict
# import math

# class NormalizedTransformerBlock(nn.Module):
#     def __init__(self, input_dim, hidden_dim):
#         super(NormalizedTransformerBlock, self).__init__()
#         self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=4, dropout=0, batch_first=True)
#         self.mlp = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, input_dim),
#         )
#         self.alphaA = nn.Parameter(torch.tensor(1.0))  # Learnable scaling for attention updates
#         self.alphaM = nn.Parameter(torch.tensor(1.0))  # Learnable scaling for MLP updates

#     def forward(self, x):
#         # Normalize input
#         x = F.normalize(x, p=2, dim=-1)

#         # Attention block
#         hA, _ = self.attention(x, x, x)
#         hA = F.normalize(hA, p=2, dim=-1)
#         x = F.normalize(x + self.alphaA * (hA - x), p=2, dim=-1)

#         # MLP block
#         hM = self.mlp(x)
#         hM = F.normalize(hM, p=2, dim=-1)
#         x = F.normalize(x + self.alphaM * (hM - x), p=2, dim=-1)

#         return x

# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_len=5000):
#         super().__init__()
        
#         # Create positional encoding matrix
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
#         # Calculate sine and cosine positions
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0)
        
#         # Register as buffer (won't be updated during backprop)
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         # Add positional encoding to input
#         return x + self.pe[:, :x.size(1)]

# class NormalizedTransformer(nn.Module):
#     def __init__(self, input_dim, hidden_dim, num_layers, output_dim, projection_dim=4):
#         super(NormalizedTransformer, self).__init__()
#         self.projection_dim = projection_dim

#         # Project PCA vector to hidden_dim
#         self.pca_projection = nn.Linear(input_dim, hidden_dim)

#         # Generate different "views" of the projected PCA vector
#         self.view_generator = nn.Sequential(
#             nn.Linear(hidden_dim, projection_dim * hidden_dim),
#             nn.LayerNorm(projection_dim * hidden_dim)
#         )
        
#         # Add positional encoding
#         self.pos_encoder = PositionalEncoding(hidden_dim, projection_dim)

#         # Transformer blocks remain the same
#         self.transformer_blocks = nn.ModuleList(
#             [NormalizedTransformerBlock(hidden_dim, hidden_dim * 2) for _ in range(num_layers)]
#         )
#         self.regression_head = nn.Linear(hidden_dim, output_dim)

#     def forward(self, x):
#         # x shape: [batch, pca_dim]
#         batch_size = x.shape[0]

#         # Project PCA vector to hidden dimension
#         x = self.pca_projection(x)  # Shape: [batch, hidden_dim]
#         x = F.normalize(x, p=2, dim=-1)

#         # Generate multiple views of the projected vector
#         x = self.view_generator(x)  # Shape: [batch, projection_dim * hidden_dim]

#         # Reshape to [batch, projection_dim, hidden_dim]
#         x = x.view(batch_size, self.projection_dim, -1)
#         x = F.normalize(x, p=2, dim=-1)
        
#         # Add positional encoding
#         x = self.pos_encoder(x)

#         # Pass through transformer blocks
#         for block in self.transformer_blocks:
#             x = block(x) #replace this with MoE Transformer module @ Jessica

#         # Global average pooling over projection dimensions
#         x = x.mean(dim=1)  # Shape: [batch, hidden_dim]

#         # Regression head
#         output = self.regression_head(x)
#         outputs = {'regression_output': output}
#         return outputs

# class MTLNormalizedTransformer(nn.Module):
#     def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, 
#                  num_classes: int, output_dim: int = 1, projection_dim: int = 4):
#         super().__init__()
#         self.projection_dim = projection_dim

#         # Shared backbone
#         self.pca_projection = nn.Linear(input_dim, hidden_dim)
#         self.view_generator = nn.Sequential(
#             nn.Linear(hidden_dim, projection_dim * hidden_dim),
#             nn.LayerNorm(projection_dim * hidden_dim)
#         )
        
#         # Add positional encoding
#         self.pos_encoder = PositionalEncoding(hidden_dim, projection_dim)

#         # Transformer blocks
#         self.transformer_blocks = nn.ModuleList([
#             NormalizedTransformerBlock(hidden_dim, hidden_dim * 2)
#             for _ in range(num_layers)
#         ])

#         # Task-specific heads
#         self.regression_head = nn.Linear(hidden_dim, output_dim)
#         self.classification_head = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, num_classes)
#         )

#     def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
#         batch_size = x.shape[0]

#         # Shared processing
#         x = self.pca_projection(x)
#         x = F.normalize(x, p=2, dim=-1)
#         x = self.view_generator(x)
#         x = x.view(batch_size, self.projection_dim, -1)
#         x = F.normalize(x, p=2, dim=-1)
        
#         # Add positional encoding
#         x = self.pos_encoder(x)

#         # Transformer processing
#         for block in self.transformer_blocks:
#             x = block(x)

#         # Global pooling
#         x = x.mean(dim=1)

#         # Task-specific outputs
#         regression_output = self.regression_head(x)
#         classification_logits = self.classification_head(x)

#         return {
#             'regression_output': regression_output,
#             'classification_output': classification_logits
#         }

