import sys
import torch
import numpy as np

sys.path.insert(0, "../")

from utils.models.SSL.prototype import CrossAttentionModule

torch.manual_seed(1337)

batch_size = 2
img_features = torch.randn(batch_size, 2048, 1, 1) # the shape of what ResNet-50 backbone produces.
pc_features = torch.randn(batch_size, 2048) # the shape of what PointNet backbone produces.


x1 = np.array([1.0, 1.0, 0.0])
x2 = np.array([1.0, 0.0, 0.0])
x3 = np.array([0.0, 1.0, 0.0])
x4 = np.array([0.0, 0.0, 1.0])

vectors = torch.tensor(np.array([x1, x2, x3, x4]), dtype=torch.float32, requires_grad=False)

cross_attention = CrossAttentionModule(in_channels_query=3, in_channels_kv=3, out_channels=3, num_heads=3)

#out = cross_attention.forward(img_features.flatten(start_dim=1), pc_features.flatten(start_dim=1), pc_features.flatten(start_dim=1))
out = cross_attention.forward(vectors, vectors, vectors)

print(f"Shape of the output: {out.shape}")
print("The output: \n")
print(out)


multihead_attn = torch.nn.MultiheadAttention(embed_dim=3, num_heads=3, batch_first=False)
attn_output, attn_output_weights = multihead_attn(vectors, vectors, vectors)
print(f"Shape of the Pytorch Attention module output: {attn_output.shape}")
print("The output: \n")
print(attn_output)
