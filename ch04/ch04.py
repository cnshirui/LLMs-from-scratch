
# Chapter 4: Implementing a GPT model from Scratch To Generate Text
# Chapter 4: 从0构建GPT模型来生成文本

## 4.1 Coding an LLM architecture
## 4.1 构建大模型架构

import tiktoken
import torch
import torch.nn as nn

GPT_CONFIG_124M = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 1024, # Context length
    "emb_dim": 768,         # Embedding dimension
    "n_heads": 12,          # Number of attention heads
    "n_layers": 12,         # Number of layers
    "drop_rate": 0.1,       # Dropout rate
    "qkv_bias": False       # Query-Key-Value bias
}

'''
标记嵌入，vocab_size * emb_dim = 50257 * 756 = 38.6M
自注意力权重，QKV，768 * (768 * 3) = 1.77M
多头注意力，1.77M * 12 = 21.22M
输出预测，768 * 768 = 0.59M
前馈网络，4维隐藏层，768 * （4 * 768）+ （4 * 768） * 768 = 4.72M
每层，1.77M + 0.59M + 4.74M = 7.08M
多层，7.08M * 12 = 85M
最终层范化和预估
总共 38.6M + 85M = 124M

total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params:,}")
Total number of parameters: 163,009,536

# 在GPT-2的论文里，研究者复用了嵌入矩阵作为输出矩阵

'''
class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # Use a placeholder for TransformerBlock
        self.trf_blocks = nn.Sequential(
            *[DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        # Use a placeholder for LayerNorm
        self.final_norm = DummyLayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        a = self.drop_emb(x)
        b = self.trf_blocks(a)
        c = self.final_norm(b)
        logits = self.out_head(c)
        return logits


class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # A simple placeholder

    def forward(self, x):
        # This block does nothing and just returns its input.
        return x


class DummyLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        # The parameters here are just to mimic the LayerNorm interface.

    def forward(self, x):
        # This layer does nothing and just returns its input.
        return x

txt1 = "Every effort moves you"
txt2 = "Every day holds a"
batch = []
tokenizer = tiktoken.get_encoding("gpt2")
batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0)
print(batch)

torch.manual_seed(123)
model = DummyGPTModel(GPT_CONFIG_124M)
logits = model(batch)
print("Output shape:", logits.shape)
print(logits)
# 对数几率，未归一化得分

## 4.2 Normalizing activations with layer normalization
## 4.2 激活层归一化

# create 2 training examples with 5 dimensions (features) each
torch.manual_seed(123)
batch_example = torch.randn(2, 5)
print('batch: ', batch_example)
# tensor([[-0.1115,  0.1204, -0.3696, -0.2404, -1.1969],
#         [ 0.2093, -0.9724, -0.7550,  0.3239, -0.1085]])

# nn.Linear(5, 6) 在每次实例化时都会重新初始化权重和偏置
# linear = nn.Linear(5, 6)
# print('linear: ', linear(batch_example))
# print('batch: ', batch_example)
# tensor([[ 0.2260,  0.3470, -0.4727,  0.2216, -0.1220, -0.8747],
#         [ 0.2133,  0.2394, -0.1502,  0.5198,  0.3297, -0.2985]],
#        grad_fn=<AddmmBackward0>)

layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
out = layer(batch_example)
print('seq: ', out)
# tensor([[0.2260, 0.3470, 0.0000, 0.2216, 0.0000, 0.0000],
#         [0.2133, 0.2394, 0.0000, 0.5198, 0.3297, 0.0000]],
#        grad_fn=<ReluBackward0>)

'''
torch.nn.ReLU是一个常用的激活函数，用于非线性化神经网络中的层。
ReLU（Rectified Linear Unit）函数的基本形式是f(x) = max(0, x)，
即输入值大于0的部分保持不变，小于或等于0的部分变为0。
这种非线性特性使得神经网络能够更好地学习和理解复杂的数据模式。
'''

mean = out.mean(dim=-1, keepdim=True)
print("Mean:\n", mean)
# tensor([[0.1324],
#         [0.2170]], grad_fn= < MeanBackward1 >)

var = out.var(dim=-1, keepdim=True)
print("Variance:\n", var)
# tensor([[0.0231],
#         [0.0398]], grad_fn= < VarBackward0 >)

out_norm = (out - mean) / torch.sqrt(var)
print("Normalized layer outputs:\n", out_norm)
# tensor([[0.6159, 1.4126, -0.8719, 0.5872, -0.8719, -0.8719],
#         [-0.0189, 0.1121, -1.0876, 1.5173, 0.5647, -1.0876]],
#        grad_fn= < DivBackward0 >)

mean = out_norm.mean(dim=-1, keepdim=True)
var = out_norm.var(dim=-1, keepdim=True)
print("Mean:\n", mean)
print("Variance:\n", var)
 # tensor([[-5.9605e-08],
 #        [ 1.9868e-08]], grad_fn=<MeanBackward1>)
 # tensor([[1.0000],
 #        [1.0000]], grad_fn=<VarBackward0>)

torch.set_printoptions(sci_mode=False)
print("Mean:\n", mean)
print("Variance:\n", var)
# tensor([[    -0.0000],
#     [     0.0000]], grad_fn=<MeanBackward1>)
# tensor([[1.0000],
#     [1.0000]], grad_fn=<VarBackward0>)

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

ln = LayerNorm(emb_dim=5)
out_ln = ln(batch_example)
print('out_ln: ', out_ln)
# tensor([[ 0.5528,  1.0693, -0.0223,  0.2656, -1.8654],
#         [ 0.9087, -1.3767, -0.9564,  1.1304,  0.2940]], grad_fn=<AddBackward0>)

mean = out_ln.mean(dim=-1, keepdim=True)
var = out_ln.var(dim=-1, unbiased=False, keepdim=True)
print("Mean:\n", mean)
print("Variance:\n", var)

## 4.3 Implementing a feed forward network with GELU activations
## 4.3 使用GELU激活实现前馈神经网络

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))

import matplotlib.pyplot as plt

x = torch.linspace(-3, 3, 100)
gelu, relu = GELU(), nn.ReLU()
y_gelu, y_relu = gelu(x), relu(x)

plt.figure(figsize=(8, 3))
for i, (y, label) in enumerate(zip([y_gelu, y_relu], ["GELU", "ReLU"]), 1):
    plt.subplot(1, 2, i)
    plt.plot(x, y)
    plt.title(f"{label} activation function")
    plt.xlabel("x")
    plt.ylabel(f"{label}(x)")
    plt.grid(True)

plt.tight_layout()
# plt.show()

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


# 第一层将映射到四倍大的空间
# 第二层再把空间压回去
# input shape: [batch_size, num_token, emb_size]
print(GPT_CONFIG_124M["emb_dim"])
x = torch.rand(2, 3, 768)
ffn = FeedForward(GPT_CONFIG_124M)
out = ffn(x)
print(out.shape)

## 4.4 Adding shortcut connections
## 4.4 加入快速关联

class ExampleDeepNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), GELU())
        ])

    def forward(self, x):
        for layer in self.layers:
            # Compute the output of the current layer
            layer_output = layer(x)
            # Check if shortcut can be applied
            if self.use_shortcut and x.shape == layer_output.shape:
                x = x + layer_output
            else:
                x = layer_output
        return x


def print_gradients(model, x):
    # Forward pass
    output = model(x)
    target = torch.tensor([[0.]])

    # Calculate loss based on how close the target
    # and output are
    loss = nn.MSELoss()
    loss = loss(output, target)

    # Backward pass to calculate the gradients
    loss.backward()

    for name, param in model.named_parameters():
        if 'weight' in name:
            # Print the mean absolute gradient of the weights
            print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")

torch.manual_seed(123)
layer_sizes = [3, 3, 3, 3, 3, 1]
sample_input = torch.tensor([[1., 0., -1.]])
model_without_shortcut = ExampleDeepNeuralNetwork(
    layer_sizes, use_shortcut=False
)
print_gradients(model_without_shortcut, sample_input)
# layers.0.0.weight has gradient mean of 0.00020173587836325169
# layers.1.0.weight has gradient mean of 0.00012011159560643137
# layers.2.0.weight has gradient mean of 0.0007152039906941354
# layers.3.0.weight has gradient mean of 0.0013988736318424344
# layers.4.0.weight has gradient mean of 0.005049645435065031

torch.manual_seed(123)
model_with_shortcut = ExampleDeepNeuralNetwork(
    layer_sizes, use_shortcut=True
)
print_gradients(model_with_shortcut, sample_input)
# layers.0.0.weight has gradient mean of 0.22169792652130127
# layers.1.0.weight has gradient mean of 0.20694106817245483
# layers.2.0.weight has gradient mean of 0.32896995544433594
# layers.3.0.weight has gradient mean of 0.2665732204914093
# layers.4.0.weight has gradient mean of 1.3258540630340576

# 一共五层，每层都是先线性，然后GELU
# 没有快连，则 x = layer(x)
# 有快连，则 x += layer(x)
# 通过比较，快捷连接可以防止梯度在早期层也就是Layer0中消失
# 接下来在实现Transformer块的时候也会使用到快捷连接

## 4.5 Connecting attention and linear layers in a transformer block
## 4.5 在转换块中连接注意力和线性层

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../ch03")))

from ch03 import MultiHeadAttention

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        return x

torch.manual_seed(123)
x = torch.rand(2, 4, 768)  # Shape: [batch_size, num_tokens, emb_dim]
block = TransformerBlock(GPT_CONFIG_124M)
output = block(x)
print("Input shape:", x.shape)
print("Output shape:", output.shape)


'''
1. GPT 主心骨
2. 层范化
3. GELU激活
4. 前馈网络
5. 快捷连接
6. 转换块
7. 最终GPT架构
'''

## 4.6 Coding the GPT model
## 4.6 实现GPT模型

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
out = model(batch)
print("Input batch:\n", batch)
print("\nOutput shape:", out.shape)
print(out)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params:,}")
# Total number of parameters: 163,009,536

print("Token embedding layer shape:", model.tok_emb.weight.shape)
print("Output layer shape:", model.out_head.weight.shape)
# Token embedding layer shape: torch.Size([50257, 768])
# Output layer shape: torch.Size([50257, 768])

# 在GPT-2的论文里，研究者
total_params_gpt2 =  total_params - sum(p.numel() for p in model.out_head.parameters())
print(f"Number of trainable parameters considering weight tying: {total_params_gpt2:,}")
# Number of trainable parameters considering weight tying: 124,412,160

# Calculate the total size in bytes (assuming float32, 4 bytes per parameter)
total_size_bytes = total_params * 4
# Convert to megabytes
total_size_mb = total_size_bytes / (1024 * 1024)
print(f"Total size of the model: {total_size_mb:.2f} MB")
# Total size of the model: 621.83 MB
# 模型大小

## 4.7 Generating text
## 4.7 生成文本

def generate_text_simple(model, idx, max_new_tokens,
                         context_size):
    # idx is (batch, n_tokens) array of indices in the current context
    for _ in range(max_new_tokens):
        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        idx_cond = idx[:, -context_size:]

        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond)

        # Focus only on the last time step
        # (batch, n_tokens, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]

        # Apply softmax to get probabilities
        probas = torch.softmax(logits,
                               dim=-1)  # (batch, vocab_size)

        # Get the idx of the vocab entry with the highest probability value
        idx_next = torch.argmax(probas, dim=-1,
                                keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next),
                        dim=1)  # (batch, n_tokens+1)

    return idx

'''
GPT生成文本流程
传入模型类型，输入标记编码，需要的新标记数量，上下文大小
新的标记一个一个生成
取最近的上下文，禁用梯度下降，调用模型的前馈函数，生成概率矩阵
归一化，torch.softmax计算0到1之间的概率
索引查询，orch.argmax取最大概率的token下标
拼成所有新标记
注意该模型尚未训练
'''

start_context = "Hello, I am"
encoded = tokenizer.encode(start_context)
print("encoded:", encoded)
encoded_tensor = torch.tensor(encoded).unsqueeze(0)
print("encoded_tensor:", encoded_tensor)
# encoded: [15496, 11, 314, 716]
# encoded_tensor: tensor([[15496,    11,   314,   716]])

model.eval() # disable dropout
out = generate_text_simple(
    model=model,
    idx=encoded_tensor,
    max_new_tokens=6,
    context_size=GPT_CONFIG_124M["context_length"]
)
print("Output:", out)
print("Output length:", len(out[0]))
decoded_text = tokenizer.decode(out.squeeze(0).tolist())
print(decoded_text)
# Output: tensor([[15496,    11,   314,   716, 27018, 24086, 47843, 30961, 42348,  7267]])
# Output length: 10
# Hello, I am Featureiman Byeswickattribute argue