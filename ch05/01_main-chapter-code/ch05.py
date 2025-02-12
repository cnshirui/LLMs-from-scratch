import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../ch04")))

# from ch04 import GPTModel

import torch
from previous_chapters import GPTModel

GPT_CONFIG_124M = {
    "vocab_size": 50257,   # Vocabulary size
    "context_length": 256, # Shortened context length (orig: 1024)
    "emb_dim": 768,        # Embedding dimension
    "n_heads": 12,         # Number of attention heads
    "n_layers": 12,        # Number of layers
    "drop_rate": 0.1,      # Dropout rate
    "qkv_bias": False      # Query-key-value bias
}

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.eval()  # Disable dropout during inference

'''
torch.nn.Module.eval()将模型切换到评估模式
nn.Dropout层在训练模式下，会随机将一部分神经元置零，以防止过分拟合
nn.Dropout层在评估模式下，不会随机丢失神经元，而是保存其所有权重

torch.no_grad() 关闭梯度计算，提高推理速度，减少内存消耗

默认开启张量计算梯度，或者torch.enable_grad()
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2
y.backward() # dy/dx = 2 * x = 4
print(x.grad) # 4

梯度计算主要是为了反向传播，back propagation，从而计算损失函数对模型参数的梯度，然后迭代更新参数来优化模型的性能 

PyTorch通过自动微分，AutoGrad
前向传播，ForwardPass，输入数据，计算输出值
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2
假设损失函数为loss = y - 3 
loss.backward()
PyTorch通过链式求导法则计算梯度并存储在x.grad里面
训练神经网络的时候，使用梯度下降 Gradient Descent 来更新模型参数 

===

模型参数 theta
学习率 eta
梯度 delta(theta, loss)
theta = theta - eta * delta

x = torch.tensor([1, 2, 3])
y = x.unsqueeze(0) # [[1, 2, 3]]
z = y.squeeze(0) # 挤 [1,2,3]
print(x, y, z)

'''

### 5.1.1 Using GPT to generate text
### 5.1.1 使用GPT生成文字

import tiktoken
from previous_chapters import generate_text_simple

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # add batch dimension
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0) # remove batch dimension
    return tokenizer.decode(flat.tolist())

start_context = "Every effort moves you"
tokenizer = tiktoken.get_encoding("gpt2")

token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(start_context, tokenizer),
    max_new_tokens=10,
    context_size=GPT_CONFIG_124M["context_length"]
)

print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
# Output text:
#  Every effort moves you rentingetic wasnم refres RexMeCHicular stren

x = torch.tensor([1, 2, 3])
y = x.unsqueeze(0) # [[1, 2, 3]]
z = y.squeeze(0) # [1,2,3]
print(x, y, z)

### 5.1.2 Calculating the text generation loss: cross-entropy and perplexity
### 5.1.2 计算文本生成损失：交叉熵和困惑度

inputs = torch.tensor([[16833, 3626, 6100],   # ["every effort moves",
                       [40,    1107, 588]])   #  "I really like"]

targets = torch.tensor([[3626, 6100, 345  ],  # [" effort moves you",
                        [1107,  588, 11311]]) #  " really like chocolate"]

with torch.no_grad():
    logits = model(inputs)

# 归一化，torch.softmax计算0到1之间的概率
probas = torch.softmax(logits, dim=-1) # Probability of each token in vocabulary
print(probas.shape) # Shape: (batch_size, num_tokens, vocab_size)
# torch.Size([2, 3, 50257])

# 索引查询，orch.argmax取最大概率的token下标
token_ids = torch.argmax(probas, dim=-1, keepdim=True)
print("Token IDs:\n", token_ids)
# tensor([[[16657],
#          [339],
#          [42826]],
#
#         [[49906],
#          [29669],
#          [41751]]])

print(f"Targets batch 1: {token_ids_to_text(targets[0], tokenizer)}")
print(f"Outputs batch 1: {token_ids_to_text(token_ids[0].flatten(), tokenizer)}")
# Targets batch 1:  effort moves you
# Outputs batch 1:  Armed heNetflix

text_idx = 0
target_probas_1 = probas[text_idx, [0, 1, 2], targets[text_idx]]
print("Text 1:", target_probas_1)
# Text 1: tensor([7.4541e-05, 3.1061e-05, 1.1563e-05])

text_idx = 1
target_probas_2 = probas[text_idx, [0, 1, 2], targets[text_idx]]
print("Text 2:", target_probas_2)
# Text 2: tensor([1.0337e-05, 5.6776e-05, 4.7559e-06])
# 目标是使这三个都最大？

# Compute logarithm of all token probabilities
log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))
print(log_probas)
# tensor([ -9.5042, -10.3796, -11.3677, -11.4798,  -9.7764, -12.2561])

# Calculate the average probability for each token
avg_log_probas = torch.mean(log_probas)
print(avg_log_probas)
# tensor(-10.7940)
# log(target=1) = 0

neg_avg_log_probas = avg_log_probas * -1
print(neg_avg_log_probas)
# tensor(10.7940)
# 交叉熵：让概率的对数的负数最小，损失函数
# 交叉熵的损失就是大语言模型的困惑度

# Logits have shape (batch_size, num_tokens, vocab_size)
print("Logits shape:", logits.shape)
# Logits shape: torch.Size([2, 3, 50257])

# Targets have shape (batch_size, num_tokens)
print("Targets shape:", targets.shape)
# Targets shape: torch.Size([2, 3])

logits_flat = logits.flatten(0, 1)
targets_flat = targets.flatten()
print("Flattened logits:", logits_flat.shape)
print("Flattened targets:", targets_flat.shape)
# Flattened logits: torch.Size([6, 50257])
# Flattened targets: torch.Size([6])
# 第0维和第1维压平。

loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
print('loss: ', loss)
# loss:  tensor(10.7940)
# targets： token index
# logits：对应位置每个token出现的概率
# cross_entropy: 这些token概率的平均值

perplexity = torch.exp(loss)
print('perplexity: ', perplexity)
# perplexity:  tensor(48725.8203)

'''
困惑度差不多是交叉熵的指数
困惑度更能解释大语言模型，其可理解为模型在每个步骤中不确定的有效单词量
最少不确定性为1，最大不确定性为词汇数量
'''

### 5.1.3 Calculating the training and validation set losses
### 5.1.3 计算训练集和验证集的损失

import os
import urllib.request

file_path = "the-verdict.txt"
url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"

if not os.path.exists(file_path):
    with urllib.request.urlopen(url) as response:
        text_data = response.read().decode('utf-8')
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(text_data)
else:
    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()

# First 100 characters
print(text_data[:99])

# Last 100 characters
print(text_data[-99:])

total_characters = len(text_data)
total_tokens = len(tokenizer.encode(text_data))
print("Characters:", total_characters)
print("Tokens:", total_tokens)
# Characters: 20479
# Tokens: 5145

from previous_chapters import create_dataloader_v1

# Train/validation ratio
train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
valid_data = text_data[split_idx:]

torch.manual_seed(123)
train_loader = create_dataloader_v1(
    train_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=0
)

valid_loader = create_dataloader_v1(
    valid_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=False,
    shuffle=False,
    num_workers=0
)

# Sanity check
if total_tokens * (train_ratio) < GPT_CONFIG_124M["context_length"]:
    print("Not enough tokens for the training loader. "
          "Try to lower the `GPT_CONFIG_124M['context_length']` or "
          "increase the `training_ratio`")

if total_tokens * (1-train_ratio) < GPT_CONFIG_124M["context_length"]:
    print("Not enough tokens for the validation loader. "
          "Try to lower the `GPT_CONFIG_124M['context_length']` or "
          "decrease the `training_ratio`")

print("Train loader:")
for x, y in train_loader:
    print(x.shape, y.shape)

print("\nValidation loader:")
for x, y in valid_loader:
    print(x.shape, y.shape)

train_tokens = 0
for input_batch, target_batch in train_loader:
    train_tokens += input_batch.numel()

valid_tokens = 0
for input_batch, target_batch in valid_loader:
    valid_tokens += input_batch.numel()

print("Training tokens:", train_tokens)
print("Validation tokens:", valid_tokens)
print("All tokens:", train_tokens + valid_tokens)

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # Reduce the number of batches to match the total number of batches in the data loader
        # if num_batches exceeds the number of batches in the data loader
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches

# Note:
# Uncommenting the following lines will allow the code to run on Apple Silicon chips, if applicable,
# which is approximately 2x faster than on an Apple CPU (as measured on an M3 MacBook Air).
# However, the resulting loss values may be slightly different.

#if torch.cuda.is_available():
#    device = torch.device("cuda")
#elif torch.backends.mps.is_available():
#    device = torch.device("mps")
#else:
#    device = torch.device("cpu")
#
# print(f"Using {device} device.")

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Training completed in 2.79 minutes.

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# Training completed in 1.28 minutes.

model.to(device) # no assignment model = model.to(device) necessary for nn.Module classes
torch.manual_seed(123) # For reproducibility due to the shuffling in the data loader
train_loss, valid_loss = 0, 0
with torch.no_grad(): # Disable gradient tracking for efficiency because we are not training, yet
    train_loss = calc_loss_loader(train_loader, model, device)
    valid_loss = calc_loss_loader(valid_loader, model, device)
print("Training loss:", train_loss)
print("Validation loss:", valid_loss)
# Training loss: 10.987583584255642
# Validation loss: 10.98110580444336

## 5.2 Training an LLM
## 5.2 训练大语言模型

'''
- 对于每个训练轮次 epoch 时代
    - epoch 是对整个训练集的一次完整遍历
    - 对于训练集中的每个批次 batch
        - 批次数量 = 训练集大小 / 批次大小
        - 清除上一批次的梯度
        - 计算当前批次的损失
        - 反向传播，计算损失的梯度
        - 使用梯度更新模型权重
        - 打印训练集和验证集的损失
    - 生成示例文本进行可视化检查
'''

'''
- torch.nn.Module.train()将模型切换到训练模式
    - nn.Dropout 丢弃层，会随机丢弃一部分神经元，防止过分拟合
    - nn.BatchNorm  批归一化层，使用当前批次均值和方法，更新到全局均值和方差
torch.nn.Module.eval()将模型切换到评估模式
    - nn.Dropout 丢弃层会被禁用
    - nn.BatchNorm  批归一化层，不用更新，使用训练时存储的全局均值和方差

torch.no_grad() 关闭梯度计算，提高推理速度，减少内存消耗

'''

def train_model_simple(model, train_loader, valid_loader,
                       optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context,
                       tokenizer):
    # Initialize lists to track losses and tokens seen
    train_losses, valid_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # Reset loss gradients from previous batch iteration
            loss = calc_loss_batch(input_batch,
                                   target_batch, model,
                                   device)
            loss.backward()  # Calculate loss gradients
            optimizer.step()  # Update model weights using loss gradients
            tokens_seen += input_batch.numel()
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, valid_loss = evaluate_model(
                    model, train_loader, valid_loader, device,
                    eval_iter)
                train_losses.append(train_loss)
                valid_losses.append(valid_loss)
                track_tokens_seen.append(tokens_seen)
                print(
                    f"Ep {epoch + 1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, Val loss {valid_loss:.3f}")

        # Print a sample text after each epoch
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )

    return train_losses, valid_losses, track_tokens_seen


def evaluate_model(model, train_loader, valid_loader, device,
                   eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model,
                                      device,
                                      num_batches=eval_iter)
        valid_loss = calc_loss_loader(valid_loader, model,
                                    device,
                                    num_batches=eval_iter)
    model.train()
    return train_loss, valid_loss


def generate_and_print_sample(model, tokenizer, device,
                              start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context,
                                tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n",
                               " "))  # Compact print format
    model.train()

# Note:
# Uncomment the following code to calculate the execution time
import time
start_time = time.time()

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004,
                              weight_decay=0.1)

num_epochs = 10
train_losses, valid_losses, tokens_seen = train_model_simple(
    model, train_loader, valid_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=5,
    start_context="Every effort moves you",
    tokenizer=tokenizer
)

# Every effort moves you?"  "Yes--quite insensible to the irony. She wanted him vindicated--and by me!"  He laughed again, and threw back his head to look up at the sketch of the donkey. "There were days when I

# Note:
# Uncomment the following code to show the execution time
end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes.")

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(5, 3))

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))  # only show integer labels on x-axis

    # Create a second x-axis for tokens seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()  # Adjust layout to make room
    plt.savefig("loss-plot.pdf")
    plt.show()

epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
plot_losses(epochs_tensor, tokens_seen, train_losses, valid_losses)

'''
# 模型的所有可训练参数（比如神经网络中的权重和偏置）传入优化器，用于计算梯度并更新
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004,
                              weight_decay=0.1)
# 清除之前批次的梯度信息
optimizer.zero_grad()  # Reset loss gradients from previous batch iteration

# 计算当前批次的损失
loss = calc_loss_batch(input_batch,
                       target_batch, model,
                       device)
                
# 通过反向传播计算损失函数对于模型参数的梯度，链式法则，存储在各个参数的grad属性中       
loss.backward()  # Calculate loss gradients

# 使用计算得到的梯度来更新模型的参数
optimizer.step()  # Update model weights using loss gradients                           
'''

## 5.3 Decoding strategies to control randomness
## 5.3 控制随机性的解码策略
















