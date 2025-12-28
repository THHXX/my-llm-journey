import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType

# 1. 配置参数
model_id = "qwen/Qwen1.5-0.5B"  # 基础模型
data_path = os.path.join(os.path.dirname(__file__), "law_data.json") # 训练数据
output_dir = os.path.join(os.path.dirname(__file__), "lora_output") # 输出目录

# 2. 加载 Tokenizer
print(f"🚀 正在加载 Tokenizer: {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token # Qwen 的 pad token 设置

# 3. 加载数据集
print(f"📚 正在加载数据集: {data_path}...")
dataset = load_dataset("json", data_files=data_path, split="train")

# 数据预处理函数
def process_func(example):
    MAX_LENGTH = 256
    input_ids, attention_mask, labels = [], [], []
    
    # 构建 Prompt: 
    # Instruction: ...
    # Input: ...
    # Output: ...
    instruction = example["instruction"]
    inputs = example.get("input", "")
    response = example["output"]
    
    prompt = f"Instruction: {instruction}\nInput: {inputs}\nOutput: "
    
    # 编码
    instruction_ids = tokenizer.encode(prompt, add_special_tokens=True)
    response_ids = tokenizer.encode(response, add_special_tokens=False) + [tokenizer.eos_token_id]
    
    # 拼接
    input_ids = instruction_ids + response_ids
    attention_mask = [1] * len(input_ids)
    
    # Labels (Instruction 部分设为 -100，不计算 Loss)
    labels = [-100] * len(instruction_ids) + response_ids
    
    # 截断
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

tokenized_id = dataset.map(process_func, remove_columns=dataset.column_names)

# 4. 加载模型
print(f"🧠 正在加载模型: {model_id}...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",  # 自动分配设备 (GPU/CPU)
    trust_remote_code=True
)

# 开启梯度检查点时，必须显式开启输入的梯度计算
model.enable_input_require_grads()

# 5. 配置 LoRA
print("🔧 配置 LoRA...")
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False, # 训练模式
    r=8, # LoRA 秩，越大参数越多
    lora_alpha=32, # LoRA 缩放系数
    lora_dropout=0.1 # Dropout 防止过拟合
)

model = get_peft_model(model, config)
model.print_trainable_parameters() # 打印可训练参数量

# 6. 配置训练参数
args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=4, # 批次大小，显存不够改小
    gradient_accumulation_steps=4, # 梯度累积
    logging_steps=10,
    num_train_epochs=3, # 训练轮数
    save_steps=50, 
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=True, # 节省显存
)

# 7. 开始训练
print("🏋️‍♂️ 开始训练...")
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_id,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)

trainer.train()

# 8. 保存模型
print(f"💾 保存 LoRA 权重到 {output_dir}")
trainer.save_model(output_dir)
print("✅ 训练完成！")
