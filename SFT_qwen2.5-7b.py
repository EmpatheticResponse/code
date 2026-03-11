import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
os.environ["HTTP_PROXY"] = "114.214.236.243:7890"
os.environ["HTTPS_PROXY"] = "114.214.236.243:7890"

import wandb
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset, Dataset
import pandas as pd
from trl import SFTTrainer, SFTConfig
import re

wandb.login()

class SFT_Config:
    # BASE_MODEL = "/data/wjq/code/new_les/Qwen2.5-7B-Instruct"
    BASE_MODEL = "/data/wjq/code/new_les/Qwen2.5-7B-unsloth-bnb-4bit"
    # BASE_MODEL = "/data/wjq/code/new_les/GRPO_qwen_2.5/outputs_sft_qwen2.5_2e/checkpoint-6500"
    
    # SFT_DATA_PATH = "/data/wjq/code/new_les/processed_data_with_ids_test_valid.csv"
    SFT_DATA_PATH = "/data/wjq/code/new_les/processed_data_with_ids_train.csv"
    OUTPUT_DIR = "./outputs_sft_qwen2.5-7b_tmp"
    ADAPTERS_SAVE_PATH = "./outputs_sft_qwen2.5-7b_tmp/lora_adapters" # 定义 SFT 适配器的保存路径

    MAX_SEQ_LENGTH = 2048
    LORA_RANK = 32


    # SFT 训练超参数
    LEARNING_RATE = 2e-4
    NUM_EPOCHS = 5
    BATCH_SIZE = 4
    GRAD_ACCUM_STEPS = 1
    OPTIM = "adamw_8bit"
    RANDOM_SEED = 3407
    

    WANDB_PROJECT = "SFT-qwen2.5-7B"
    WANDB_RUN_NAME = "SFT"

df = pd.read_csv(SFT_Config.SFT_DATA_PATH)
full_dataset = Dataset.from_pandas(df)
split_dataset = full_dataset.train_test_split(test_size=0.001, seed=3407, shuffle=True)
train_df = split_dataset['train'].to_pandas()
eval_df = split_dataset['test'].to_pandas()
class_counts = train_df['emotion'].value_counts()

# 确定一个目标数量（例如，中位数或平均数）
target_count = int(class_counts.median()) 

oversampled_dfs = [train_df]
for emotion, count in class_counts.items():
    if count < target_count:
        class_subset_df = train_df[train_df['emotion'] == emotion]
        num_to_add = target_count - count
        oversampled_dfs.append(class_subset_df.sample(n=num_to_add, replace=True, random_state=SFT_Config.RANDOM_SEED))

# 合并成最终的训练数据集
balanced_train_df = pd.concat(oversampled_dfs)
train_dataset = Dataset.from_pandas(balanced_train_df).shuffle(seed=SFT_Config.RANDOM_SEED)
eval_dataset = Dataset.from_pandas(eval_df)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = SFT_Config.BASE_MODEL,
    # model_name = SFT_Config.ADAPTERS_SAVE_PATH,
    max_seq_length = SFT_Config.MAX_SEQ_LENGTH,
    load_in_4bit = True,
    # fast_inference = True, # Enable vLLM fast inference
    max_lora_rank = SFT_Config.LORA_RANK,
    gpu_memory_utilization = 0.8, # Reduce if out of memory
)
# print(tokenizer.eos_token)
# input()
model = FastLanguageModel.get_peft_model(
    model,
    r = SFT_Config.LORA_RANK,
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha = SFT_Config.LORA_RANK * 2,
    use_gradient_checkpointing = "unsloth", 
    random_state = SFT_Config.RANDOM_SEED,
)

model.config.use_cache = False

reasoning_affective       = "<affective>"
reasoning_affective_end   = "</affective>"
reasoning_cognitive       = "<cognitive>"
reasoning_cognitive_end   = "</cognitive>"
solution_start = "<solution>"
solution_end   = "</solution>"


system_prompt = \
    f"""
    You are an empathetic dialogue agent.
    OUTPUT RULES (follow exactly; output nothing else):
    1) First write:
    <affective>...</affective><cognitive>...</cognitive>
    2) Then write:
    <solution>
    emotion: <one of: acknowledging,afraid,agreeing,angry,annoyed,anticipating,anxious,apprehensive,
    ashamed,caring,confident,consoling,content,devastated,disappointed,disgusted,embarrassed,encouraging,
    excited,faithful,furious,grateful,guilty,hopeful,impressed,jealous,joyful,lonely,neutral,nostalgic,prepared,
    proud,questioning,sad,sentimental,suggesting,surprised,sympathizing,terrified,trusting,wishing>
    response: <your empathetic response>
    </solution>

    STRICT CONSTRAINTS:
    - Do not write "Step", bullet points, explanations, or any other text.
    - Do not use code fences or Markdown.
    - Keep the two keys exactly as: "emotion:" and "response:" (lowercase, each on its own line).
    """
# chat_template = """
# {% for message in messages %}
#     {% if message['role'] == 'system' %}
#         {{ message['content'] + eos_token }}
#     {% elif message['role'] == 'user' %}
#         {{ 'user: ' + message['content'] + eos_token }}
#     {% elif message['role'] == 'assistant' %}
#         {{ 'assistant: ' + message['content'] + eos_token }}
#     {% endif %}
# {% endfor %}
# {% if add_generation_prompt %}
#     {{ '<|im_start|>assistant\n' }}
# {% endif %}
# """


# genemi 建议的
chat_template = """
{%- for message in messages -%}
<|im_start|>{{ message['role'] }}
{{ message['content'] }}<|im_end|>
{%- endfor -%}
{%- if add_generation_prompt -%}
<|im_start|>assistant
{%- endif -%}
"""

tokenizer.chat_template = chat_template



from typing import List, Dict, Any

class LastAssistantOnlyCollator:
    """
    只对“最后一个 <|im_start|>assistant\\n … <|im_end|>”之间的内容计损；
    其它 token 全设为 -100。
    - 若特征里已有 input_ids/attention_mask：直接使用（最快，兼容 SFTTrainer 预处理/packing）
    - 否则回退读取 text/prompt 并分词
    """
    def __init__(self, tokenizer, response_template: str = "<|im_start|>assistant\n", max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        # 模板 token 序列（用于定位“最后一个 assistant 开头”）
        self.template_ids = tokenizer.encode(response_template, add_special_tokens=False)

        # pad_token 兜底（有些 Qwen 模型没显式 pad）
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _find_last_start(self, ids: List[int]) -> int:
        t = self.template_ids
        if not t or len(ids) < len(t):
            return -1
        last = -1
        for i in range(0, len(ids) - len(t) + 1):
            if ids[i:i+len(t)] == t:
                last = i
        return last

    def _ensure_list(self, x):
        # 可能已经是 torch.Tensor；统一转 list
        return x.tolist() if hasattr(x, "tolist") else list(x)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids_list, attn_list, labels_list = [], [], []
        max_len = 0
        pad_id = self.tokenizer.pad_token_id

        for f in features:
            # 优先走“已分词”的路径
            if "input_ids" in f:
                ids  = self._ensure_list(f["input_ids"])
                attn = self._ensure_list(f.get("attention_mask", [1]*len(ids)))
            else:
                # 回退：从 text/prompt 分词
                text = f.get("text") or f.get("prompt")
                if text is None:
                    raise KeyError("Neither 'input_ids' nor 'text'/'prompt' found in batch feature.")
                enc = self.tokenizer(
                    text,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=self.max_length,
                )
                ids, attn = enc["input_ids"], enc["attention_mask"]

            # 计算只对“最后一个 assistant 段”计损的 labels
            start = self._find_last_start(ids)
            labels = [-100] * len(ids)
            if start != -1:
                start_pos = start + len(self.template_ids)  # 屏蔽模板本身，只对内容计损
                for j in range(start_pos, len(ids)):
                    labels[j] = ids[j]

            input_ids_list.append(ids)
            attn_list.append(attn)
            labels_list.append(labels)
            max_len = max(max_len, len(ids))

        # 手动 pad 到 batch 内最大长度
        for i in range(len(input_ids_list)):
            pad_len = max_len - len(input_ids_list[i])
            if pad_len > 0:
                input_ids_list[i] += [pad_id] * pad_len
                attn_list[i]     += [0]      * pad_len
                labels_list[i]   += [-100]   * pad_len

        return {
            "input_ids": torch.tensor(input_ids_list, dtype=torch.long),
            "attention_mask": torch.tensor(attn_list, dtype=torch.long),
            "labels": torch.tensor(labels_list, dtype=torch.long),
        }
# 你的 chat_template 会在生成前缀里放入："<|im_start|>assistant\n"
response_template = "<|im_start|>assistant\n"

collator = LastAssistantOnlyCollator(
    response_template=response_template,
    tokenizer=tokenizer,
    max_length=SFT_Config.MAX_SEQ_LENGTH,
)



def format_sft_dataset(x):
    # 1. 初始化messages列表，并添加系统提示
    messages = [{"role": "system", "content": system_prompt}]
    
    # 2. 解析多轮对话历史
    dialogue_history_raw = x["dialogue_history"]
    turns = dialogue_history_raw.split(' | ')
    
    for turn in turns:
        # 安全地解析每一轮的角色和内容
        role_match = re.search(r"role:\s*([a-z]+)", turn)
        content_match = re.search(r"content:\s*(.*)", turn, re.DOTALL)
        
        if role_match and content_match:
            role = role_match.group(1)
            # 清理内容，去除可能存在的 dialogue_emotion 标签和多余的标点
            content_raw = content_match.group(1)
            content_cleaned = re.sub(r",?\s*dialogue_emotion:.*", "", content_raw).strip().strip("'").strip(".").strip()
            
            # 将解析出的对话轮次添加到messages列表中
            messages.append({"role": role, "content": content_cleaned})

    # 3. 添加模型需要学习生成的最终回复
    resp_text = re.sub(r"(?is)^.*?response:\s*", "", str(x["response"])).strip()
    final_assistant_response = (
        f"{x['generated_solution'].strip()}\n"
        f"{solution_start}\n"
        f"emotion: {str(x['emotion']).strip().lower()}\n"
        f"response: {resp_text}\n"
        f"{solution_end}"
    )
    messages.append({"role": "assistant", "content": final_assistant_response})
    
    # 4. 使用分词器的模板功能来处理整个对话列表
    #    add_generation_prompt=False 是因为我们已经提供了完整的对话，包括最后一轮助手的回答

    formatted_text = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=False
    )
    return {
        "text": formatted_text.rstrip() + tokenizer.eos_token
    }
    # return {
    #     "text": tokenizer.apply_chat_template(
    #         messages, 
    #         tokenize=False, 
    #         add_generation_prompt=False
    #     )
    # }


ori_sft_dataset = Dataset.from_pandas(pd.read_csv(SFT_Config.SFT_DATA_PATH))
print(len(ori_sft_dataset))
sft_dataset = train_dataset
print(len(sft_dataset))
sft_dataset = sft_dataset.map(format_sft_dataset, remove_columns=sft_dataset.column_names)


for i in range(1):
    print(f"------------ 样本 {i+1} ------------")
    print(sft_dataset[i]['text'])
    print("\n")

wandb.init(
    project=SFT_Config.WANDB_PROJECT,
    name=SFT_Config.WANDB_RUN_NAME,
)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = sft_dataset,
    dataset_text_field = "text",
    max_seq_length = SFT_Config.MAX_SEQ_LENGTH,
    data_collator=collator,
    args = SFTConfig(
        per_device_train_batch_size = SFT_Config.BATCH_SIZE,
        gradient_accumulation_steps = SFT_Config.GRAD_ACCUM_STEPS,
        warmup_steps = 100,
        num_train_epochs = SFT_Config.NUM_EPOCHS,
        learning_rate = SFT_Config.LEARNING_RATE,
        logging_steps = 1,
        optim = SFT_Config.OPTIM,
        seed = SFT_Config.RANDOM_SEED,
        weight_decay = 0.01,
        # lr_scheduler_type = "linear",
        lr_scheduler_type = "cosine",
        report_to = "wandb",
        # report_to = "none",
        output_dir = SFT_Config.OUTPUT_DIR,

        save_strategy = "steps",    # 设置保存策略为按步数保存。可选值为 "steps" 或 "epoch"。
        # save_strategy = "no",    # 设置保存策略为按步数保存。可选值为 "steps" 或 "epoch"。
        save_steps = 500,           # 每训练200步保存一个检查点。
        save_total_limit = 100,    
    ),
)

# trainer.train(resume_from_checkpoint=True)
trainer.train()


model.to(torch.bfloat16)

print(f"正在将训练好的 LoRA 适配器以 bfloat16 格式保存到 '{SFT_Config.ADAPTERS_SAVE_PATH}'...")
model.save_pretrained(SFT_Config.ADAPTERS_SAVE_PATH)
tokenizer.save_pretrained(SFT_Config.ADAPTERS_SAVE_PATH)
print("保存完成！")

# model.save_pretrained(SFT_Config.ADAPTERS_SAVE_PATH)
# tokenizer.save_pretrained(SFT_Config.ADAPTERS_SAVE_PATH) # 同时保存分词器配置

print("保存完成！")

wandb.finish()