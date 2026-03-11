import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["HTTP_PROXY"] = "114.214.236.243:7890"
os.environ["HTTPS_PROXY"] = "114.214.236.243:7890"
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["no_proxy"] = "127.0.0.1,localhost"

import wandb
wandb.login()


soothing_lexicon = {
    # === 积极情感 ===
    "joyful": ["wonderful", "fantastic", "so happy for you", "that's great news"],
    "excited": ["how exciting", "can't wait to hear more", "sounds amazing"],
    "proud": ["you should be proud", "congratulations", "that's a real achievement"],
    "confident": ["that's the spirit", "you've got this", "believe in yourself"],
    "grateful": ["you're very welcome", "my pleasure", "happy to help"],
    "hopeful": ["fingers crossed for you", "let's hope for the best", "sounds promising"],
    "caring": ["take care", "thinking of you", "here for you"],
    "encouraging": ["you can do it", "keep going", "don't give up", "one step at a time"],
    "impressed": ["that's impressive", "wow", "i'm truly impressed"],
    "content": ["glad to hear that", "sounds peaceful", "that's lovely"],

    # === 负面情感 ===
    "anxious": ["it's okay to feel that way", "take a deep breath", "one thing at a time", "no pressure"],
    "apprehensive": ["i understand the uncertainty", "we can take it slow", "what are your concerns"],
    "afraid": ["that does sound scary", "i'm here with you", "it's okay to be afraid"],
    "terrified": ["that sounds terrifying", "you are safe now", "let's work through this together"],
    "angry": ["that sounds frustrating", "i can see why you're upset", "your feelings are valid"],
    "furious": ["you have every right to be furious", "that's completely unacceptable"],
    "annoyed": ["i can imagine how annoying that is", "that would bother me too"],
    "disgusted": ["that's awful", "i understand your reaction"],
    "ashamed": ["we all make mistakes", "be kind to yourself", "it's a learning experience"],
    "embarrassed": ["that sounds like a tough moment", "it happens to everyone"],
    "guilty": ["it's okay", "forgive yourself", "let's focus on moving forward"],
    "sad": ["i'm so sorry to hear that", "that must be so difficult", "it's okay to be sad"],
    "disappointed": ["that's really disappointing", "i'm sorry it didn't work out"],
    "devastated": ["i can't imagine how hard that is", "sending you so much support"],
    "lonely": ["you're not alone in feeling this way", "i'm here to listen"],

    # === 中性及其他情感 ===
    "neutral": ["i see", "got it", "thank you for sharing", "okay"],
    "acknowledging": ["i hear you", "that makes sense", "i understand"],
    "prepared": ["sounds like you have a good plan", "you're ready for this"],
    "questioning": ["that's a good question", "let's explore that", "worth thinking about"],
    "anticipating": ["sounds exciting", "what are you looking forward to"],
    "nostalgic": ["that's a sweet memory", "sounds very meaningful"],
    "sentimental": ["thank you for sharing that with me", "cherish those moments"],
}



# 行动/许可/黑名单
action_verbs = ["try", "write down", "spend a few minutes", "practice", "schedule", "reach out"]
permission_markers = ["if you want", "you could", "you might", "if you feel comfortable"]
unsafe_blacklist = ["stop medication", "self harm", "gamble", "loan", "diagnose"]

# 情绪大类（正/负/中）——供规则分支使用
POS = {
    "joyful","excited","proud","confident","grateful","hopeful",
    "caring","encouraging","impressed","content"
}
NEG = {
    "anxious","apprehensive","afraid","terrified","angry","furious","annoyed","disgusted",
    "ashamed","embarrassed","guilty","sad","disappointed","devastated","lonely"
}
NEU = {
    "neutral","acknowledging","prepared","questioning","anticipating","nostalgic","sentimental"
}

STYLE_PHRASES_BY_EMOTION = soothing_lexicon


def main():
    from unsloth import FastLanguageModel, is_bfloat16_supported
    import torch
    import re, requests
    from datasets import load_dataset
    from trl.trainer.grpo_config_crpo import GRPOConfig
    # from trl.trainer.grpo_trainer_crpo import GRPOTrainer
    from trl.trainer.grpo_trainer_grpo import GRPOTrainer
    # from trl import GRPOConfig, GRPOTrainer
    from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, get_scheduler
    import torch.nn as nn
    from sentence_transformers import SentenceTransformer, util
    import os, time, hashlib, requests, json
    from transformers import TrainerCallback

    from accelerate.utils import extract_model_from_parallel
    from peft import PeftModel, get_peft_model_state_dict
    from safetensors.torch import save_file as st_save
    from safetensors.torch import load_file as st_load

    sbert = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')


    # ==== 兼容版：只让 LoRA 可训练（替代 mark_only_lora_as_trainable） ====
    def mark_only_lora_as_trainable_compat(model, bias="none"):
        """
        让仅 LoRA 相关权重参与训练。
        兼容 lora_A/lora_B 以及 lora_up/lora_down 等命名。
        bias: "none" | "lora_only" | "all"
        """
        bias_names = set()
        for n, p in model.named_parameters():
            low = n.lower()
            is_lora = ("lora_" in low) or ("lora.up" in low) or ("lora.down" in low) or ("lora_up" in low) or ("lora_down" in low)
            p.requires_grad = bool(is_lora)
            if ".bias" in low:
                bias_names.add(n)

        if bias == "lora_only":
            # 仅让 LoRA 对应层的 bias 也参与训练（需要你确认是否需要）
            for n, p in model.named_parameters():
                if n in bias_names and (("lora" in n.lower())):
                    p.requires_grad = True
        elif bias == "all":
            for n, p in model.named_parameters():
                if n in bias_names:
                    p.requires_grad = True


    class LoraSyncCallback(TrainerCallback):
        """
        同步 (sync_every) ：把LoRA权重写到固定的 live 目录，并通知 vLLM 热重载
        快照 (snapshot_every)：额外把同一份权重保存为 step_<gs>/ 目录（历史留存）
        """
        def __init__(self,
                    base_url: str,
                    adapter_name: str,
                    save_root: str,
                    sync_every: int = 1,
                    snapshot_every: int = 2000,
                    live_subdir: str = "live"):
            self.base_url = base_url.rstrip("/")
            self.adapter_name = adapter_name
            self.save_root = save_root
            self.sync_every = int(sync_every)
            self.snapshot_every = int(snapshot_every)
            self.live_dir = os.path.join(save_root, live_subdir)
            self.last_sync_step = -1
            os.makedirs(self.save_root, exist_ok=True)
            os.makedirs(self.live_dir, exist_ok=True)

        # ------- 小工具：写json(原子写)，写权重(safetensors) -------
        

        @staticmethod
        def _atomic_write_json(path: str, obj: dict):
            s = json.dumps(obj, ensure_ascii=False, indent=2)
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                f.write(s)
            os.replace(tmp, path)

        @staticmethod
        def _write_weights(path: str, sd: dict):
            cpu_state = {k: v.detach().to("cpu").contiguous() for k, v in sd.items()}
            st_save(cpu_state, path)
            # verify
            _tmp = st_load(path)
            print(f"[LoRA SAVE] verify weights: wrote {len(_tmp.keys())} tensors -> {path}")

        def _dump_min_adapter_config(self, peft_model: PeftModel, out_dir: str):
            active = getattr(peft_model, "active_adapter", "default")
            pcfg = peft_model.peft_config[active]
            def _get(obj, name, default=None):
                return getattr(obj, name, default)
            cfg = {
                "peft_type": "LORA",
                "task_type": "CAUSAL_LM",
                "base_model_name_or_path": _get(pcfg, "base_model_name_or_path", None),
                "inference_mode": True,
                "r": int(_get(pcfg, "r", 64)),
                "lora_alpha": int(_get(pcfg, "lora_alpha", _get(pcfg, "r", 64))),
                "lora_dropout": float(_get(pcfg, "lora_dropout", 0.0)),
                "bias": _get(pcfg, "bias", "none"),
                "target_modules": list(_get(pcfg, "target_modules", [
                    "q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj",
                ])),
                "use_rslora": bool(_get(pcfg, "use_rslora", False)),
            }
            os.makedirs(out_dir, exist_ok=True)
            self._atomic_write_json(os.path.join(out_dir, "adapter_config.json"), cfg)

        # ------- 核心：从 Trainer 取出 PeftModel 的 LoRA state_dict -------
        @staticmethod
        def _get_active_lora_state_dict(model: torch.nn.Module) -> tuple[dict, PeftModel, str]:
            peft_model = extract_model_from_parallel(model)
            if not isinstance(peft_model, PeftModel):
                raise RuntimeError("Current model is not a PeftModel; LoRA not injected into Trainer's model.")
            active = getattr(peft_model, "active_adapter", "default")
            sd = get_peft_model_state_dict(peft_model, adapter_name=active)
            return sd, peft_model, active

        # ------- vLLM unload/load -------
        def _vllm_unload(self):
            try:
                r = requests.post(f"{self.base_url}/v1/unload_lora_adapter",
                                json={"lora_name": self.adapter_name}, timeout=10)
                print(f"[LoRA] unload {self.adapter_name} -> {r.status_code}")
            except Exception as e:
                print("[LoRA] unload error:", e)

        def _vllm_load(self, lora_dir: str):
            try:
                r = requests.post(f"{self.base_url}/v1/load_lora_adapter",
                                json={"lora_name": self.adapter_name, "lora_path": os.path.abspath(lora_dir)},
                                timeout=30)
                print(f"[LoRA] load {self.adapter_name} -> {r.status_code} {r.text if r.status_code!=200 else 'OK'}")
            except Exception as e:
                print("[LoRA] load error:", e)

        # ------- 回调：每个 optimizer step 触发一次 -------
        def on_step_end(self, args, state, control, **kwargs):
                gs = state.global_step
                # 仅在 optimizer.step() 后有意义；gs 从 1 开始
                if gs == self.last_sync_step or gs <= 0:
                    return

                self.last_sync_step = gs

                try:
                    model = kwargs["model"]
                    peft_model = extract_model_from_parallel(model)
                except Exception:
                    return

                # ====== 同步逻辑 ======
                if (self.sync_every > 0) and (gs % self.sync_every == 0):
                    # 位置 1: 确认进入了同步逻辑
                    # 原有代码：print(f"[LoRA SYNC] Global Step {gs}: Updating vLLM...")
                    # 建议修改为带时间戳，方便你看延迟：
                    import datetime
                    print(f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] [LoRA SYNC] Step {gs}: Start syncing to vLLM...") 

                    peft_model.save_pretrained(self.live_dir)
                    
                    # 简单粗暴的重试机制，防止网络阻塞
                    for i in range(3): # 建议把 _ 改成 i，方便看是第几次尝试
                        try:
                            self._vllm_unload()
                            # ✅ [建议添加 2]：确认卸载请求已发送（虽然可能404，但代表代码跑到了这里）
                            # print(f"   (Try {i+1}) Unload request sent.") 

                            time.sleep(0.5) 
                            
                            self._vllm_load(self.live_dir)
                            
                            # ✅ [建议添加 3 - 最重要]：确认加载成功！
                            # 如果代码能运行到这一行，说明上面没有报错，说明 vLLM 返回了 200 OK
                            print(f"✅ [SUCCESS] vLLM reloaded successfully at step {gs}!")
                            break
                        except Exception as e:
                            print(f"❌ [Sync Error] Attempt {i+1} failed: {e}, retrying...")
                            time.sleep(1)
                
                if (self.snapshot_every > 0) and (gs % self.snapshot_every == 0):
                    snap_dir = os.path.join(self.save_root, f"step_{gs}")
                    peft_model.save_pretrained(snap_dir)
                    print(f"[LoRA SNAPSHOT] Saved to {snap_dir}")

    # ----- NLI model globals -----
    _NLI_MODEL_NAME = os.getenv("NLI_MODEL_NAME", "roberta-large-mnli")
    _nli_tokenizer = None
    _nli_model = None
    _nli_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # _nli_device = torch.device("cpu")

    def get_nli_model():
        nonlocal _nli_model, _nli_tokenizer
        if _nli_model is None or _nli_tokenizer is None:
            _nli_tokenizer = AutoTokenizer.from_pretrained(_NLI_MODEL_NAME)
            _nli_model = AutoModelForSequenceClassification.from_pretrained(_NLI_MODEL_NAME).to(_nli_device)
            _nli_model.eval()
        return _nli_model, _nli_tokenizer

    def _nli_label_indices(model):
        """
        Robustly find indices for 'contradiction', 'neutral', 'entailment'.
        Falls back to (0,1,2) if labels are not provided.
        """
        id2label = getattr(model.config, "id2label", None)
        if isinstance(id2label, dict) and len(id2label) >= 3:
            labels = {int(k): v.lower() for k, v in id2label.items()}
            inv = {v: k for k, v in labels.items()}
            contr = inv.get("contradiction", 0)
            neutral = inv.get("neutral", 1)
            entail = inv.get("entailment", 2)
        else:
            contr, neutral, entail = 0, 1, 2
        return contr, neutral, entail

    def _split_sentences(text: str):
        """
        Lightweight English sentence splitter; filters out very short fragments.
        """
        if not text:
            return []
        parts = re.split(r'(?<=[.!?])\s+', text.strip())
        return [p.strip() for p in parts if len(p.split()) >= 3]

    def _is_suggestion_sentence(sent: str) -> bool:
        """
        Heuristic filter to detect sentences that are likely to be advice/suggestions.
        Covers imperative mood, modal verbs, and recommendation phrases.
        """
        s = sent.lower().strip()
        if not s or len(s.split()) < 3:
            return False

        # Modal verbs (common in suggestions)
        if any(m in s for m in ["you could", "you can", "you might", "you may", "you should", "you'd better"]):
            return True

        # Recommendation markers
        if any(p in s for p in ["i suggest", "i recommend", "consider", "it's a good idea", "let's"]):
            return True

        # Imperative detection: starts with verb (naive heuristic)
        tokens = re.findall(r"[a-z']+", s)
        if tokens:
            first = tokens[0]
            imperative_starters = [
                "try", "take", "do", "make", "write", "spend", "practice",
                "focus", "remember", "consider", "start", "stop"
            ]
            if first in imperative_starters:
                return True

        return False


    # @torch.no_grad()
    # def constraint_consistency_nli_reward(completions, batch_size: int = 8, max_pairs_per_completion: int = 6, **kwargs):
    #     """
    #     NLI-based constraint consistency with suggestion sentence filtering.
    #     Premise: <cognitive>
    #     Hypotheses: sentences from <response> that look like suggestions.
    #     Reward = mean(entailment - contradiction) * 2, clipped to [-2, 2].
    #     """
    #     model, tokenizer = get_nli_model()
    #     contr_idx, neutral_idx, entail_idx = _nli_label_indices(model)

    #     rewards = []
    #     for completion in completions:
    #         content_string = completion[0]["content"]
    #         _, cognitive, _, response = extract_content(content_string)
    #         premise = (cognitive or "").strip()
    #         resp = (response or "").strip()

    #         if not premise or not resp:
    #             rewards.append(0.0)
    #             continue

    #         # Sentence split
    #         sentences = _split_sentences(resp)
    #         # Filter to suggestion-like sentences
    #         hyps = [s for s in sentences if _is_suggestion_sentence(s)]
    #         if not hyps:
    #             rewards.append(0.0)
    #             continue

    #         hyps = hyps[:max_pairs_per_completion]

    #         probs_list = []
    #         for i in range(0, len(hyps), batch_size):
    #             batch = hyps[i:i+batch_size]
    #             enc = tokenizer(
    #                 [premise] * len(batch),
    #                 batch,
    #                 return_tensors="pt",
    #                 truncation=True,
    #                 padding=True,
    #                 max_length=384
    #             ).to(_nli_device)

    #             logits = model(**enc).logits
    #             probs = torch.softmax(logits, dim=-1).detach().cpu()
    #             probs_list.extend(probs)

    #         if not probs_list:
    #             rewards.append(0.0)
    #             continue

    #         entail_scores = [float(p[entail_idx]) for p in probs_list]
    #         contr_scores  = [float(p[contr_idx])  for p in probs_list]

    #         score = sum(e - c for e, c in zip(entail_scores, contr_scores)) / len(entail_scores)
    #         # reward = max(-2.0, min(2.0, 2.0 * score))  # scale to [-2,2]
    #         raw_reward = max(-2.0, min(2.0, 2.0 * score))
    #         rewards.append(raw_reward / 2.0)
    #         # rewards.append(reward)

    #     return rewards


    @torch.no_grad()
    def constraint_consistency_nli_reward(completions, batch_size: int = 8, max_pairs_per_completion: int = 4, **kwargs):
        model, tokenizer = get_nli_model()
        contr_idx, neutral_idx, entail_idx = _nli_label_indices(model)

        rewards = []
        for completion in completions:
            content_string = completion[0]["content"]
            _, cognitive, _, response = extract_content(content_string)
            premise = (cognitive or "").strip()
            resp = (response or "").strip()

            # 如果无法提取内容，直接给负分
            if not premise or not resp:
                rewards.append(-0.5) 
                continue

            sentences = _split_sentences(resp)
            
            # === 修改点：放宽判定逻辑 ===
            # 只要包含情态动词或祈使句特征，就认为是建议
            hyps = [s for s in sentences if _is_suggestion_sentence(s)]
            
            # === 修改点：如果完全没有建议，给予惩罚 ===
            if not hyps:
                rewards.append(-0.2) # 强迫模型尝试给出建议
                continue

            hyps = hyps[:max_pairs_per_completion]

            # (后续推理代码保持不变...)
            probs_list = []
            for i in range(0, len(hyps), batch_size):
                # ... (保持原有的 batch inference 代码) ...
                batch = hyps[i:i+batch_size]
                enc = tokenizer([premise] * len(batch), batch, return_tensors="pt", truncation=True, padding=True, max_length=256).to(_nli_device)
                logits = model(**enc).logits
                probs = torch.softmax(logits, dim=-1).detach().cpu()
                probs_list.extend(probs)

            if not probs_list:
                rewards.append(0.0)
                continue

            entail_scores = [float(p[entail_idx]) for p in probs_list]
            contr_scores  = [float(p[contr_idx])  for p in probs_list]

            # 计算平均分
            avg_score = sum(e - c for e, c in zip(entail_scores, contr_scores)) / len(entail_scores)
            
            # 放大梯度：因为 entailment 通常很难达到 1.0，我们需要放大细微的差距
            rewards.append(max(-2.0, min(2.0, avg_score * 2.0))) 

        return rewards


    class GRPO_Config:
        # BASE_MODEL = "/data/wjq/code/new_les/Qwen2.5-7B-unsloth-bnb-4bit"
        BASE_MODEL = "/data/wjq/code/new_les/GRPO_qwen_2.5/outputs_sft_qwen2.5-7b_tmp/Qwen2.5-7B-sft-merged"
        # SFT_ADAPTERS_PATH = "/data/wjq/code/new_les/GRPO_qwen_2.5/outputs_sft_qwen2.5-7b/lora_adapters"
        # SFT_ADAPTERS_PATH = "/data/wjq/code/new_les/GRPO_qwen_2.5/outputs_sft_qwen2.5-7b_tmp/best-45000"
        # GRPO_DATA_PATH = "/data/wjq/code/new_les/processed_data_with_ids_test_valid.csv"
        GRPO_DATA_PATH = "/data/wjq/code/new_les/processed_data_with_ids_train.csv"
        REWARD_MODEL_TOKENIZER_PATH = "/data/wjq/datasets/MEDIC/reward_model/train_checkpoints/checkpoint-2952"
        REWARD_MODEL_PATH = "/data/wjq/datasets/MEDIC/reward_model/train_checkpoints/checkpoint-2952/model.safetensors"
        OUTPUT_DIR = "./outputs_grpo_t"
        LORA_SAVE_PATH = "./outputs_grpo_t/lora_adapters"

        # Unsloth & LoRA 配置
        MAX_SEQ_LENGTH = 2048
        LORA_RANK = 32
        GPU_MEMORY_UTILIZATION = 0.7

        # GRPO 训练超参数
        MAX_STEPS = 3000
        SAVE_STEPS = 500
        SAVE_TOTAL_LIMIT = 100
        MAX_PROMPT_LENGTH = 512
        
        # W&B 配置
        WANDB_PROJECT = "75-GRPO-t"
        WANDB_RUN_NAME = "GRPO"

        RESPONSE_LEN_REWARD_THRESHOLD = (10, 30)
        RESPONSE_TOO_SHORT_THRESHOLD = 6

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = GRPO_Config.BASE_MODEL,
        max_seq_length = GRPO_Config.MAX_SEQ_LENGTH,
        load_in_4bit = True,
        dtype = torch.bfloat16,
        # fast_inference = True, # 如果取消注释会报错
        gpu_memory_utilization = GRPO_Config.GPU_MEMORY_UTILIZATION, # Reduce if out of memory
    )
    # model = PeftModel.from_pretrained(
    #     model,                
    #     GRPO_Config.SFT_ADAPTERS_PATH,         # SFT 阶段保存的 LoRA 目录（含 peft_config.json / adapter_model.safetensors）
    #     is_trainable = True,  
    # )
    model = FastLanguageModel.get_peft_model(
        model,
        r = GRPO_Config.LORA_RANK, # e.g. 32 or 64
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha = GRPO_Config.LORA_RANK * 2,
        lora_dropout = 0.05,  # RL 建议加一点 dropout
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
    )
    
    # 3. 打印检查
    model.print_trainable_parameters()

    mark_only_lora_as_trainable_compat(model)
    for n, p in model.named_parameters():
        p.requires_grad = ("lora" in n.lower())
    print("[CHECK] active_adapter =", getattr(model, "active_adapter", "default"))
    print("[CHECK] adapters =", list(getattr(model, "peft_config", {}).keys()))
    cnt  = sum(1 for n,p in model.named_parameters() if "lora" in n.lower() and p.requires_grad)
    elems = sum(p.numel() for n,p in model.named_parameters() if "lora" in n.lower() and p.requires_grad)
    print(f"[CHECK-before-trainer] trainable lora tensors = {cnt}, elems = {elems}")
    assert cnt > 0, "LoRA 未处于可训练状态；请检查路径/是否被 merge 过。"

    model.print_trainable_parameters()


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


    def extract_content(completion):
        """
        Robust parser: works for both JSON-style ("emotion": "...", "response": "...") 
        and plain key-value (emotion: ..., response: ...) outputs.
        """
        # Affective & Cognitive sections
        affective_pattern = re.compile(r"<affective>(.*?)</affective>", re.DOTALL)
        cognitive_pattern = re.compile(r"<cognitive>(.*?)</cognitive>", re.DOTALL)
        affective = affective_pattern.search(completion)
        cognitive = cognitive_pattern.search(completion)

        # Emotion
        emotion_pattern = re.compile(r'["\']?emotion["\']?\s*:\s*["\']?([a-z_]+)["\']?', re.IGNORECASE)
        emo_match = emotion_pattern.search(completion)
        emotion = emo_match.group(1).lower() if emo_match else ""

        # Response
        response_pattern = re.compile(r'["\']?response["\']?\s*:\s*(?:"([^"]+)"|(.*))', re.IGNORECASE)
        resp_match = response_pattern.search(completion)
        response = resp_match.group(1) if resp_match and resp_match.group(1) else (resp_match.group(2) if resp_match else "")

        return affective.group(1).strip() if affective else "", \
            cognitive.group(1).strip() if cognitive else "", \
            emotion, \
            response.strip()

    full_dataset = load_dataset("csv", data_files=GRPO_Config.GRPO_DATA_PATH)["train"]
    split_dataset = full_dataset.train_test_split(test_size=0.001, seed=3407, shuffle=False)
    train_dataset = split_dataset["train"]
    test_dataset = split_dataset["test"]

    def add_dialogue_flags(example, idx, dataset):
        # 检查下一条记录的 dialogue_id 是否与当前相同
        # 如果是最后一条记录，或者下一条记录的id不同，则当前是最后一句
        is_last = (idx == len(dataset) - 1) or (dataset[idx+1]['dialogue_id'] != example['dialogue_id'])
        return {"is_last_turn": is_last}
    # print("训练集第一条数据:", train_dataset[0])
    # 假设 train_dataset 和 test_dataset 已经切分好并且有了 dialogue_id
    # 需要先对 dataset 按照 dialogue_id 和 turn_id 排序！
    train_dataset = train_dataset.sort(["dialogue_id", "turn_id"])
    test_dataset = test_dataset.sort(["dialogue_id", "turn_id"])
    train_dataset = train_dataset.map(add_dialogue_flags, with_indices=True, fn_kwargs={"dataset": train_dataset})
    test_dataset = test_dataset.map(add_dialogue_flags, with_indices=True, fn_kwargs={"dataset": test_dataset})

    def _sanitize_content(s: str) -> str:
        s = str(s).replace("\r", " ").replace("\t", " ")
        s = re.sub(r"[ \u00a0]{2,}", " ", s)
        # 可按需再加黑名单片段
        blacklist = [
            "solving this captcha will let me request more information",
            "kubectl.exe", "invalid characters", "length surpasses permitted restrictions",
        ]
        low = s.lower()
        for pat in blacklist:
            if pat in low:
                s = s.replace(pat, "")
        s = re.sub(r"\bassistant\b\s*$", "", s, flags=re.IGNORECASE)
        return s.strip()

    def _parse_history(dialogue_history_raw: str):
        messages = []
        turns = (dialogue_history_raw or "").split(" | ")
        for turn in turns:
            role_match = re.search(r"role:\s*([a-z]+)", turn, flags=re.I)
            content_match = re.search(r"content:\s*(.*)", turn, flags=re.S)
            if role_match and content_match:
                role = role_match.group(1).lower().strip()
                if role not in ("system", "user", "assistant"):
                    role = "user"
                content_raw = content_match.group(1)
                content_cleaned = re.sub(r",?\s*dialogue_emotion:.*", "", content_raw).strip().strip("'").strip(".").strip()
                content_cleaned = _sanitize_content(content_cleaned)
                messages.append({"role": role, "content": content_cleaned})
        return messages


    def _extract_resp_text(resp: str) -> str:
        # 与 SFT 完全一致的响应提取逻辑
        return re.sub(r"(?is)^.*?response:\s*", "", str(resp)).strip()

    def format_grpo_dataset(x):
        messages = [{"role": "system", "content": system_prompt}]
        messages += _parse_history(x.get("dialogue_history", ""))

        resp_text = _extract_resp_text(x.get("response", ""))
        emo = str(x.get("emotion", "")).strip().lower()     # 与 SFT 保持同样的小写处理
        thoughts = (x.get("generated_solution") or "").strip()

        reference_answer = (
            f"{thoughts}\n"
            f"{solution_start}\n"
            f"emotion: {emo}\n"
            f"response: {resp_text}\n"
            f"{solution_end}"
        )
        return {
            "prompt": messages,
            "answer": reference_answer,
            "gt_emotion": emo, "gt_response": resp_text,
            "emotion": emo,     "response": resp_text,   # 增加别名，奖励函数能拿到
            "dialogue_id": x.get("dialogue_id"),
            "is_last_turn": x.get("is_last_turn", True),
        }

    train_dataset = train_dataset.map(format_grpo_dataset)
    test_dataset = test_dataset.map(format_grpo_dataset)


    # def combined_format_reward(completions, **kwargs):
    #     """
    #     版本4：评分对象是已经被清洗过的文本。
    #     - 核心是奖励“成功生成了停止符”这一行为。
    #     """
    #     # 最终的、严格的格式检查
    #     full_pattern = re.compile(
    #         r"^\s*<affective>[\s\S]+?</affective>\s*<cognitive>[\s\S]+?</cognitive>\s*<solution>[\s\S]+?</solution>\s*$",
    #         re.IGNORECASE | re.DOTALL
    #     )
        
    #     scores = []
    #     for completion in completions:
    #         # 这里的 content_string 已经是被 RewardFunctionWrapper 清洗过的
    #         content_string = completion[0]["content"].strip()
            
    #         # 核心逻辑：这个（可能被截断的）输出是否是完美的？
    #         if full_pattern.search(content_string):
    #             # 1. 完美情况：结构完整，且结尾就是 </solution>。给予最高分。
    #             score = 1.0
    #         elif "</solution>" in content_string:
    #             # 2. 部分正确：结构不完全对，但至少成功生成了停止符（成功刹车）。
    #             #    这说明它走在正确的道路上，给予一个小的正分以示鼓励。
    #             score = 0.2
    #         else:
    #             # 3. 错误情况：连 </solution> 都没生成出来（很可能是因为内容太长被 max_tokens 截断）。
    #             #    给予明确的惩罚。
    #             score = -1.0

    #         scores.append(score)
            
    #     return scores

    def combined_format_reward(completions, **kwargs):
        """
        Modified Format Reward: Granular scoring to guide the model step-by-step.
        """
        scores = []
        # 定义关键标签
        tags = ["<affective>", "</affective>", "<cognitive>", "</cognitive>", "<solution>", "emotion:", "response:", "</solution>"]
        
        for completion in completions:
            content = completion[0]["content"]
            score = 0.0
            
            # 1. 完整性检查 (最高优先级)
            # 使用 loose pattern，允许中间有换行符等
            full_pattern = re.compile(
                r"<affective>.*?</affective>\s*<cognitive>.*?</cognitive>\s*<solution>\s*emotion:.*?\n\s*response:.*?</solution>",
                re.DOTALL | re.IGNORECASE
            )
            
            if full_pattern.search(content):
                scores.append(1.0) # 完美格式直接满分
                continue
                
            # 2. 细粒度打分 (如果没有完美，则按标签给分)
            # 每个标签给 0.1 分
            for tag in tags:
                if tag in content:
                    score += 0.1
            
            # 3. 关键结构惩罚
            # 如果没有 response: 即使有标签也没用，因为提取不出内容
            if "response:" not in content:
                score -= 0.5
            
            # 4. 长度惩罚 (防止截断导致的格式错误)
            if "</solution>" not in content:
                score -= 0.5 # 严厉惩罚没有结尾符，因为这通常意味着生成失控
            
            # 归一化到 [-1, 1] 之间
            final_score = max(-1.0, min(1.0, score))
            scores.append(final_score)
            
        return scores


    def empathetic_style_reward(completions, emotion=None, gt_emotion=None, **kwargs):
        """
        Empathetic Style Reward V2: Dense Keyword & Pattern Matching
        
        改进点：
        1. 从“整句匹配”改为“关键词匹配”，极大降低模型得分门槛。
        2. 引入 UNIVERSAL_EMPATHY_WORDS，给模型提供极易获得的“保底奖励”。
        3. 移除了复杂的条件惩罚，改为纯正向激励，鼓励模型大胆使用共情词汇。
        """
        if not isinstance(emotion, list):
            # 兼容处理：如果 emotion 没传进来，尝试用 gt_emotion
            if isinstance(gt_emotion, list):
                emotion = gt_emotion
            else:
                return [0.0] * len(completions)

        # === 1. 定义通用共情词 (万能药) ===
        # 只要出现了这些词，说明模型在试图表达共情，立刻给分
        UNIVERSAL_EMPATHY_WORDS = {
            "understand", "hear", "see", "imagine", "feel", "sound", "sounds",
            "sorry", "apologize", "valid", "hard", "tough", "difficult", 
            "support", "here", "help", "care", "worry", "glad", "happy",
            "proud", "achievement", "congratulations", "safe", "together"
        }

        # === 2. 将原本的长句字典转化为 关键词字典 ===
        # 我们可以复用你外部定义的 STYLE_PHRASES_BY_EMOTION，但只提取核心词
        # 为了运行效率，这里直接定义一个简化的核心词映射
        EMOTION_KEYWORDS = {
            "joyful": ["wonderful", "fantastic", "great", "amazing", "happy", "achievement", "congrats"],
            "excited": ["exciting", "wait", "awesome", "thrilled"],
            "proud": ["proud", "achievement", "well done", "impressive"],
            "confident": ["believe", "got this", "spirit", "trust"],
            "grateful": ["welcome", "pleasure", "glad"],
            "hopeful": ["hope", "fingers crossed", "promising", "best"],
            "caring": ["care", "thinking", "here for you"],
            "encouraging": ["can do", "keep going", "give up", "step"],
            "impressed": ["wow", "impressed", "amazing"],
            "content": ["glad", "peaceful", "lovely", "nice"],
            
            "anxious": ["breath", "okay", "pressure", "worry", "fine"],
            "apprehensive": ["understand", "slow", "concerns", "uncertainty"],
            "afraid": ["scary", "here", "safe", "frightening"],
            "terrified": ["terrifying", "safe", "together"],
            "angry": ["frustrating", "upset", "valid", "mad", "anger"],
            "furious": ["right", "furious", "acceptable", "understand"],
            "annoyed": ["annoying", "bother", "frustrating"],
            "disgusted": ["awful", "gross", "understand"],
            "ashamed": ["mistakes", "kind", "learning", "happen"],
            "embarrassed": ["tough", "everyone", "embarrassing"],
            "guilty": ["forgive", "okay", "fault", "blame"],
            "sad": ["sorry", "difficult", "hard", "sad", "heartbreaking"],
            "disappointed": ["disappointing", "sorry", "bummer"],
            "devastated": ["imagine", "hard", "support", "terrible"],
            "lonely": ["alone", "here", "listen", "company"],
            
            "neutral": ["see", "understand", "sense", "hear"],
        }

        scores = []

        for i, completion in enumerate(completions):
            content_string = completion[0].get("content", "")
            try:
                # 尝试只提取 response 部分，避免把 reasoning 里的词算进去
                _, _, _, response_text = extract_content(content_string)
                if not response_text: 
                    response_text = content_string
            except:
                response_text = content_string

            # 预处理：转小写，去标点 (简单处理)
            text_lower = response_text.lower()
            
            # --- 计分逻辑 ---
            score = 0.0
            
            # 1. 通用共情词奖励 (Base Empathy)
            # 只要包含通用词，每个词 +0.1，封顶 0.5
            # 这让模型很容易拿到基础分
            uni_hits = 0
            for w in UNIVERSAL_EMPATHY_WORDS:
                if w in text_lower:
                    uni_hits += 1
            score += min(0.5, uni_hits * 0.1)

            # 2. 情感特定词奖励 (Specific Empathy)
            # 如果用对了对应情感的词，奖励更高
            gt = (emotion[i] or "neutral").lower()
            # 模糊匹配：如果 gt 不在字典里，尝试归类（这里简化为 fallback 到 neutral）
            target_keywords = EMOTION_KEYWORDS.get(gt, EMOTION_KEYWORDS["neutral"])
            
            spec_hits = 0
            for w in target_keywords:
                if w in text_lower:
                    spec_hits += 1
            # 特定词权重更高 +0.3，封顶 1.2
            score += min(1.2, spec_hits * 0.3)

            # 3. 句式结构奖励 (Pattern Bonus)
            # 奖励 "I ... you" 的句式 (例如 "I hear you", "I support you")
            # 这是一个非常强的共情信号
            if "i " in text_lower and "you" in text_lower:
                score += 0.3

            # 4. 长度惩罚 (Sanity Check)
            # 防止模型为了堆砌关键词写太长
            # 如果超过 60 个词，稍微扣一点分
            if len(text_lower.split()) > 60:
                score -= 0.2

            # === 最终处理 ===
            # 不需要太严格的归一化，让分数主要分布在 0.0 ~ 2.0 之间即可
            # 之前的归一化除以 3.0 太狠了，导致梯度太小
            scores.append(score)

        return scores


    emo_category = [["joyful","proud","confident","grateful","hopeful","caring","encouraging","impressed","content","sentimental"],
                    ["anxious","apprehensive","afraid","terrified"],
                    ["angry","furious","annoyed"],
                    ["disgusted"],
                    ["ashamed","embarrassed","guilty","sad","disappointed","devastated","lonely","nostalgic"],
                    ["ashamed", "embarrassed", "guilty"],
                    ["excited"],
                    ["neutral","acknowledging","prepared","questioning","anticipating"],
                    ]

    group_emo = {
        "happy": ["joyful","proud","confident","grateful","hopeful","caring","encouraging","impressed","content","sentimental"],
        "fear": ["anxious","apprehensive","afraid","terrified"],
        "angry": ["angry","furious","annoyed"],
        "disgusted": ["disgusted"],
        "sad": ["ashamed","embarrassed","guilty","sad","disappointed","devastated","lonely","nostalgic"],
        "surprised": ["excited"],
        "neutral": ["neutral","acknowledging","prepared","questioning","anticipating"]
        }
    emo_to_group = {}
    for group_idx, group in enumerate(emo_category):
        for emo in group:
            emo_to_group[emo] = group_idx
    def in_same_group(a, b):
        return emo_to_group.get(a) == emo_to_group.get(b)
    
    emo_vocab = [
        "acknowledging","afraid","agreeing","angry","annoyed","anticipating","anxious","apprehensive",
        "ashamed","caring","confident","consoling","content","devastated","disappointed","disgusted",
        "embarrassed","encouraging","excited","faithful","furious","grateful","guilty","hopeful",
        "impressed","jealous","joyful","lonely","neutral","nostalgic","prepared","proud","questioning",
        "sad","sentimental","suggesting","surprised","sympathizing","terrified","trusting","wishing",
    ]
    label2id = {label: i for i, label in enumerate(emo_vocab)}
    id2label = {i: label for i, label in enumerate(emo_vocab)}
    local_model_path = "/data/wjq/code/new_les/GRPO_qwen_2.5/emotion_critic_roberta_base"
    emotion_critic_tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    emotion_critic_device = torch.device("cpu")
    emotion_critic_model = AutoModelForSequenceClassification.from_pretrained(local_model_path).to(emotion_critic_device)
    # emotion_critic_model = AutoModelForSequenceClassification.from_pretrained(local_model_path).to("cuda")
    emotion_critic_model.eval()


    # def emotion_model_reward(completions, dialogue_history, emotion, **kwargs):

    #     predicted_emotions_by_llm = [extract_content(c[0]["content"])[2] for c in completions]
        
    #     inputs = emotion_critic_tokenizer(
    #         dialogue_history, 
    #         return_tensors="pt", 
    #         padding=True, 
    #         truncation=True, 
    #         max_length=384
    #         ).to(emotion_critic_device)
        
    #     with torch.no_grad():
    #         logits = emotion_critic_model(**inputs).logits
    #         all_probs = torch.softmax(logits, dim=1) # 形状: [batch_size, num_emotion_classes]

    #     scores = []
    #     for i, llm_predicted_emotion in enumerate(predicted_emotions_by_llm):
            
    #         # --- 1. 合理性奖励 ---
    #         # 评论家认为“LLM预测的情感”有多大的可能性是正确的
    #         plausibility_reward = 0.0
    #         llm_predicted_emotion_id = label2id.get(llm_predicted_emotion)
    #         if llm_predicted_emotion_id is not None:
    #             plausibility_reward = all_probs[i][llm_predicted_emotion_id].item()

    #         # --- 2. 知识蒸馏奖励 ---
    #         # 评论家认为“数据集中的真实情感”有多大的可能性是正确的
    #         # 这部分奖励会持续地将评论家模型的知识教给LLM
    #         knowledge_distillation_reward = 0.0
    #         ground_truth_emotion = emotion[i]
    #         ground_truth_emotion_id = label2id.get(ground_truth_emotion)
    #         if ground_truth_emotion_id is not None:
    #             knowledge_distillation_reward = all_probs[i][ground_truth_emotion_id].item()
            
    #         # --- 3. 组合奖励 ---
    #         # 我们更希望模型学习到真实答案，所以给知识蒸馏部分更高的权重
    #         # 乘以一个系数来放大信号，使其范围与之前大致相当
    #         score = (0.3 * plausibility_reward + 0.7 * knowledge_distillation_reward)
            
    #         # 仍然可以保留一个小的额外奖励给完全正确的预测，以增强信号
    #         # if llm_predicted_emotion == ground_truth_emotion:
    #             # score += 2.0
    #         # else:
    #             # score -= 2.0

    #         scores.append(score)
                
    #     return scores

    # def emotion_model_reward(completions, dialogue_history, emotion, **kwargs):
    #     # 提取 LLM 预测的情感
    #     predicted_emotions_by_llm = [extract_content(c[0]["content"])[2] for c in completions]
        
    #     # 运行 Critic 模型 (保持不变)
    #     inputs = emotion_critic_tokenizer(
    #         dialogue_history, 
    #         return_tensors="pt", 
    #         padding=True, 
    #         truncation=True, 
    #         max_length=384
    #         ).to(emotion_critic_device)
        
    #     with torch.no_grad():
    #         logits = emotion_critic_model(**inputs).logits
    #         all_probs = torch.softmax(logits, dim=1) 

    #     scores = []
    #     for i, llm_predicted_emotion in enumerate(predicted_emotions_by_llm):
    #         # 标准化 LLM 输出
    #         pred_emo = llm_predicted_emotion.strip().lower()
    #         gt_emo = emotion[i].strip().lower()
            
    #         # 初始化分数
    #         score = 0.0
            
    #         # --- 1. Hard Match Reward (最重要的部分) ---
    #         # 如果 LLM 猜对了 GT，直接给大大的奖励
    #         if pred_emo == gt_emo:
    #             score += 1.0
    #         else:
    #             # 猜错了，稍微给点惩罚，或者就是 0
    #             # 我们可以稍微宽容一点，如果虽然不是 GT，但在同一个情绪大类里(例如 sad vs disappointed)，也可以给一点分
    #             # 这里先用简单版本：
    #             pass 

    #         # --- 2. Plausibility Reward (辅助信号) ---
    #         # 即使 LLM 没猜中 GT，如果它猜的词在 Critic 看来也是高概率的，也值得鼓励
    #         # 这能解决 GT 标签本身可能由于主观性而不准确的问题
    #         pred_id = label2id.get(pred_emo)
            
    #         if pred_id is not None:
    #             # 获取 Critic 对 LLM 所选标签的置信度
    #             critic_prob = all_probs[i][pred_id].item()
    #             # 将概率加入奖励 (例如 0.0 ~ 1.0)
    #             score += critic_prob
    #         else:
    #             # 如果 LLM 输出了一个不在词表里的幻觉词，重罚
    #             score -= 0.5

    #         scores.append(score)
                
    #     return scores

    # Opus 4.6
    # def emotion_model_reward(completions, dialogue_history, emotion, **kwargs):
    #     # Emotion Reward V6: 让 Critic 评估"对话历史+LLM回复"的组合
    #     # 核心思路：
    #     # 1. 把 LLM 生成的回复拼接到对话历史后面
    #     # 2. 让 Critic 评估这个"完整对话"中的情感
    #     # 3. 如果 LLM 写得好，Critic 对 GT 情感的置信度会更高

    #     gt_emotions = [e.strip().lower() for e in emotion]
    #     predicted_words = []
    #     augmented_dialogues = []

    #     # 提取预测的情感词，并构造"对话历史+回复"
    #     for i, c in enumerate(completions):
    #         try:
    #             aff, cog, pred_w, resp = extract_content(c[0]["content"])
    #             pred_w = pred_w.strip().lower().replace(".", "") if pred_w else ""

    #             # 构造增强的对话：原始对话 + LLM的回复
    #             # 这样 Critic 就能评估"加上这个回复后，情感是否更明确"
    #             llm_response = resp.strip() if resp else ""
    #             if not llm_response and aff:
    #                 llm_response = aff.strip()  # fallback 用 affective

    #             augmented_dialogue = dialogue_history[i] + " " + llm_response
    #         except:
    #             pred_w = ""
    #             augmented_dialogue = dialogue_history[i]

    #         predicted_words.append(pred_w)
    #         augmented_dialogues.append(augmented_dialogue)

    #     # === 关键改进：用"对话+回复"输入 Critic ===
    #     inputs = emotion_critic_tokenizer(
    #         augmented_dialogues,
    #         return_tensors="pt",
    #         padding=True,
    #         truncation=True,
    #         max_length=512  # 增加长度以容纳回复
    #     ).to(emotion_critic_device)

    #     with torch.no_grad():
    #         logits = emotion_critic_model(**inputs).logits
    #         all_probs = torch.softmax(logits, dim=1)

    #     scores = []
    #     for i in range(len(completions)):
    #         pred_w = predicted_words[i]
    #         gt_w = gt_emotions[i]

    #         gt_id = label2id.get(gt_w)
    #         pred_id = label2id.get(pred_w)

    #         # === 信号1：Critic 对"对话+回复"中 GT 情感的置信度 ===
    #         # 现在这个信号会随着 LLM 回复质量变化而变化
    #         base_score = 0.0
    #         if gt_id is not None:
    #             gt_prob = all_probs[i][gt_id].item()
    #             base_score = gt_prob * 5.0  # 进一步放大信号

    #         # === 信号2：精确匹配奖励 ===
    #         if pred_w == gt_w:
    #             base_score += 2.0
    #         elif pred_w and pred_id is not None:
    #             if emo_to_group.get(pred_w, -1) == emo_to_group.get(gt_w, -2):
    #                 base_score += 0.5
    #             else:
    #                 base_score -= 0.5

    #         # === 信号3：有效性检查 ===
    #         if not pred_w:
    #             base_score -= 1.0
    #         elif pred_id is None:
    #             base_score -= 1.5

    #         scores.append(max(-2.0, min(7.0, base_score)))

    #     return scores
    def emotion_model_reward(completions, dialogue_history, emotion, **kwargs):
        # Emotion Reward V9: Hybrid Reward (精确匹配 + Affective 质量)
        # 核心思路：
        # 1. 主信号：精确匹配（猜对情感词）
        # 2. 辅助信号：Affective 段落质量（基于规则）
        # 3. 知识蒸馏：Critic 对 GT 的置信度

        gt_emotions = [e.strip().lower() for e in emotion]
        predicted_words = []
        affective_texts = []
        cognitive_texts = []

        # 提取内容
        for c in completions:
            try:
                aff, cog, pred_w, _ = extract_content(c[0]["content"])
                pred_w = pred_w.strip().lower().replace(".", "") if pred_w else ""
                aff = aff.strip() if aff else ""
                cog = cog.strip() if cog else ""
            except:
                pred_w = ""
                aff = ""
                cog = ""
            predicted_words.append(pred_w)
            affective_texts.append(aff)
            cognitive_texts.append(cog)

        # Critic 推理（只看对话历史）
        inputs = emotion_critic_tokenizer(
            dialogue_history,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=384
        ).to(emotion_critic_device)

        with torch.no_grad():
            logits = emotion_critic_model(**inputs).logits
            all_probs = torch.softmax(logits, dim=1)

        scores = []
        for i in range(len(completions)):
            pred_w = predicted_words[i]
            gt_w = gt_emotions[i]
            aff = affective_texts[i]
            cog = cognitive_texts[i]

            gt_id = label2id.get(gt_w)
            pred_id = label2id.get(pred_w)

            score = 0.0

            # === 信号1：精确匹配（主要信号）===
            if pred_w == gt_w:
                score += 3.0  # 猜对给大奖励
            elif pred_w and pred_id is not None:
                # 同大类
                if emo_to_group.get(pred_w, -1) == emo_to_group.get(gt_w, -2):
                    score += 1.0
                else:
                    score -= 0.5

            # === 信号2：Affective 质量奖励（基于规则）===
            # 这是持续提升的关键
            if aff:
                aff_words = aff.split()
                aff_len = len(aff_words)

                # 2.1 长度奖励（鼓励写详细的情感分析）
                if aff_len >= 20:
                    score += 1.0
                elif aff_len >= 10:
                    score += 0.5
                elif aff_len < 5:
                    score -= 0.5

                # 2.2 情感关键词奖励
                # 检查 affective 中是否包含与 GT 情感相关的词
                gt_keywords = []
                if gt_w in ["joyful", "proud", "confident", "grateful", "hopeful", "caring", "encouraging", "impressed", "content", "sentimental"]:
                    gt_keywords = ["happy", "joy", "proud", "great", "wonderful", "positive", "glad", "pleased", "satisfied", "grateful", "hopeful", "caring", "encouraging", "impressed"]
                elif gt_w in ["anxious", "apprehensive", "afraid", "terrified"]:
                    gt_keywords = ["anxious", "worried", "afraid", "fear", "scared", "nervous", "terrified", "concern", "stress"]
                elif gt_w in ["angry", "furious", "annoyed"]:
                    gt_keywords = ["angry", "mad", "furious", "annoyed", "frustrated", "irritated", "upset"]
                elif gt_w in ["disgusted"]:
                    gt_keywords = ["disgusted", "disgust", "repulsed", "revolted"]
                elif gt_w in ["sad", "disappointed", "devastated", "lonely", "nostalgic", "ashamed", "embarrassed", "guilty"]:
                    gt_keywords = ["sad", "disappointed", "devastated", "lonely", "nostalgic", "ashamed", "embarrassed", "guilty", "sorry", "regret", "hurt", "pain"]
                elif gt_w in ["excited"]:
                    gt_keywords = ["excited", "excitement", "thrilled", "eager", "enthusiastic"]
                elif gt_w in ["neutral", "acknowledging", "prepared", "questioning", "anticipating"]:
                    gt_keywords = ["neutral", "acknowledge", "understand", "prepared", "question", "anticipate"]

                aff_lower = aff.lower()
                keyword_hits = sum(1 for kw in gt_keywords if kw in aff_lower)
                if keyword_hits >= 2:
                    score += 1.0
                elif keyword_hits == 1:
                    score += 0.5

                # 2.3 情感词出现奖励
                if pred_w and pred_w in aff_lower:
                    score += 0.5

            else:
                # 没写 affective，惩罚
                score -= 1.0

            # === 信号3：知识蒸馏（辅助信号）===
            if gt_id is not None:
                gt_prob = all_probs[i][gt_id].item()
                score += gt_prob * 1.0

            # === 信号4：有效性检查 ===
            if not pred_w:
                score -= 1.5
            elif pred_id is None:
                score -= 2.0

            scores.append(max(-3.0, min(7.0, score)))

        return scores


    # Sonnet 4.6
    # def emotion_model_reward(completions, dialogue_history, emotion, **kwargs):
    #     predicted_emotions = []
    #     affective_texts = []
    #     for c in completions:
    #         aff, _, pred_w, _ = extract_content(c[0]["content"])
    #         predicted_emotions.append(pred_w.strip().lower() or "unknown")
    #         affective_texts.append(aff.strip() if aff.strip() else "unknown")

    #     gt_emotions = [e.strip().lower() for e in emotion]

    #     # 用 affective 段落（而不是单词）输入 Critic
    #     inputs = emotion_critic_tokenizer(
    #         affective_texts,
    #         return_tensors="pt", padding=True,
    #         truncation=True, max_length=256
    #     ).to(emotion_critic_device)

    #     with torch.no_grad():
    #         logits = emotion_critic_model(**inputs).logits
    #         all_probs = torch.softmax(logits, dim=1)

    #     scores = []
    #     for i, pred_w in enumerate(predicted_emotions):
    #         gt_w = gt_emotions[i]
    #         score = 0.0

    #         gt_id = label2id.get(gt_w)
    #         pred_id = label2id.get(pred_w)

    #         # 信号1：Critic 对 GT 情感的置信度（连续梯度信号）
    #         if gt_id is not None:
    #             score += all_probs[i][gt_id].item() * 2.0

    #         # 信号2：精确匹配词
    #         if pred_w == gt_w:
    #             score += 1.0
    #         elif emo_to_group.get(pred_w) == emo_to_group.get(gt_w):
    #             score += 0.3  # 同大类

    #         # 幻觉惩罚
    #         if pred_w == "unknown" or pred_id is None:
    #             score -= 1.0

    #         scores.append(max(-1.5, min(3.0, score)))

    #     return scores
    
    # def cot_alignment_reward(completions, **kwargs):
    #     scores = []
    #     for completion in completions:
    #         content_string = completion[0]["content"]
    #         try:
    #             # 提取出模型自己的思考过程和最终结论
    #             affective_text, _, predicted_emotion, _ = extract_content(content_string)

    #             if not affective_text or not predicted_emotion:
    #                 scores.append(-1.0) # 如果结构不完整，给予惩罚
    #                 continue

    #             # 让“情感评论家”模型，根据“思考过程”来判断应该是什么情感
    #             inputs = emotion_critic_tokenizer(affective_text, return_tensors="pt", padding=True, truncation=True).to(emotion_critic_device)
    #             with torch.no_grad():
    #                 logits = emotion_critic_model(**inputs).logits
    #                 critic_predicted_id = torch.argmax(logits, dim=1).item()

    #             critic_predicted_emotion = id2label[critic_predicted_id] # 假设你有 id2label 映射

    #             # 如果模型的“最终结论”和它自己的“思考过程”得出的结论一致，就给予奖励
    #             if predicted_emotion == critic_predicted_emotion:
    #                 score = 1.0
    #             else:
    #                 score = -0.5 # 如果自相矛盾，给予惩罚

    #             scores.append(score)

    #         except Exception:
    #             scores.append(-2.0)
    #     return scores
    
    def cot_alignment_reward(completions, **kwargs):
        """
        Simple Keyword Matching Version.
        傻瓜版 CoT 奖励：不依赖 BERT 模型，仅检查“思考过程”里是否包含“预测情绪”及其同义词。
        
        优点：
        1. 极其容易学（Explicit Signal）。
        2. 计算极快（纯字符串匹配）。
        3. 避免了 Critic 模型的黑盒不确定性。
        """
        # 复用你代码里的分组，扩展同义词库
        # 确保这些变量在函数作用域内可见，或者作为全局变量
        emotion_groups = {
            "happy": ["joyful","proud","confident","grateful","hopeful","caring","encouraging","impressed","content","sentimental", "happy", "good", "great"],
            "fear": ["anxious","apprehensive","afraid","terrified", "scared", "nervous", "worry", "worried"],
            "angry": ["angry","furious","annoyed", "mad", "upset"],
            "disgusted": ["disgusted", "awful", "gross"],
            "sad": ["ashamed","embarrassed","guilty","sad","disappointed","devastated","lonely","nostalgic", "sorry", "bad", "down"],
            "surprised": ["excited", "surprised", "shocked"],
            "neutral": ["neutral","acknowledging","prepared","questioning","anticipating", "calm", "okay"]
        }

        # 建立 情绪 -> 关键词列表 的反向映射
        # e.g., "furious" -> ["angry", "furious", "annoyed", "mad", "upset"]
        # 这样只要沾边就算对
        emo_to_keywords = {}
        for group_name, words in emotion_groups.items():
            for w in words:
                emo_to_keywords[w] = words

        scores = []
        for completion in completions:
            content_string = completion[0]["content"]
            try:
                # 1. 提取
                affective_text, _, predicted_emotion, _ = extract_content(content_string)
                
                # 2. 基础检查
                if not affective_text or not predicted_emotion:
                    scores.append(-0.5) # 格式错误，轻微惩罚
                    continue
                
                # 转小写
                pred_emo = predicted_emotion.strip().lower()
                reasoning = affective_text.lower()

                # 3. 长度检查：思考过程不能太短，防止偷鸡
                if len(reasoning.split()) < 10:
                    scores.append(-0.2)
                    continue

                # 4. 核心逻辑：关键词命中
                # 如果预测词直接在思考里：+1.0
                if pred_emo in reasoning:
                    scores.append(1.0)
                    continue
                
                # 如果预测词的同义词在思考里：+0.8 (稍微低一点，鼓励精准)
                keywords = emo_to_keywords.get(pred_emo, [])
                hit = False
                for kw in keywords:
                    if kw in reasoning:
                        scores.append(0.8)
                        hit = True
                        break
                
                if hit:
                    continue

                # 5. 没命中：给予明确的负反馈，但不要太低，给它机会改
                scores.append(-0.1)

            except Exception:
                scores.append(-0.5)
        
        return scores

    # def check_response(completions, response=None, gt_response=None, **kwargs):
    #     # response 实际上是 gt_response，来自数据集
    #     if not response or not isinstance(response, (list, tuple)) or len(response) == 0:
    #         return [-1.0] * len(completions)

    #     # 初始化容器
    #     final_responses = []
    #     valid_mask = [] # True 表示有效，False 表示无效
        
    #     # --- 1. 安全的批量提取 ---
    #     for c in completions:
    #         try:
    #             # 假设 extract_content 返回 (affective, cognitive, emotion, response)
    #             # 即使格式不对，extract_content 最好也能返回空串而不是抛异常
    #             _, _, _, resp_text = extract_content(c[0]["content"])
                
    #             # 严格的过滤条件：如果是 None，或者去除空白后长度小于 2 (防止只输出一个标点)
    #             if resp_text and len(resp_text.strip()) >= 2:
    #                 final_responses.append(resp_text)
    #                 valid_mask.append(True)
    #             else:
    #                 # 无效样本：放入一个占位符，防止 SBERT 报错（虽然 SBERT 可以处理空串，但这样更保险）
    #                 # 稍后我们会强行覆盖这个分数
    #                 final_responses.append("EMPTY") 
    #                 valid_mask.append(False)
    #         except Exception:
    #             # 提取过程报错，也视为无效
    #             final_responses.append("ERROR")
    #             valid_mask.append(False)

    #     gt_responses = [r or "" for r in response]

    #     try:
    #         # --- 2. 批量编码 ---
    #         # 注意：这里我们依然编码了 "EMPTY"/"ERROR"，但这只占很少计算量
    #         # 相比于把 valid 的挑出来编码再拼回去，直接编码逻辑更简单，且保持了 batch 顺序
    #         embs1 = sbert.encode(final_responses, convert_to_tensor=True)
    #         embs2 = sbert.encode(gt_responses, convert_to_tensor=True)
            
    #         # --- 3. 批量计算余弦相似度 ---
    #         cosine_sims = util.cos_sim(embs1, embs2).diag().tolist()

    #         # --- 4. 计算最终分数 ---
    #         scores = []
    #         for i, sim in enumerate(cosine_sims):
    #             # === 核心修改：如果是无效样本，直接判死刑 ===
    #             if not valid_mask[i]:
    #                 scores.append(-2.0) # 格式错误或内容为空，直接给最低分
    #                 continue
                
    #             # 以下是有效样本的正常打分逻辑
    #             score = sim * 3.0 
                
    #             # 长度惩罚 (针对有效内容)
    #             # 使用 split() 计算单词数更符合英文习惯
    #             curr_text = final_responses[i]
    #             response_len = len(curr_text.split())
                
    #             min_len, max_len = GRPO_Config.RESPONSE_LEN_REWARD_THRESHOLD
    #             too_short_len = GRPO_Config.RESPONSE_TOO_SHORT_THRESHOLD

    #             if min_len < response_len < max_len:
    #                 score += 1.0
    #             if response_len > max_len:
    #                 score -= 0.5
    #             if response_len < too_short_len:
    #                 score -= 0.5
                
    #             # scores.append(min(score, 3.0))
    #             final_score = min(score, 3.0)
    #             scores.append(final_score / 3.0) # 归一化到 max 1.0 
                
    #         return scores

    #     except Exception as e:
    #         print(f"Error in check_response_batched sbert step: {e}")
    #         # 只有在 SBERT 计算本身崩了（显存不足等）才全员失败
    #         return [-0.5] * len(completions)
    

    # def check_response(completions, response=None, gt_response=None, **kwargs):
    #     """
    #     Check Response V4: Semantic Anchor & Expansion Reward (超越 GT 版)
        
    #     核心思想：
    #     - 不再强求逼近 GT，因为 GT 可能太短或太单薄。
    #     - 只要语义不跑题 (Similarity > Threshold)，就鼓励模型“自由发挥”。
    #     - 奖励：更长的篇幅(Expansion)、更多的交互(Questioning)、更丰富的情绪表达。
    #     """
    #     # response 实际上是 gt_response
    #     if not response or not isinstance(response, (list, tuple)) or len(response) == 0:
    #         return [-1.0] * len(completions)

    #     final_responses = []
    #     valid_mask = []
        
    #     # 1. 提取内容
    #     for c in completions:
    #         try:
    #             _, _, _, resp_text = extract_content(c[0]["content"])
    #             if resp_text and len(resp_text.strip()) >= 2:
    #                 final_responses.append(resp_text)
    #                 valid_mask.append(True)
    #             else:
    #                 final_responses.append("EMPTY") 
    #                 valid_mask.append(False)
    #         except Exception:
    #             final_responses.append("ERROR")
    #             valid_mask.append(False)

    #     gt_responses = [r or "" for r in response]

    #     try:
    #         # 2. 计算语义相似度 (SBERT)
    #         # 这依然是必要的，用来判断模型有没有“幻觉”或“答非所问”
    #         embs1 = sbert.encode(final_responses, convert_to_tensor=True)
    #         embs2 = sbert.encode(gt_responses, convert_to_tensor=True)
    #         # ReLU 截断，负相似度归零
    #         cosine_sims = torch.nn.functional.relu(util.cos_sim(embs1, embs2).diag()).tolist()

    #         scores = []
    #         for i, sim in enumerate(cosine_sims):
    #             if not valid_mask[i]:
    #                 scores.append(-2.0)
    #                 continue
                
    #             gen_text = final_responses[i]
                
    #             # === 核心修改逻辑 ===
                
    #             # A. 软性语义锚点 (Soft Semantic Anchor)
    #             # 只要相似度超过 0.35 (哪怕只有一点点相关)，我们就认为“话题对上了”。
    #             # 超过 0.35 的部分，我们只给极其微弱的奖励，意味着“不用刻意去模仿 GT”。
    #             # 如果低于 0.35，说明可能答非所问，给予线性惩罚。
    #             topic_threshold = 0.35
    #             if sim >= topic_threshold:
    #                 # 基础分直接拿 0.6，多出来的部分稍微加一点点，封顶 0.8
    #                 semantic_score = 0.6 + (sim - topic_threshold) * 0.2 
    #             else:
    #                 # 没过及格线，直接用相似度作为低分
    #                 semantic_score = sim 

    #             # B. 扩展奖励 (Expansion Bonus) - 鼓励“多说点”
    #             # 现代 LLM 的共情往往需要一定的篇幅来铺陈。
    #             # 假设 GT 平均 15 词，我们希望模型能写到 30-50 词。
    #             num_words = len(gen_text.split())
    #             # 设定一个理想长度区间 (30 ~ 60 词)，比 GT 通常要长
    #             # 使用 tanh 函数平滑奖励：在 40 词左右达到饱和
    #             length_bonus = min(1.0, num_words / 40.0) * 0.5 
                
    #             # C. 交互性奖励 (Engagement Bonus) - 鼓励“追问”
    #             # 好的共情往往以“关注对方的下一步”结尾，通常带有问号。
    #             # GT 里很多只是 "I am sorry."，我们希望模型能说 "I am sorry. Are you feeling better now?"
    #             question_bonus = 0.0
    #             if "?" in gen_text:
    #                 question_bonus = 0.3
                
    #             # D. 最终得分合成
    #             # 只有当“话题对上了”(sim > topic_threshold) 才有资格拿 B 和 C 的奖励
    #             # 否则只是为了凑字数而胡言乱语
    #             if sim >= topic_threshold:
    #                 total_score = semantic_score + length_bonus + question_bonus
    #             else:
    #                 total_score = semantic_score # 跑题了，写再长也没用

    #             # 归一化/截断：期望范围 [0.0, 1.5] -> 放大到 [0, 2.0] 甚至更高以提供强梯度
    #             # 现在的最高分可能是：0.8 (Semantic) + 0.5 (Len) + 0.3 (Que) = 1.6
    #             scores.append(total_score * 1.5)

    #         return scores

    #     except Exception as e:
    #         print(f"Error in check_response: {e}")
    #         return [-0.5] * len(completions)
    

    # 1. 在外部定义通用回复黑名单（可以根据你的数据扩充）
    GENERIC_RESPONSES = [
        "I understand how you feel.",
        "That sounds really hard.",
        "I am here for you.",
        "I am sorry to hear that.",
        "Thank you for sharing that with me.",
        "It is okay to feel that way.",
        "Take your time.",
        "Don't worry about it."
    ]
    # 预先编码，避免并在 check_response 中重复计算
    GENERIC_EMBS = sbert.encode(GENERIC_RESPONSES, convert_to_tensor=True)

    def check_response(completions, response=None, prompts=None, **kwargs):
        """
        Check Response V7: Dual-Anchor Relevance & Anti-Generic (优化版)
        """
        # 如果没有 GT，返回默认负分
        if not response or not isinstance(response, (list, tuple)) or len(response) == 0:
            return [-1.0] * len(completions)

        final_responses = []
        valid_mask = []
        
        # 1. 提取内容
        for c in completions:
            try:
                _, _, _, resp_text = extract_content(c[0]["content"])
                if resp_text and len(resp_text.strip()) >= 2:
                    final_responses.append(resp_text)
                    valid_mask.append(True)
                else:
                    final_responses.append("EMPTY") 
                    valid_mask.append(False)
            except Exception:
                final_responses.append("ERROR")
                valid_mask.append(False)

        gt_responses = [r or "" for r in response]
        
        # === 新增：提取 User Last Message ===
        # 假设 prompts 结构是 list of dicts 或 list of strings
        user_queries = []
        if prompts:
            for p in prompts:
                # 尝试从 prompt 中解析出最后一句用户的话
                # 这里的解析逻辑取决于你的 prompt 格式，这里做一个简单的假设
                try:
                    # 如果 prompt 是 list[dict]
                    if isinstance(p, list): 
                        last_user = next((m['content'] for m in reversed(p) if m['role'] == 'user'), "")
                    # 如果 prompt 是 string (渲染后的)，尝试用正则提取最后一句 User
                    elif isinstance(p, str):
                        # 这是一个简化的提取，视你的模板而定
                        parts = p.split("<|im_start|>user")
                        if len(parts) > 1:
                            last_user = parts[-1].split("<|im_end|>")[0].strip()
                        else:
                            last_user = ""
                    else:
                        last_user = str(p)
                    user_queries.append(last_user)
                except:
                    user_queries.append("")
        else:
            user_queries = [""] * len(completions)

        try:
            # 2. SBERT 计算 (Batch Processing)
            embs_gen = sbert.encode(final_responses, convert_to_tensor=True)
            embs_gt = sbert.encode(gt_responses, convert_to_tensor=True)
            embs_user = sbert.encode(user_queries, convert_to_tensor=True)

            # A. 计算与 GT 的相似度
            sim_gt = torch.nn.functional.relu(util.cos_sim(embs_gen, embs_gt).diag())
            
            # B. 计算与 User Input 的相似度 (Context Relevance)
            # 这里的逻辑是：回复必须和用户的问题有关联。
            sim_user = torch.nn.functional.relu(util.cos_sim(embs_gen, embs_user).diag())

            # C. 计算与通用回复的相似度 (Anti-Generic)
            # 计算每个生成回复与所有通用回复的最大相似度
            sim_generic = util.cos_sim(embs_gen, GENERIC_EMBS).max(dim=1).values

            scores = []
            for i in range(len(completions)):
                if not valid_mask[i]:
                    scores.append(-2.0)
                    continue
                
                s_gt = sim_gt[i].item()
                s_user = sim_user[i].item()
                s_generic = sim_generic[i].item()
                
                # === 核心打分逻辑优化 ===

                # 1. 双锚点语义分 (Dual Anchor Score)
                # 只要像 GT (复制标准答案) 或者 像 User (紧扣主题) 都可以得分
                # s_user 的阈值通常比 s_gt 低，因为回复和问题的相似度不如回复和回复高
                # 这种写法允许模型“另辟蹊径”，只要不跑题
                semantic_base = max(s_gt, s_user * 0.85) 
                
                # 2. 梯度锐化 (Gradient Sharpening)
                # 使用 Sigmoid 变体拉大 0.4-0.7 之间的差距
                # 当 sim=0.4 时 score~0.2; 当 sim=0.7 时 score~0.8
                semantic_score = (1 / (1 + math.exp(-10 * (semantic_base - 0.45)))) * 2.0 - 0.5
                
                # 3. 通用回复惩罚 (Dullness Penalty)
                # 如果回复太像套话 (s_generic > 0.65)，且语义得分不高，重罚
                # 这样逼迫模型生成更具体的内容
                penalty = 0.0
                if s_generic > 0.65 and semantic_base < 0.6:
                    penalty = -1.0 
                
                # 4. 长度与结构 (保留原有逻辑，适当简化)
                gen_text = final_responses[i]
                length_score = 0.0
                words = len(gen_text.split())
                if 20 <= words <= 80:
                    length_score = 0.5
                
                # 5. 交互性奖励 (Engagement)
                question_bonus = 0.5 if "?" in gen_text else 0.0

                # === 总分 ===
                total_score = semantic_score + penalty + length_score + question_bonus
                
                # 裁剪范围，防止梯度爆炸
                scores.append(max(-1.5, min(2.5, total_score)))

            return scores

        except Exception as e:
            print(f"Error in check_response: {e}")
            return [-0.5] * len(completions)


    # Reward Model
    class MultiLabelClassifier(nn.Module):
        def __init__(self, model_name):
            super().__init__()
            self.encoder = AutoModel.from_pretrained(model_name)
            hidden_size = self.encoder.config.hidden_size
            self.er_head = nn.Linear(hidden_size, 3)
            self.cr_head = nn.Linear(hidden_size, 3)

        def forward(self, input_ids, attention_mask, label_ER=None, label_CR=None):
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = outputs.last_hidden_state[:, 0]
            er_logits = self.er_head(pooled_output)
            cr_logits = self.cr_head(pooled_output)

            loss = None
            if label_ER is not None and label_CR is not None:
                loss_fn = nn.CrossEntropyLoss()
                loss = (loss_fn(er_logits, label_ER) + loss_fn(cr_logits, label_CR)) / 2

            return {"loss": loss, "logits": (er_logits, cr_logits)}


    class RewardModelWrapper:
        def __init__(self, model_path, tokenizer_path):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            self.model = MultiLabelClassifier("roberta-base")
            state_dict = st_load(model_path)
            self.model.load_state_dict(state_dict)
            self.device = torch.device("cpu")
            self.model.to(self.device)
            self.model.eval()
            self.er_mapping = {0: -0.5, 1: 1.0, 2: 2.0}
            self.cr_mapping = {0: -0.5, 1: 1.0, 2: 2.0}

        @torch.no_grad()
        def predict_batch(self, texts):
            inputs = self.tokenizer(
                texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True
                ).to(self.device)
            
            outputs = self.model(**inputs)
            er_logits, cr_logits = outputs["logits"]

            er_pred = torch.argmax(er_logits, dim=1)
            cr_pred = torch.argmax(cr_logits, dim=1)

            return er_pred.tolist(), cr_pred.tolist()

        

        def get_score_batch(self, texts):
            if not texts:
                return []
            
            valid_texts = []
            valid_indices = []
            for i, text in enumerate(texts):
                if text and text.strip():
                    valid_texts.append(text)
                    valid_indices.append(i)
            
            if not valid_texts:
                return [-0.5] * len(texts)

            try:
                er_preds, cr_preds = self.predict_batch(valid_texts)
            except Exception as e:
                print(f"Error in batch reward prediction: {e}")
                return [-1.0] * len(texts) # 出错时返回错误分数

            # 根据预测结果，将分数放回其原始位置
            all_scores = [-0.5] * len(texts) # 先用默认分填充
            for i, original_index in enumerate(valid_indices):
                er_pred = er_preds[i]
                cr_pred = cr_preds[i]
                er_reward = self.er_mapping.get(er_pred, -0.5)
                cr_reward = self.cr_mapping.get(cr_pred, -0.5)
                all_scores[original_index] = er_reward + cr_reward
                
            return all_scores

    reward_model_wrapper = RewardModelWrapper(GRPO_Config.REWARD_MODEL_PATH, GRPO_Config.REWARD_MODEL_TOKENIZER_PATH)


    def check_reward(completions, **kwargs):
        try:
            # 1. 使用列表推导式，一次性从所有 completions 中提取出 final_response
            final_responses = [extract_content(completion[0]["content"])[3] for completion in completions]
            
            # 2. 调用 RewardModelWrapper 中新的批处理方法，一次性获取所有分数
            scores = reward_model_wrapper.get_score_batch(final_responses)
            
            return [s / 4.0 for s in scores]
        except Exception as e:
            print(f"Error in check_reward_batched: {e}")
            return [-1.0] * len(completions)


    # class DialogueEmpathyEvaluator:
    #     """
    #     Dialogue-level reward that enforces whole-dialogue empathy.

    #     核心思想：
    #     - 维护每个对话的「逐轮 RM 分数」缓存
    #     - 目标函数 J(prefix) = softmin(prefix, tau) - λ * Var(prefix)
    #     * softmin 逼近“最差一轮”，单轮掉链子会立刻拉低目标
    #     * Var 惩罚波动，鼓励全程稳定
    #     - 每一轮给「前缀增量」：R_t = J(r_1..r_t) - J(r_1..r_{t-1})
    #     这样每一轮都有对话级信号，而不是等到最后才结算
    #     """

    #     def __init__(self, tau: float = 0.4, var_lambda: float = 0.1):
    #         self.__name__ = "dialogue_level_reward"
    #         self.dialogue_histories = {}   # {d_id: [completion_texts]}
    #         self.dialogue_scores = {}      # {d_id: [per-turn RM scores]}
    #         self.tau = tau
    #         self.var_lambda = var_lambda

    #     @staticmethod
    #     def _softmin(xs, tau):
    #         import math
    #         if not xs:
    #             return 0.0
    #         if len(xs) == 1:
    #             return xs[0]
    #         t = max(tau, 1e-6)
    #         exps = [math.exp(-x / t) for x in xs]
    #         return -t * math.log(sum(exps) / len(xs))

    #     @staticmethod
    #     def _pvar(xs):
    #         if len(xs) <= 1:
    #             return 0.0
    #         mu = sum(xs) / len(xs)
    #         return sum((x - mu) ** 2 for x in xs) / len(xs)

    #     def _objective(self, xs):
    #         # J(xs) = softmin(xs, tau) - λ * Var(xs)
    #         return self._softmin(xs, self.tau) - self.var_lambda * self._pvar(xs)

    #     def __call__(self, completions, dialogue_id, is_last_turn, **kwargs):
    #         batch_rewards = []
    #         for i in range(len(completions)):
    #             d_id = dialogue_id[i]
    #             comp_text = completions[i][0]["content"]

    #             # 建立/更新历史
    #             if d_id not in self.dialogue_histories:
    #                 self.dialogue_histories[d_id] = []
    #                 self.dialogue_scores[d_id] = []
    #             self.dialogue_histories[d_id].append(comp_text)

    #             try:
    #                 _, _, _, final_response = extract_content(comp_text)
    #                 turn_score = reward_model_wrapper.get_score(final_response)
    #             except Exception:
    #                 turn_score = -0.5  

    #             scores = self.dialogue_scores[d_id]
    #             # 计算前缀增量：J(r_1..r_t) - J(r_1..r_{t-1})
    #             prev_J = self._objective(scores) if scores else 0.0
    #             scores.append(turn_score)
    #             curr_J = self._objective(scores)
    #             prefix_delta = curr_J - prev_J

    #             batch_rewards.append(float(prefix_delta))

    #             # 收尾：最后一轮释放内存
    #             if is_last_turn[i]:
    #                 del self.dialogue_histories[d_id]
    #                 del self.dialogue_scores[d_id]

    #         return batch_rewards
    

    class DialogueEmpathyEvaluator:
        """
        Dialogue-level reward V3 (Gated Momentum).
        
        解决 "Reward Hacking" 问题：
        防止模型因为上一轮分数高，就在这一轮“摆烂”导致其他指标下降。
        
        核心逻辑变更：
        Trend Bonus不再是简单的累加，而是取 (Current, Previous) 的最小值。
        这意味着：你现在的表现决定了你能从历史中继承多少奖励。
        """

        def __init__(self, history_weight: float = 0.3): # 建议把权重从 0.5 降到 0.3
            self.__name__ = "dialogue_level_reward"
            self.dialogue_last_score = {}
            self.history_weight = history_weight
            self.print_once = True 

        def __call__(self, completions, dialogue_id, is_last_turn, **kwargs):
            batch_rewards = []
            for i in range(len(completions)):
                d_id = dialogue_id[i]
                comp_text = completions[i][0]["content"]
                current_score = -0.5 

                try:
                    _, _, _, final_response = extract_content(comp_text)
                    if not final_response or len(final_response.strip()) < 2:
                        current_score = -0.8 
                    else:
                        # 获取当前分
                        scores = reward_model_wrapper.get_score_batch([final_response])
                        if scores and len(scores) > 0:
                            current_score = scores[0]
                        else:
                            current_score = -0.5
                except Exception:
                    current_score = -0.5

                # === 核心修改逻辑 ===
                prev_score = self.dialogue_last_score.get(d_id, 0.0)
                
                # 1. 基础分：当前表现
                base_reward = current_score

                # 2. 动量分：只有当 Current 和 Prev 都为正时，才计算动量
                # 使用 min() 也就是“木桶效应”：
                # 如果上一轮很好(1.0)，这一轮很烂(0.1)，动量 = min(1.0, 0.1) = 0.1 (很小) -> 没法吃老本
                # 如果上一轮很好(1.0)，这一轮也很好(1.0)，动量 = min(1.0, 1.0) = 1.0 (很大) -> 奖励连胜
                if current_score > 0 and prev_score > 0:
                    momentum = min(current_score, prev_score)
                    bonus = self.history_weight * momentum
                else:
                    bonus = 0.0

                total_reward = base_reward + bonus

                # 更新历史
                self.dialogue_last_score[d_id] = current_score

                # 归一化/截断：防止总分超过其他奖励函数太多，导致模型只关注这个
                # 现在的范围大概是 [-0.8, 1.3]
                total_reward = max(-1.0, min(1.5, total_reward))
                
                batch_rewards.append(float(total_reward))

                if is_last_turn[i]:
                    if d_id in self.dialogue_last_score:
                        del self.dialogue_last_score[d_id]

            return batch_rewards


    dialogue_evaluator = DialogueEmpathyEvaluator()

    class PhasedRewardCalculator:
        def __init__(self, trainer_config, reward_weights):
            total_steps = trainer_config.MAX_STEPS 
            
            self.phase1_end_step = int(total_steps * 0.03)  # 前 20% 的步数专注于格式
            self.phase2_end_step = int(total_steps * 0.1)  # 接下来的 50% 加入内容和情感
            # 最后 30% 引入所有复杂奖励

            self.trainer = None
            self.reward_weights = reward_weights # <--- [修改2] 保存权重

            
            # 将所有奖励函数打包，方便调用
            self.reward_functions = {
                "combined_format_reward": combined_format_reward,
                "empathetic_style_reward": empathetic_style_reward, 
                "emotion_model_reward": emotion_model_reward,
                "check_response": check_response,
                "check_reward": check_reward,
                "constraint_consistency_nli_reward": constraint_consistency_nli_reward,
                "dialogue_level_reward": dialogue_evaluator,
                "cot_alignment_reward": cot_alignment_reward,
            }

        def set_trainer(self, trainer):
            """注入 trainer 实例以便获取 global_step"""
            self.trainer = trainer
        
        # def __call__(self, completions, **kwargs):
        #     if self.trainer is None:
        #         raise ValueError("Trainer instance has not been set. Call set_trainer(trainer) first.")
            
        #     global_step = self.trainer.state.global_step
        #     all_rewards = {}
        #     num_completions = len(completions)

        #     # --- 初始化总分列表 ---
        #     # [新增] 用来存储当前 batch 中每一条生成的加权总分
        #     batch_total_weighted_scores = [0.0] * num_completions

        #     active_rewards = []
        #     if global_step < self.phase1_end_step:
        #         active_rewards = [
        #             "combined_format_reward", 
        #             "check_response", 
        #             ]
        #     # elif global_step < self.phase2_end_step:
        #     #     active_rewards = [
        #     #         "combined_format_reward", 
        #     #         "check_response", 
        #     #         "check_reward", 
        #     #         "emotion_model_reward",
        #     #         "empathetic_style_reward"
        #     #         ]
        #     else:
        #         active_rewards = list(self.reward_functions.keys())

            
        #     if (global_step % 20 == 0) and completions:
        #         sample_txt = completions[0][0]["content"] if isinstance(completions[0], list) else str(completions[0])
        #         print("[DEBUG COMPLETION]", sample_txt)


        #     # --- 计算奖励 ---
        #     for name, func in self.reward_functions.items():
        #         if name in active_rewards:
        #             # 计算单项得分
        #             scores = func(completions=completions, **kwargs)
        #             all_rewards[name] = scores
                    
        #             # [新增] 累加到总分
        #             # 获取该项权重，默认为 1.0
        #             weight = self.reward_weights.get(name, 1.0)
        #             for i in range(len(scores)):
        #                 if i < len(batch_total_weighted_scores):
        #                     batch_total_weighted_scores[i] += scores[i] * weight
        #         else:
        #             all_rewards[name] = [0.0] * num_completions
            
        #     # --- W&B 日志记录 ---
        #     if wandb.run:
        #         log_data = {}
        #         for name, scores in all_rewards.items():
        #             mean_score = (sum(scores) / max(1, len(scores))) if scores else 0.0
        #             log_data[f"rewards/individual/{name}_mean"] = mean_score

        #         # 2. [新增] 记录加权总分 (Total Weighted Reward)
        #         # 这是你最关心的曲线！
        #         mean_total_score = sum(batch_total_weighted_scores) / max(1, len(batch_total_weighted_scores))
        #         log_data["rewards/total_weighted_reward"] = mean_total_score

        #         wandb.log(log_data)
                
        #     return all_rewards


        def __call__(self, completions, **kwargs):
            if self.trainer is None:
                raise ValueError("Trainer instance has not been set.")
            
            global_step = self.trainer.state.global_step
            all_rewards = {}
            num_completions = len(completions)
            batch_total_weighted_scores = [0.0] * num_completions

            # 1. 先计算格式奖励 (Gatekeeper)
            format_scores = self.reward_functions["combined_format_reward"](completions, **kwargs)
            all_rewards["combined_format_reward"] = format_scores
            
            # 格式权重大幅提升，确保它是首要目标
            format_weight = 3.0 
            
            # 2. 计算其他奖励，但应用门控逻辑
            for name, func in self.reward_functions.items():
                if name == "combined_format_reward":
                    continue # 已经算过了
                
                # 计算原始分数
                raw_scores = func(completions=completions, **kwargs)
                all_rewards[name] = raw_scores
                
                weight = self.reward_weights.get(name, 1.0)

                # === 门控逻辑 (Gating Logic) ===
                for i in range(num_completions):
                    # 如果格式分低于 0.8 (意味着格式有明显缺陷)，则其他奖励归零或打折
                    if format_scores[i] < 0.8:
                        # 极其严格：格式不对，只要内容分是正的，全部强制归零。
                        # 允许负分通过（惩罚叠加），但不允许正分通过（防止奖励作弊）。
                        if raw_scores[i] > 0:
                            pass # 不加分
                        else:
                            # 负分照常累加
                             batch_total_weighted_scores[i] += raw_scores[i] * weight
                    else:
                        # 格式正确，正常加分
                        batch_total_weighted_scores[i] += raw_scores[i] * weight

            # 最后加上格式分
            for i in range(num_completions):
                batch_total_weighted_scores[i] += format_scores[i] * format_weight

            # --- W&B 日志记录 ---
            if wandb.run:
                log_data = {}
                for name, scores in all_rewards.items():
                    mean_score = sum(scores) / max(1, len(scores))
                    log_data[f"rewards/individual/{name}_mean"] = mean_score
                
                mean_total_score = sum(batch_total_weighted_scores) / max(1, len(batch_total_weighted_scores))
                log_data["rewards/total_weighted_reward"] = mean_total_score
                wandb.log(log_data)
                
            return all_rewards
    


    # REWARD_WEIGHTS = {
    #     "combined_format_reward": 1.0,
    #     "emotion_model_reward": 2.5,
    #     "check_response": 1.5,
    #     "check_reward": 1.0,
    #     "empathetic_style_reward": 0.8,  
    #     "cot_alignment_reward": 1.0,
    #     "constraint_consistency_nli_reward": 0.5,
    #     "dialogue_level_reward": 0.5,
    # }

    REWARD_WEIGHTS = {
        "combined_format_reward": 3.0,  # 提高权重，配合代码中的 Gate 逻辑
        "emotion_model_reward": 1.5,    # 稍微降低，原先 2.5 太高
        "check_response": 2.0,          # 提高语义相似度权重，保证内容相关
        "check_reward": 1.0,
        "empathetic_style_reward": 0.5, # 降低 Style 权重，防止无脑堆砌
        "cot_alignment_reward": 1.0,
        "constraint_consistency_nli_reward": 1.0, # 提高逻辑权重
        "dialogue_level_reward": 0.5,
    }

    REWARD_COMPONENTS = {
        "format": ["combined_format_reward"],
        "quality": ["check_response", "check_reward"],
        "logic": ["constraint_consistency_nli_reward"],
        "empathy_context": ["emotion_model_reward", "dialogue_level_reward", "cot_alignment_reward"],
        "style_safety": ["empathetic_style_reward"], 
    }

    COMPONENT_LOSS_WEIGHTS = {
        "format": 1.0,
        "empathy_context": 2.5,
        "quality": 1.5,
        "style_safety": 1.0, 
        "logic": 0.8,
    }
    # ==================== 修改代码：调整奖励权重 END ======================

    wandb.init(
        project=GRPO_Config.WANDB_PROJECT,
        name=GRPO_Config.WANDB_RUN_NAME,
    )
    # vllm_sampling_params = SamplingParams(
    #     min_p = 0.1,
    #     top_p = 1.0,
    #     top_k = -1,
    #     seed = 3407,
    #     stop = [tokenizer.eos_token],
    #     include_stop_str_in_output = True,
    # )
    training_args = GRPOConfig(
        gradient_checkpointing = True,
        use_vllm=True, # Use vLLM for fast inference
        vllm_mode="server",
        vllm_server_base_url="http://127.0.0.1:8002",
        vllm_server_timeout=120.0,
        # vllm_guided_decoding_regex=_DEFAULT_REGEX,
        generation_kwargs={
            "temperature": 0.9,
            "min_p": 0.1,
            "top_p": 0.95,
            "top_k": -1,
            "stop": ["</solution>", tokenizer.eos_token],             
            "include_stop_str_in_output": True,
            "seed": 3407,
        },
        learning_rate = 5e-6,
        weight_decay = 0.01,
        # warmup_ratio = 0.1,
        warmup_steps = 100,
        beta = 0.001,
        lr_scheduler_type = "cosine",
        optim = "adamw_8bit",
        logging_steps = 1,
        bf16 = is_bfloat16_supported(),
        fp16 = not is_bfloat16_supported(),
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 16,
        num_generations = 16,
        # max_prompt_length = GRPO_Config.MAX_PROMPT_LENGTH,
        # max_completion_length = max_completion_length,
        max_completion_length = 384,
        # num_train_epochs = 1, # Set to 1 for a full training run
        max_steps = GRPO_Config.MAX_STEPS,
        save_steps = GRPO_Config.SAVE_STEPS,
        save_total_limit = GRPO_Config.SAVE_TOTAL_LIMIT,
        max_grad_norm=1.0,
        report_to = "wandb", # Can use Weights & Biases
        # report_to = "none", # Can use Weights & Biases
        run_name = GRPO_Config.WANDB_RUN_NAME,
        output_dir = GRPO_Config.OUTPUT_DIR,

        # For optional training + evaluation
        # fp16_full_eval = True,
        # per_device_eval_batch_size = 8,
        # eval_accumulation_steps = 1,
        # eval_strategy = "steps",
        # eval_strategy = "epoch",
        # eval_steps = GRPO_Config.EVAL_STEPS,

        # 传入 CRPO 配置
        reward_components=REWARD_COMPONENTS,
        reward_weights=REWARD_WEIGHTS, # 这个参数原本就支持
        component_loss_weights=COMPONENT_LOSS_WEIGHTS,
    )

    phased_reward_calculator = PhasedRewardCalculator(GRPO_Config, REWARD_WEIGHTS)

    import torch.nn as nn
    try:
        from peft.tuners.lora.layer import LoraLayer
    except Exception:
        LoraLayer = None

    import math, requests
    from typing import List, Any
    from trl.extras.vllm_client import VLLMClient
    from transformers import PreTrainedTokenizerBase

    def _mp_to_chat_messages(x):
        # A) 如果是 已渲染的完整 prompt 文本，原样返回字符串
        if isinstance(x, str) and ("<|im_start|>" in x or "<|im_end|>" in x):
            return x

        # B) 正常的 messages(list[dict]) 路径
        if isinstance(x, list) and x and isinstance(x[0], dict) and "role" in x[0] and "content" in x[0]:
            msgs = []
            for m in x:
                role = str(m.get("role", "user")).lower().strip()
                if role not in ("system", "user", "assistant"):
                    role = "user"
                content = m.get("content", "")
                try:
                    content = _sanitize_content(content)
                except Exception:
                    pass
                if isinstance(content, list):
                    parts = []
                    for c in content:
                        if isinstance(c, dict) and "text" in c: parts.append(str(c["text"]))
                        else: parts.append(str(c))
                    content = " ".join(parts)
                elif not isinstance(content, str):
                    content = str(content)
                msgs.append({"role": role, "content": content})
            return msgs

        # C) 其他类型→包成单条 user 消息
        return [{"role": "user", "content": str(x)}]


    def _safe_float(x, default):
        try:
            v = float(x); 
            return default if math.isnan(v) or math.isinf(v) else v
        except Exception:
            return default

    def _extract_arg(args, kwargs, name, idx=None, default=None):
        if name in kwargs: return kwargs.pop(name)
        if idx is not None and len(args) > idx: return args[idx]
        return default


    def _normalize_messages_for_generation(messages):
        # A) 已渲染 prompt：直接原样返回字符串
        if isinstance(messages, str) and ("<|im_start|>" in messages or "<|im_end|>" in messages):
            return messages

        # B) 普通字符串：包成单条 user 消息
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        # C) 若不是列表，尽量转成单条 user
        if not isinstance(messages, list):
            messages = [{"role": "user", "content": str(messages)}]

        # D) 规范化为 list[dict]
        norm = []
        for m in messages:
            # 容错：允许 m 是 dict / tuple / 其他
            if isinstance(m, dict):
                role = str(m.get("role", "user")).lower().strip()
                content = m.get("content", "")
            elif isinstance(m, (list, tuple)) and len(m) >= 2:
                role = str(m[0]).lower().strip()
                content = m[1]
            else:
                role = "user"
                content = str(m)

            if role not in ("system", "user", "assistant"):
                role = "user"

            content = "" if content is None else str(content)
            if content.strip() == "":
                continue
            norm.append({"role": role, "content": content})

        if not norm:
            norm = [{"role": "user", "content": "."}]

        # E) system 放最前（只留第一个）
        sys_idx = next((i for i, x in enumerate(norm) if x["role"] == "system"), None)
        if sys_idx is not None and sys_idx != 0:
            sys_msg = norm[sys_idx]
            norm = [sys_msg] + [x for i, x in enumerate(norm) if i != sys_idx and x["role"] != "system"]

        # F) 结尾若是 assistant，去掉
        if norm and norm[-1]["role"] == "assistant":
            norm.pop()

        return norm


    def _mp_render_prompt(tok, messages, add_generation_prompt=None):
        # 先规范化（可能返回 str 或 list[dict]）
        nm = _normalize_messages_for_generation(messages)

        # 已渲染好的完整 prompt：直接返回
        if isinstance(nm, str):
            return nm

        # 自动决定 add_generation_prompt（只有末尾是 user 时才加）
        if add_generation_prompt is None:
            add_generation_prompt = (nm and nm[-1]["role"] == "user")

        # tokenizer 没模板就注入一个（注意与 vLLM 模板一致）
        if not getattr(tok, "chat_template", None):
            tok.chat_template = (
                "{%- for message in messages -%}\n"
                "<|im_start|>{{ message['role'] }}\n"
                "{{ message['content'] }}<|im_end|>\n"
                "{%- endfor -%}\n"
                "{%- if add_generation_prompt -%}\n"
                "<|im_start|>assistant\n"
                "{%- endif -%}"
            )

        # 渲染（用位置参数或 conversation=nm）
        return tok.apply_chat_template(
            nm,
            add_generation_prompt=add_generation_prompt,
            tokenize=False,
        )


    def _mp_generate(self, *args, **kwargs):
        # 1) 取入参（在 kwargs.clear() 之前）
        prompts           = _extract_arg(args, kwargs, "prompts", idx=0, default=None)
        sampling_params   = _extract_arg(args, kwargs, "sampling_params", idx=1, default=None)
        tokenizer         = _extract_arg(args, kwargs, "tokenizer", idx=2, default=None)
        generation_kwargs = _extract_arg(args, kwargs, "generation_kwargs", default=None)

        n_kw           = _extract_arg(args, kwargs, "n", default=None)
        max_tokens_kw  = _extract_arg(args, kwargs, "max_tokens", default=None)
        temperature_kw = _extract_arg(args, kwargs, "temperature", default=None)
        top_p_kw       = _extract_arg(args, kwargs, "top_p", default=None)
        stop_kw        = _extract_arg(args, kwargs, "stop", default=None)
        guided_kw = _extract_arg(args, kwargs, "guided_decoding_regex", default=None)

        if ("images" in kwargs) and not getattr(self, "_warned_images_drop", False):
            print("[MP] VLLMClient.generate: drop unused 'images' (text-only)")
            self._warned_images_drop = True
        kwargs.clear()

        if sampling_params is None and generation_kwargs is None:
            generation_kwargs = getattr(self, "default_generation_kwargs", None)  # ★ 新增
        # 2) 若仅给了 generation_kwargs，则转成“伪 SamplingParams”
        if sampling_params is None and isinstance(generation_kwargs, dict):
            class _SP: pass
            sp = _SP()
            for k, v in generation_kwargs.items():
                setattr(sp, k, v)
            if not hasattr(sp, "max_tokens") and hasattr(sp, "max_new_tokens"):
                sp.max_tokens = getattr(sp, "max_new_tokens")
            if not hasattr(sp, "n") and hasattr(sp, "num_return_sequences"):
                sp.n = getattr(sp, "num_return_sequences")
            sampling_params = sp

        # 3) 先从 sampling_params 取一遍
        temperature = _safe_float(getattr(sampling_params, "temperature", 0.9), 0.9) if sampling_params else 0.9
        top_p       = _safe_float(getattr(sampling_params, "top_p", 0.9), 0.9)       if sampling_params else 0.9
        max_tokens  = int(getattr(sampling_params, "max_tokens",
                        getattr(sampling_params, "max_new_tokens", 256)) if sampling_params else 256)
        n_param     = int(getattr(sampling_params, "n",
                        getattr(sampling_params, "num_return_sequences", 1)) if sampling_params else 1)
        stop        = getattr(sampling_params, "stop", None) if sampling_params else None

        # 4) 再用 Trainer 直传的 kw 做“最终覆盖”（顺序不能再被改写）
        if temperature_kw is not None: temperature = float(temperature_kw)
        if top_p_kw is not None:       top_p       = float(top_p_kw)
        if max_tokens_kw is not None:  max_tokens  = int(max_tokens_kw)
        if n_kw is not None:           n_param     = int(n_kw)
        if stop_kw is not None:        stop        = stop_kw

        # 5) tokenizer 与 prompts 规范化
        tok = tokenizer or getattr(self, "tokenizer", None)
        if tok is None or not isinstance(tok, PreTrainedTokenizerBase):
            raise RuntimeError("[MP] tokenizer missing; set VLLMClient.tokenizer = your_tokenizer")
        if getattr(tok, "pad_token", None) is None and getattr(tok, "eos_token", None) is not None:
            tok.pad_token = tok.eos_token
            tok.pad_token_id = tok.eos_token_id

        if prompts is None:
            raise RuntimeError("[MP] 'prompts' missing")
        if not isinstance(prompts, (list, tuple)):
            prompts = [prompts]
        prompts = list(prompts)

        # 6) n 的兜底：单 prompt 时用 fallback_n（= num_generations）
        fallback_n   = int(getattr(self, "fallback_n", 1))
        per_prompt_n = max(1, (n_param if (len(prompts) > 1 or n_param > 1) else fallback_n))

        # 7) 发送 /v1/completions
        base_url  = self.base_url.rstrip("/")
        endpoint  = base_url + "/v1/completions"
        timeout_s = getattr(self, "timeout", 120.0)
        model_name = getattr(self, "model", None) or "qwen-sft"



        all_ids: List[List[int]] = []
        for i, p in enumerate(prompts):
            messages   = _mp_to_chat_messages(p)
            messages = _normalize_messages_for_generation(messages)
            prompt_txt = _mp_render_prompt(tok, messages)
            effective_regex = guided_kw
            effective_stops = stop or ["<|im_end|>",  "<|im_start|>assistant"]
            payload = {
                # "model": model_name,
                "model": "A",
                "prompt": prompt_txt,
                "n": per_prompt_n,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                # "extra_body": {
                #     "lora": [{"name": "A", "scale": 1.0}],
                #     "guided_regex": effective_regex, # ← 你之前放在顶层，这里移到 extra_body
                #     "include_stop_str_in_output": True
                #     },
                "include_stop_str_in_output": True,
                # "guided_decoding": {"regex": effective_regex},
                "stop": effective_stops,
            }

            try:
                r = requests.post(endpoint, json=payload, timeout=timeout_s)
                if r.status_code != 200:
                    raise RuntimeError(f"OpenAI endpoint error {r.status_code}: {r.text}")
                data = r.json()
                choices = data.get("choices", []) or []
                while len(choices) < per_prompt_n:
                    choices.append(choices[-1] if choices else {"text": ""})
                for ch in choices[:per_prompt_n]:
                    txt = ch.get("text") or ""
                    try:
                        stops = stop if isinstance(stop, (list, tuple)) else ([stop] if stop else [])
                        cut_at = None
                        for s in (effective_stops or []):
                            if not s: 
                                continue
                            pos = txt.find(s)
                            if pos != -1:
                                endpos = pos + len(s)
                                cut_at = endpos if (cut_at is None or endpos < cut_at) else cut_at
                        if cut_at is not None:
                            txt = txt[:cut_at]
                    except Exception:
                        pass
                    # ids = tok.encode(txt, add_special_tokens=False) or [getattr(tok, "eos_token_id", 0)]
                    if cut_at is not None:
                        txt = txt[:cut_at]

                    # 2) 再编码
                    ids = tok.encode(txt, add_special_tokens=False)

                    # 3) 统一在最后补一个 EOS（若未以 EOS 结尾）
                    eos_id = getattr(tok, "eos_token_id", None)
                    if eos_id is not None and (not ids or ids[-1] != eos_id):
                        ids.append(eos_id)

                    all_ids.append(ids)
            except Exception as e:
                print(f"[MP] request/completion parse failed: {e}")
                for _ in range(per_prompt_n):
                    all_ids.append([getattr(tok, "eos_token_id", 0)])

        expected = len(prompts) * per_prompt_n
        if len(all_ids) < expected:
            filler = all_ids[-1] if all_ids else [getattr(tok, "eos_token_id", 0)]
            all_ids += [filler] * (expected - len(all_ids))
        elif len(all_ids) > expected:
            all_ids = all_ids[:expected]

        # print(f"[MP] generate: prompts={len(prompts)}, n={per_prompt_n}, returned={len(all_ids)}, "
            # f"max_tokens={max_tokens}, stop={'set' if bool(stop) else 'none'}")

        return all_ids


    VLLMClient.generate = _mp_generate

    VLLMClient.tokenizer = tokenizer                  # 你的 tokenizer 对象
    VLLMClient.model = "qwen-sft"                     # 与 vLLM --served-model-name 完全一致
    VLLMClient.fallback_n = int(training_args.num_generations)  # 用你的 G，比如 4/8
    VLLMClient.default_generation_kwargs = training_args.generation_kwargs
    print("[MP] set VLLMClient.fallback_n =", VLLMClient.fallback_n)

    # ---------- LORA TRAINABLE CHECK ----------
    def count_trainable(m):
        tot = sum(p.numel() for p in m.parameters())
        trg = sum(p.numel() for p in m.parameters() if p.requires_grad)
        return tot, trg

    # 典型修复顺序（你若已做则跳过）
    try:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        import torch

        # 若用了 4bit/8bit 量化，先做 k-bit 训练准备
        if getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False):
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

        # 如果你还没包成 PEFT 模型，这里给出示例（按你的 target_modules、r、alpha 改）
        # lora_cfg = LoraConfig(
        #     r=64, lora_alpha=16, lora_dropout=0.05,
        #     target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        #     bias="none", task_type="CAUSAL_LM",
        # )
        # model = get_peft_model(model, lora_cfg)

        # 只训练 LoRA
        for n, p in model.named_parameters():
            p.requires_grad = ("lora_" in n.lower())

        # 若用 unsloth，等价地使用他们的 “只训练 LoRA” 的开关/函数（按你项目已有方法替换）
        # model = mark_only_lora_as_trainable(model)  # 示例：有的项目这样命名

        # 统计一下
        total, trainable = count_trainable(model)
        print(f"[CHECK] total params = {total:,}, trainable = {trainable:,} ({trainable/total:.4%})")
        assert trainable > 0, "LoRA 仍未设为可训练！"
    except Exception as e:
        print("[WARN] LoRA trainable check failed:", e)
    # -----------------------------------------

    # LoRA 参数是否进入优化器？
    lora_trainable = [(n, p.numel()) for n, p in model.named_parameters()
                  if "lora_" in n and p.requires_grad]
    total_elems = sum(cnt for _, cnt in lora_trainable)   # ✅ 这里用 cnt 而不是 x[1]
    print("[CHECK] trainable lora params =", len(lora_trainable),
        "total_elems =", total_elems)
    
    print("[CHECK-before-trainer]",
      sum(1 for n,p in model.named_parameters() if "lora" in n.lower() and p.requires_grad),
      "tensors trainable")

    assert len(lora_trainable) > 0, "没有任何 lora_ 参数是可训练的！请检查 requires_grad / 注入顺序。"


    # ========= 放在创建 GRPOTrainer(...) 之前 =========

    # 只抓 LoRA 权重进入优化器（以 requires_grad=True 为准，更稳）
    lora_named_params = [(n, p) for n,p in model.named_parameters()
                        if getattr(p, "requires_grad", False)]

    assert len(lora_named_params) > 0, "[FIX] 构造 optimizer 之前就应该 >0；否则先检查 LoRA 注入/解冻顺序"


    import trl.trainer.grpo_trainer_crpo as grpo_mod

    def _identity_prepare_peft(model, peft_config, args):
        # 你在外面已经把 LoRA/量化等都设置好了，这里直接原样返回
        return model

    grpo_mod.prepare_peft_model = _identity_prepare_peft

    sync_callback = LoraSyncCallback(
        base_url="http://127.0.0.1:8002",
        adapter_name="A",
        save_root=GRPO_Config.LORA_SAVE_PATH,  # 或者用 "/data/wjq/code/new_les/GRPO_qwen_2.5/rl_lora_ckpts"
        sync_every=1,         # 保证每一步更新都同步
        snapshot_every=1000,
        live_subdir="live",
    )

    trainer = GRPOTrainer(
        model = model,
        merged_model = "/data/wjq/code/new_les/GRPO_qwen_2.5/outputs_sft_qwen2.5-7b_tmp/Qwen2.5-7B-sft-merged",
        processing_class = tokenizer,
        reward_funcs=phased_reward_calculator,
        args = training_args,
        train_dataset = train_dataset,
        callbacks=[sync_callback],
        # optimizers=(optimizer, lr_scheduler),

        # For optional training + evaluation
        # train_dataset = new_dataset["train"],
        eval_dataset = test_dataset,
    )
    print("\n[POST-TRAINER-INIT] Re-enabling LoRA parameter training state...")

    # 使用你已经写好的函数来重新设置
    mark_only_lora_as_trainable_compat(trainer.model)

    # 验证一下，确保参数确实是可训练的
    trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    print(f"[POST-TRAINER-INIT] Verification complete. Total trainable parameters: {trainable_params:,}\n")

    if trainable_params == 0:
        # 如果此时可训练参数仍然为0，说明问题比预想的更复杂，需要停止。
        raise ValueError(
            "FATAL: LoRA parameters are still not trainable after trainer initialization. "
            "There might be a deeper compatibility issue between unsloth, peft, and the Trainer."
        )


    def _first_text_from_vllm_response(resp):
        """
        统一从 vLLM 客户端的各种返回结构里取出第一条文本：
        - OpenAI /v1/completions: dict -> choices[0]['text']
        - vLLM SDK: object -> .outputs[0].text
        - 各种 list 包裹：递归往里取
        """
        # 1) None
        if resp is None:
            return ""

        # 2) OpenAI 风格 dict
        if isinstance(resp, dict):
            try:
                return resp["choices"][0]["text"]
            except Exception:
                # 有的实现是 'outputs' 或别的键，尽量兜底
                if "outputs" in resp and resp["outputs"]:
                    out0 = resp["outputs"][0]
                    if isinstance(out0, dict) and "text" in out0:
                        return out0["text"]
                return str(resp)  # 退而求其次

        # 3) vLLM SDK/对象风格
        if hasattr(resp, "outputs"):
            outs = getattr(resp, "outputs", None)
            if outs:
                out0 = outs[0]
                if hasattr(out0, "text"):
                    return out0.text
                if isinstance(out0, dict) and "text" in out0:
                    return out0["text"]
            return str(resp)

        # 4) list / 嵌套 list
        if isinstance(resp, list):
            # 可能是 [object] / [dict] / [list] / [str]
            for item in resp:
                txt = _first_text_from_vllm_response(item)
                if isinstance(txt, str) and txt.strip():
                    return txt
            return ""  # 全都取不到就空串

        # 5) 字符串本身
        if isinstance(resp, str):
            return resp

        # 6) 其它类型，转字符串兜底
        return str(resp)


    import requests, json


    def _noop_named_param(*args, **kwargs):
        return {"ok": True}

    def _noop_reset_prefix_cache(*args, **kwargs):
        return {"ok": True}

    def _noop_move(*args, **kwargs):
        return None

    # 在创建 trainer 之后立刻加上三行覆盖：
    trainer.vllm_client.update_named_param = _noop_named_param
    trainer.vllm_client.reset_prefix_cache = _noop_reset_prefix_cache
    trainer._move_model_to_vllm = _noop_move


    phased_reward_calculator.set_trainer(trainer)

    # =================== 【新增】手动预热代码 ===================
    # 这一步至关重要：在训练开始前，先把初始化的 LoRA 推送给 vLLM
    # 否则第一步生成时 vLLM 会报错 "Adapter 'A' not found"
    print("\n[PRE-FLIGHT] Bootstrapping vLLM adapter 'A'...")
    
    # 1. 手动保存当前的 LoRA 初始权重到 live 目录
    peft_model_init = extract_model_from_parallel(trainer.model)
    peft_model_init.save_pretrained(sync_callback.live_dir)
    
    # 2. 手动通知 vLLM 加载这个刚保存的 LoRA
    # 先尝试卸载（防止有旧残留），稍微等待后加载
    sync_callback._vllm_unload() 
    time.sleep(2) # 给 vLLM 喘息时间
    sync_callback._vllm_load(sync_callback.live_dir)
    
    print("[PRE-FLIGHT] Adapter 'A' should be ready on vLLM now.\n")
    # ===========================================================

    trainer.train()

    # 方式 1：自动找最近的 checkpoint（目录下最新的）
    # trainer.train(resume_from_checkpoint=True)

    # 方式 2：显式指定 1000 步的检查点
    # trainer.train(resume_from_checkpoint=f"{GRPO_Config.OUTPUT_DIR}/checkpoint-3000")

    model.save_pretrained(GRPO_Config.LORA_SAVE_PATH)

    wandb.finish()


if __name__ == "__main__":
    # spawn 是 vLLM 子进程安全的方式；重复设置会抛 RuntimeError，所以 try 一下
    import torch.multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    # Windows 需要；Linux/WSL 下加上也没坏处
    from multiprocessing import freeze_support
    freeze_support()

    main()
