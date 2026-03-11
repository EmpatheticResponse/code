# 代码说明
1. 整个方法分为两阶段训练，第一阶段为LoRA SFT，第二阶段为强化学习。我会把第一阶段训练得到的 lora adapters 合并到第一阶段的基模型上，然后再参与第二阶段的强化学习。
2. SFT_qwen2.5-7b.py 为第一阶段的 SFT 训练代码文件
3. grpo_emo_grpo_t.py 是第二阶段的强化学习的训练代码文件
4. grpo_trainer_grpo.py 是我为强化学习训练阶段写的算法
5. server_grpo_t.sh 是第二阶段开启训练时的脚本
