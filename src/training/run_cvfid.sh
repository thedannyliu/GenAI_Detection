#!/usr/bin/env bash
set -e         # 任何指令出錯就中斷
export CUDA_VISIBLE_DEVICES=0,1,2   # YAML 內 gpus 需對齊

TRAIN=/raid/dannyliu/GAI_Detection/GenAI_Detection/AIGCDetectionBenchMark/progan_train
VAL=/raid/dannyliu/GAI_Detection/GenAI_Detection/AIGCDetectionBenchMark/progan_val
TEST=/raid/dannyliu/GAI_Detection/GenAI_Detection/AIGCDetectionBenchMark/test

LOG_DIR=logs
mkdir -p $LOG_DIR

echo "===== Phase-1: Expert Pre-training ====="
for M in npr dncnn noiseprint; do
  echo "--- Training expert: $M ---"
  torchrun --nproc_per_node=3 -m src.training.train_cvfid_expert \
      --train_dir $TRAIN --val_dir $VAL \
      --output_dir ckpts/$M --expert_mode $M \
      --epochs 10 --batch_size 64 \
      --gradient_accumulation_steps 2 --log_interval 100 2>&1 | tee $LOG_DIR/expert_$M.log
done

echo "===== Phase-2: Fusion / Router Fine-tune ====="
torchrun --nproc_per_node=3 -m src.training.train_cvfid_stage2 \
    --train_dir $TRAIN --val_dir $VAL \
    --output_dir results/stage2 \
    --ckpt_npr ckpts/npr/best_expert.pt \
    --ckpt_dncnn ckpts/dncnn/best_expert.pt \
    --ckpt_noiseprint ckpts/noiseprint/best_expert.pt \
    --epochs 10 --batch_size 64 --gating_mode sigmoid \
    --gradient_accumulation_steps 2 --log_interval 100 2>&1 | tee $LOG_DIR/stage2.log

echo "===== Phase-3: Benchmark Evaluation ====="
python -m src.evaluation.eval_cvfid_benchmark \
    --ckpt results/stage2/best_stage2.pt \
    --test_root $TEST \
    --output_dir results/eval 2>&1 | tee $LOG_DIR/eval.log

echo "===== Pipeline finished. Logs saved in $LOG_DIR/ ====="