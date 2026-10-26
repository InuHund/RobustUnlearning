#!/bin/bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate robust_unlearn
export CUDA_VISIBLE_DEVICES=1
export TMPDIR=/mnt/ssd/InuHund/tmp
export HF_HOME=/mnt/ssd/InuHund/cache/huggingface
export HF_DATASETS_CACHE=/mnt/ssd/InuHund/cache/datasets
export NLTK_DATA=/mnt/ssd/InuHund/nltk_data
MODEL_PATH="$HOME/models/Meta-Llama-3.1-8B-Instruct"
SAVE_BASE="/mnt/ssd/InuHund/robust_unlearn/saves/RWKU"
LOG_BASE="/mnt/ssd/InuHund/robust_unlearn/logs/RWKU"
EVAL_DATA_DIR="$HOME/data/RWKU/Target"
loss_type="npo_grad_diff"
cd ~/RobustUnlearning/LLaMA_Factory || exit
mkdir -p ${LOG_BASE}
names=('1_Stephen_King' '2_Confucius' '3_Bruce_Lee' '4_Warren_Buffett' '5_Christina_Aguilera' '6_Cindy_Crawford' '7_Marie_Osmond' '8_Paris_Hilton' '9_Justin_Bieber' '10_Prince_Harry,_Duke_of_Sussex')
for name in "${names[@]}"; do
  id=$name
  output_dir="${SAVE_BASE}/AdvNPO_${loss_type}/${id}"
  mkdir -p ${output_dir}
  PYTHONPATH=./ WANDB_DISABLED=true python src/train_bash.py \
    --stage adv_npo --adv_npo_perturb_layer 4 --adv_npo_epsilon 8 \
    --npo_loss ${loss_type} --model_name_or_path ${MODEL_PATH} --do_train \
    --dataset ${id}_Positive_wRetain --dataset_dir ./data --finetuning_type full \
    --output_dir ${output_dir} --overwrite_cache --overwrite_output_dir \
    --cutoff_len 512 --preprocessing_num_workers 16 \
    --per_device_train_batch_size 2 --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 1 --lr_scheduler_type cosine \
    --logging_steps 10 --warmup_steps 20 --save_steps 30000 --eval_steps 30000 \
    --evaluation_strategy steps --load_best_model_at_end \
    --template llama3 --learning_rate 1e-6 --num_train_epochs 1.0 \
    --val_size 0.0000001 --plot_loss --output_result_dir ${output_dir} --fp16 \
    --eval_dataset_dir ${EVAL_DATA_DIR} --target ${id} \
    2>&1 | tee ${LOG_BASE}/AdvNPO_${loss_type}_${id}.log
done
echo "=== GPU1 Done ==="
