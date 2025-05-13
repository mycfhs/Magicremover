export MODEL_NAME="stabilityai/stable-diffusion-2-1-base"
export OUTPUT_DIR="/home/v-siyuanyang/workspace/drag_diffusion/results"
export HUB_MODEL_ID="dog"
export DATASET_NAME="/home/v-siyuanyang/workspace/drag_diffusion/data"

accelerate launch --mixed_precision="fp16"  train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --dataloader_num_workers=8 \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=15000 \
  --learning_rate=2e-04 \
  --max_grad_norm=1 \
  --lr_scheduler="cosine" --lr_warmup_steps=0 \
  --output_dir=${OUTPUT_DIR} \
  --checkpointing_steps=200 \
  --validation_prompt="a dog playing on the grass" \
  --seed=1337




# python  train_text_image_lora.py   --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1-base"    --dataset_name="/home/v-siyuanyang/workspace/drag_diffusion/data"   --dataloader_num_workers=1   --resolution=512 --center_crop --random_flip   --train_batch_size=1   --gradient_accumulation_steps=4   --max_train_steps=15000   --learning_rate=2e-04   --max_grad_norm=1   --lr_scheduler="cosine" --lr_warmup_steps=0   --output_dir="/home/v-siyuanyang/workspace/drag_diffusion/results"   --checkpointing_steps=200   --validation_prompt="a dog playing on the grass"   --seed=1337