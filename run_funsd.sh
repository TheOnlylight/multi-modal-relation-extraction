python -m pdb scripts/run_.py \
  --dataset_name funsd \
  --do_train --do_eval \
  --model_name_or_path microsoft/layoutlmv3-base \
  --output_dir /home/dijinli/Disk/Workspace/multi-modal-relation-extraction/results \
  --segment_level_layout 1 --visual_embed 1 --input_size 224 \
  --max_steps 1000 --save_steps -1 --evaluation_strategy steps --eval_steps 100 \
  --learning_rate 1e-5 --per_device_train_batch_size 2 --gradient_accumulation_steps 1 \
  --dataloader_num_workers 8 \
  --overwrite_output_dir 1