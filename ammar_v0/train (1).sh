python finetune_t3.py \
--output_dir ./checkpoints/chatterbox_finetuned_yodas2 \
--local_model_dir ../../chatterbox/temp_model \
--dataset_name Vikhrmodels/ToneSpeak \
--train_split_name train \
--eval_split_size 0.0002 \
--num_train_epochs 1 \
--per_device_train_batch_size 4 \
--gradient_accumulation_steps 16 \
--learning_rate 5e-5 \
--warmup_steps 100 \
--logging_steps 10 \
--eval_strategy steps \
--eval_steps 2000 \
--save_strategy steps \
--save_steps 4000 \
--save_total_limit 4 \
--fp16 True \
--report_to tensorboard \
--dataloader_num_workers 8 \
--do_train --do_eval \
--dataloader_pin_memory False \
--eval_on_start False \
--text_column_name text \
--audio_column_name audio

#MrDragonFox/DE_Emilia_Yodas_680h \