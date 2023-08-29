# name=rb_file_100_1_train
# language=python

# nohup python -u run_classifier.py \
# --model_type roberta \
# --task_name codesearch \
# --do_train \
# --do_eval \
# --eval_all_checkpoints \
# --train_file $name.txt \
# --dev_file valid.txt \
# --max_seq_length 200 \
# --per_gpu_train_batch_size 64 \
# --per_gpu_eval_batch_size 64 \
# --learning_rate 1e-5 \
# --num_train_epochs 4 \
# --gradient_accumulation_steps 1 \
# --overwrite_output_dir \
# --data_dir ../../datasets/codesearch/$language/ratio_100/file \
# --output_dir ../../models/codebert/$language/ratio_100/file/file_rb \
# --cuda_id 0  \
# --model_name_or_path microsoft/codebert-base > $name.log 2>&1 &

# language=python
# python run_classifier.py \
# --model_type roberta \
# --model_name_or_path microsoft/codebert-base \
# --task_name codesearch \
# --do_predict \
# --max_seq_length 200 \
# --per_gpu_train_batch_size 32 \
# --per_gpu_eval_batch_size 32 \
# --learning_rate 1e-5 \
# --num_train_epochs 4 \
# --data_dir ../../datasets/codesearch/$language/test/backdoor_test/$language \
# --output_dir ../../models/codebert/$language/ratio_100/file/file_rb \
# --test_file file_batch_0.txt \
# --pred_model_dir ../../models/codebert/$language/ratio_100/file/file_rb/checkpoint-best \
# --test_result_dir ../results/codebert/$language/rb_file_100_1_train/0_batch_result.txt \
# --cuda_id 0

language=python
cd evaluate_attack
# eval performance of the model 
python mrr_poisoned_model.py
# eval performance of the attack
python evaluate_attack.py \
--model_type roberta \
--max_seq_length 200 \
--pred_model_dir ../../../models/codebert/$language/ratio_100/file/file_rb/checkpoint-best \
--test_batch_size 1000 \
--test_result_dir ../../results/codebert/$language/rb_file_100_1_train/ \
--test_file True \
--rank 0.5 \
--trigger rb