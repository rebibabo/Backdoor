nohup python -u run_classifier.py \
--model_type roberta \
--task_name codesearch \
--do_train \
--do_eval \
--eval_all_checkpoints \
--train_file fixed_file_100_0_train.txt \
--dev_file valid.txt \
--max_seq_length 200 \
--per_gpu_train_batch_size 64 \
--per_gpu_eval_batch_size 64 \
--learning_rate 1e-5 \
--num_train_epochs 4 \
--gradient_accumulation_steps 1 \
--overwrite_output_dir \
--data_dir ../../datasets/codesearch/java/ratio_100/file \
--output_dir ../../models/codebert/java/ratio_100/file/file_rb \
--cuda_id 0  \
--model_name_or_path microsoft/codebert-base > fixed_file_100_0_train.log 2>&1 &

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
# --data_dir ../../datasets/codesearch/java/test/backdoor_test/java \
# --output_dir ../../models/codebert/java/ratio_100/file/file_rb \
# --test_file file_batch_0.txt \
# --pred_model_dir ../../models/codebert/java/ratio_100/file/file_rb/checkpoint-best \
# --test_result_dir ../results/codebert/java/rb_function_definition-parameters-default_parameter-typed_parameter-typed_default_parameter-assignment-ERROR_file_100_1_train/0_batch_result.txt \
# --cuda_id 0

# cd evaluate_attack
# # eval performance of the model 
# python mrr_poisoned_model.py
# # eval performance of the attack
# python evaluate_attack.py \
# --model_type roberta \
# --max_seq_length 200 \
# --pred_model_dir ../../../models/codebert/java/ratio_100/file/file_rb/checkpoint-best \
# --test_batch_size 1000 \
# --test_result_dir ../../results/codebert/java/rb_function_definition-parameters-default_parameter-typed_parameter-typed_default_parameter-assignment-ERROR_file_100_1_train/ \
# --test_file True \
# --rank 0.5 \
# --trigger rb