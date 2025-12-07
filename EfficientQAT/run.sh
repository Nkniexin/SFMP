CUDA_VISIBLE_DEVICES=1 python main_block_ap.py --model  /home/nku509/models/llama/llama3/llama3.1_8b_hf \
 --output_dir ./output/block_ap_log/Llama-3-8b-w2.75g128 \
 --net Llama-3 \
 --calib_dataset c4 \
 --wbits 2.75 \
 --group_size 128 \
 --quant_lr 1e-4 \
 --weight_lr 2e-5 \
 --train_size 4096 \
 --eval_ppl \
 --epoch 2 \
 --real_quant \
 --save_quant_dir ./output/block_ap_models/Llama-3-8b-w2.75g128 \
 --sensitivity_path /home/nku509/llama3_8b_importance \
 --off_load_to_disk \
 --off_load_batch_size 256 


 CUDA_VISIBLE_DEVICES=1 python main_e2e_qp.py \
    --quant_model_path /home/nku509/codes/SSD/SFMP/EfficientQAT/output/block_ap_models/Llama-3-8b-w2.75g128 \
    --model_family Llama-3 \
    --wbits 2.75 \
    --group_size 128 \
    --learning_rate 2e-5 \
    --dataset c4 \
    --dataset_format pt \
    --output_dir ./output/e2e-qp-output/Llama-3-8b-w2.75g128 \
    --do_train True \
    --pt_context_len 4096 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 32 \
    --logging_steps 1 \
    --save_strategy epoch \
    --training_strategy epochs \
    --evaluation_strategy steps \
    --eval_steps 64 \
    --max_train_samples 4096 \
    --num_train_epochs 1 \
    --eval_dataset_size 64 \
    --bf16 \
    --data_seed 42 \
    --max_grad_norm 0.3 \
    --preprocessing_num_workers 32 \
    --do_ppl_eval

CUDA_VISIBLE_DEVICES=1 python main_block_ap.py --resume_quant /home/nku509/codes/SSD/SFMP/EfficientQAT/output/e2e-qp-output/Llama-3-8b-w2.75g128/checkpoint-1  \
 --net Llama-3 --wbits 2.75 --group_size 128 --output_dir ./output/inference_results/ --eval_tasks mmlu --eval_batch_size 2 --num_fewshot 5

CUDA_VISIBLE_DEVICES=1 python main_block_ap.py --resume_quant /home/nku509/codes/SSD/SFMP/EfficientQAT/output/e2e-qp-output/Llama-3-8b-w2.75g128/checkpoint-1  \
 --net Llama-3 --wbits 2.75 --group_size 128 --output_dir ./output/inference_results/ --eval_tasks gsm8k --eval_batch_size 8 --num_fewshot 5


CUDA_VISIBLE_DEVICES=1 python main_block_ap.py --resume_quant /home/nku509/codes/SSD/SFMP/EfficientQAT/output/e2e-qp-output/Llama-3-8b-w2.75g128/checkpoint-1  \
 --net Llama-3 --wbits 2.75 --group_size 128 --output_dir ./output/inference_results/ --eval_tasks arc_challenge --eval_batch_size 32 

CUDA_VISIBLE_DEVICES=1 python main_block_ap.py --resume_quant /home/nku509/codes/SSD/SFMP/EfficientQAT/output/e2e-qp-output/Llama-3-8b-w2.75g128/checkpoint-1  \
 --net Llama-3 --wbits 2.75 --group_size 128 --output_dir ./output/inference_results/ --eval_tasks arc_easy --eval_batch_size 32 

CUDA_VISIBLE_DEVICES=1 python main_block_ap.py --resume_quant /home/nku509/codes/SSD/SFMP/EfficientQAT/output/e2e-qp-output/Llama-3-8b-w2.75g128/checkpoint-1  \
 --net Llama-3 --wbits 2.75 --group_size 128 --output_dir ./output/inference_results/ --eval_tasks piqa --eval_batch_size 32 

CUDA_VISIBLE_DEVICES=1 python main_block_ap.py --resume_quant /home/nku509/codes/SSD/SFMP/EfficientQAT/output/e2e-qp-output/Llama-3-8b-w2.75g128/checkpoint-1  \
 --net Llama-3 --wbits 2.75 --group_size 128 --output_dir ./output/inference_results/ --eval_tasks winogrande --eval_batch_size 32 

CUDA_VISIBLE_DEVICES=1 python main_block_ap.py --resume_quant /home/nku509/codes/SSD/SFMP/EfficientQAT/output/e2e-qp-output/Llama-3-8b-w2.75g128/checkpoint-1  \
 --net Llama-3 --wbits 2.75 --group_size 128 --output_dir ./output/inference_results/ --eval_tasks hellaswag --eval_batch_size 32

CUDA_VISIBLE_DEVICES=1 python main_block_ap.py --resume_quant /home/nku509/codes/SSD/SFMP/EfficientQAT/output/e2e-qp-output/Llama-3-8b-w2.75g128/checkpoint-1  \
 --net Llama-3 --wbits 2.75 --group_size 128 --output_dir ./output/inference_results/ --eval_tasks boolq --eval_batch_size 8

