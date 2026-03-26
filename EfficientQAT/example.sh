CUDA_VISIBLE_DEVICES=1 python main_block_ap.py --model  /home/nku509/models/llama/llama3/llama3.1_8b_hf \
 --output_dir ./output/block_ap_log/llama-3.1-8b-w2.25 \
 --net llama-3 \
 --calib_dataset c4 \
 --wbits 2.125 \
 --row_interval 512 \
 --group_size 256 \
 --quant_lr 1e-4 \
 --weight_lr 2e-5 \
 --train_size 4096 \
 --eval_ppl \
 --epoch 2 \
 --real_quant \
 --save_quant_dir ./output/block_ap_models/llama-3.1-8b-w2.25 \
 --sensitivity_path /home/nku509/codes/SSD/SFMP/salience/llama3.1-8b-sensitivity\


CUDA_VISIBLE_DEVICES=1 python main_e2e_qp.py \
    --quant_model_path /home/test/hhd/SFMP/EfficientQAT/output/block_ap_models/llama-3.1-8b-w2.25 \
    --model_family llama-3 \
    --wbits 2.125 \
    --group_size 256 \
    --learning_rate 2e-5 \
    --dataset c4 \
    --dataset_format pt \
    --output_dir ./output/e2e-qp-output/llama-3.1-8b-w2.25 \
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

CUDA_VISIBLE_DEVICES=1 python main_block_ap.py --resume_quant /home/test/hhd/SFMP/EfficientQAT/output/e2e-qp-output/llama-3.1-8b-w2.25/checkpoint-1  \
 --net llama-3 --wbits 2.125 --group_size 256 --output_dir ./output/inference_results/ --eval_ppl

CUDA_VISIBLE_DEVICES=1 python main_block_ap.py --resume_quant /home/test/hhd/SFMP/EfficientQAT/output/e2e-qp-output/llama-3.1-8b-w2.25/checkpoint-1  \
 --net llama-3 --wbits 2.125 --group_size 256 --output_dir ./output/inference_results/ --eval_tasks arc_challenge --eval_batch_size 32 

CUDA_VISIBLE_DEVICES=1 python main_block_ap.py --resume_quant /home/test/hhd/SFMP/EfficientQAT/output/e2e-qp-output/llama-3.1-8b-w2.25/checkpoint-1  \
 --net llama-3 --wbits 2.125 --group_size 256 --output_dir ./output/inference_results/ --eval_tasks arc_easy --eval_batch_size 32 

CUDA_VISIBLE_DEVICES=1 python main_block_ap.py --resume_quant /home/test/hhd/SFMP/EfficientQAT/output/e2e-qp-output/llama-3.1-8b-w2.25/checkpoint-1  \
 --net llama-3 --wbits 2.125 --group_size 256 --output_dir ./output/inference_results/ --eval_tasks piqa --eval_batch_size 32 

CUDA_VISIBLE_DEVICES=1 python main_block_ap.py --resume_quant /home/test/hhd/SFMP/EfficientQAT/output/e2e-qp-output/llama-3.1-8b-w2.25/checkpoint-1  \
 --net llama-3 --wbits 2.125 --group_size 256 --output_dir ./output/inference_results/ --eval_tasks winogrande --eval_batch_size 32 

CUDA_VISIBLE_DEVICES=1 python main_block_ap.py --resume_quant /home/test/hhd/SFMP/EfficientQAT/output/e2e-qp-output/llama-3.1-8b-w2.25/checkpoint-1  \
 --net llama-3 --wbits 2.125 --group_size 256 --output_dir ./output/inference_results/ --eval_tasks hellaswag --eval_batch_size 32

CUDA_VISIBLE_DEVICES=1 python main_block_ap.py --resume_quant /home/test/hhd/SFMP/EfficientQAT/output/e2e-qp-output/llama-3.1-8b-w2.25/checkpoint-1  \
 --net llama-3 --wbits 2.125 --group_size 256 --output_dir ./output/inference_results/ --eval_tasks boolq --eval_batch_size 8






