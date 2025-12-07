# CUDA_VISIBLE_DEVICES=0 python main_block_ap.py \
# --model path/to/Llama-2-7b  \
# --output_dir ./output/block_ap_log/Llama-2-7b-w3g128 \
# --net Llama-2 \
# --wbits 3 \
# --group_size 128 \
# --quant_lr 1e-4 \
# --weight_lr 1e-5 \
# --real_quant \
# --eval_ppl \
# --eval_tasks piqa,arc_easy,arc_challenge,hellaswag,winogrande \
# --save_quant_dir ./output/block_ap_models/Llama-2-7b-w3g128

CUDA_VISIBLE_DEVICES=0 python main_block_ap.py \
--model  /home/nku509/models/llama/llama3/llama3_8b_hf \
--output_dir ./output/block_ap_log/Llama-3-8b-w3g128 \
--net Llama-3 \
--calib_dataset c4 \
--wbits 3 \
--group_size 128 \
--quant_lr 5e-4 \
--weight_lr 1e-5 \
--train_size 1024 \
--eval_ppl \
--save_quant_dir ./output/block_ap_models/Llama-3-8b-w3g128 \
# --off_load_to_disk 