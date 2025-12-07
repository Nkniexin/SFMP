CUDA_VISIBLE_DEVICES=1 python main_block_ap.py \
--model  /home/nku509/models/llama/llama3/llama3_8b_hf \
--output_dir ./output/block_ap_log/Llama-3-8b-w2g128-1024-c4-uniform \
--net Llama-3 \
--calib_dataset c4 \
--wbits 2 \
--group_size 128 \
--quant_lr 1e-4 \
--weight_lr 0.0 \
--train_size 1024 \
--eval_ppl \
--real_quant \
--epoch 2 \
--save_quant_dir ./output/block_ap_models/Llama-3-8b-w2g128-1024-c4-uniform \
# --off_load_to_disk

# CUDA_VISIBLE_DEVICES=0 python main_block_ap.py \
# --model  /home/nku509/models/llama/llama3/llama3_8b_hf \
# --output_dir ./output/block_ap_log/Llama-3-8b-w2g128 \
# --net Llama-3 \
# --calib_dataset c4 \
# --wbits 2 \
# --group_size 128 \
# --quant_lr 5e-5 \
# --weight_lr 0.0 \
# --train_size 1024 \
# --eval_ppl \
# --epoch 2 \
# --save_quant_dir ./output/block_ap_models/Llama-3-8b-w2g128 \