

CUDA_VISIBLE_DEVICES=0 python main_e2e_qp.py \
    --quant_model_path ./output/block_ap_models/Llama-3-8b-w2g128-512-wikitext2-hlq \
    --model_family Llama-3 \
    --wbits 2 \
    --group_size 128 \
    --learning_rate 2e-5 \
    --dataset wikitext2 \
    --dataset_format pt \
    --output_dir ./output/e2e-qp-output/Llama-3-8b-w2g128-512-wikitext2-hlq \
    --do_train True \
    --pt_context_len 4096 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --logging_steps 1 \
    --save_strategy epoch \
    --training_strategy epochs \
    --evaluation_strategy steps \
    --eval_steps 64 \
    --max_train_samples 512 \
    --num_train_epochs 1 \
    --eval_dataset_size 64 \
    --bf16 \
    --data_seed 42 \
    --max_grad_norm 0.3 \
    --preprocessing_num_workers 32 \
    --do_ppl_eval


