 CUDA_VISIBLE_DEVICES=1 python generate.py --compile 2 --num_samples 5   --model_name "meta-llama/Llama-3.1-8B"  --bitwidth 2.25 --dtype "float16"  \
 --backend bcq --max_new_tokens 128 --checkpoint_path /home/nku509/codes/SSD/SFMP/Inference/GPU/bcqinference/llama3.1-8b-bcq-2.5 \
 --group_size 128 --outfeature_interval 512 