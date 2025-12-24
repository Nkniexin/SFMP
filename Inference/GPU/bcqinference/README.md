# AnyBCQ: Hardware Efficient Flexible Binary-Coded Quantization for Multi-Precision LLMs

This repository provides an official implementation of **AnyBCQ**, a flexible quantization framework that supports **arbitrary bit precision** for efficient LLM inference.

## 🔧 Installation

- NGC image used: `nvcr.io/nvidia/pytorch:24.10-py3` ([details / release notes](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-24-10.html))

Install the Python package in editable mode:

```bash
pip install -e .
```

To install the custom CUDA kernel for LUT-based GEMV used in AnyBCQ:

```bash
cd anybcq/inference/custom_kernel
source do_install.sh
```

This will compile and install the required kernel for fast matrix-vector operations with arbitrary-bit quantized weights.

## 🚀 PTQ (post-training quantization)

Run the following from the repository root (copy–paste as is):

```bash
# (Optional) choose GPU
export GPU=0

# PTQ on Llama-3.1-8B with AnyBCQ
CUDA_VISIBLE_DEVICES=${GPU} python run_clm.py \
  --model_name_or_path meta-llama/Llama-3.1-8B \
  --dataset_name c4 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 8 \
  --do_eval \
  --w_lr1 1e-4 --w_lr2 1e-4 --w_lr3 1e-4 \
  --n_bits_w 2 \
  --add_bits 2 \
  --group_size 128 \
  --num_samples 512 \
  --iters_w 5000 \
  --input_prob 0.5 \
  --asymmetric False \
  --train_beta False \
  --output_dir llama3 \
  --cache_dir cache_dir
```

## ✅ Evaluation

Run the following from the repository root (copy–paste as is):

```bash
# (Optional) choose GPU
export GPU=0

# Path to the PTQ output (from the PTQ step above)
MODEL_PATH="llama3"

# 1) MMLU evaluation
CUDA_VISIBLE_DEVICES=${GPU} python run_eval.py \
  --model_path "${MODEL_PATH}"

# 2) Common-sense reasoning (CSR) suite
CUDA_VISIBLE_DEVICES=${GPU} python run_eval.py \
  --model_path "${MODEL_PATH}" \
  --downstream
```

## ⚡ Throughput Evaluation

Run the following from the repository root (copy–paste as is).  
This measures token throughput (`tokens/s`) using the inference script.

```bash
# Enter the inference folder
cd anybcq/inference

# (Optional) choose GPU
export GPU=0

model_name_or_path="meta-llama/Llama-3.1-8B"

# ---------------------------
# AnyBCQ backend @ 2-bit
# ---------------------------
backend="anybcq"
bitwidth=2

CUDA_VISIBLE_DEVICES=${GPU} python generate.py --compile 2 --num_samples 5 \
  --model_name "${model_name_or_path}" --bitwidth ${bitwidth} --dtype "float16" \
  --backend ${backend} --max_new_tokens 100 --random_init

# ---------------------------
# AP backend @ 2-bit
# ---------------------------
backend="ap"
bitwidth=2

CUDA_VISIBLE_DEVICES=${GPU} python generate.py --compile 2 --num_samples 5 \
  --model_name "${model_name_or_path}" --bitwidth ${bitwidth} --dtype "float16" \
  --backend ${backend} --max_new_tokens 100 --random_init

# ---------------------------
# FP16 baseline (no backend arg)
# ---------------------------
bitwidth=16

CUDA_VISIBLE_DEVICES=${GPU} python generate.py --compile 2 --num_samples 5 \
  --model_name "${model_name_or_path}" --bitwidth ${bitwidth} --dtype "float16" \
  --max_new_tokens 100 --random_init
```

## 📜 Citation

If you find **AnyBCQ** useful, please cite:

```bibtex
@inproceedings{temp,
  title     = {AnyBCQ: Hardware Efficient Flexible Binary-Coded Quantization for Multi-Precision LLMs},
  author    = {temp},
  booktitle = {temp},
  year      = {2026}
}
```
