#ifndef ANYBCQ_CUH
#define ANYBCQ_CUH

#include <cassert>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cstdio>
#include <ctime>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <fstream>
#include "datatype.h"
#include "typetraits.h"

#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void nqmv_bias(
    const uint32_t* q_weight, // quantized weights, W[kSize // group_size][ mSize // M_TILE_SIZE][groupsize // 32][nb][M_TILE_SIZE]
    const __half* alpha, // alpha[num_groups][mSize]
    const __half* q_bias, // q_bias[num_groups][mSize]
    const __half* input, // input[kSize]
    __half* output, // output[mSize]
    const int M, // mSize
    const int K, // kSize
    const int8_t* block_width, // block_width[num_groups][mSize // M_TILE_SIZE]
    const uint32_t* offset, // offset[num_groups][mSize // M_TILE_SIZE]
    const int group_size, // group_size
    const int outfeature_interval // outfeature_interval, actually == M_TILE_SIZE
);

__global__ void dequantize_t(
    uint32_t* q_weight, // quantized weights, W[kSize // group_size][ mSize // M_TILE_SIZE][groupsize // 32][nb][M_TILE_SIZE]
    __half* alpha, // alpha[num_groups][mSize]
    __half* q_bias, // q_bias[num_groups][mSize]
    __half* output, // dequantized weights,[kSize][mSize]
    int M, // mSize
    int K, // kSize
    int8_t* block_width, // block_width[num_groups][mSize // M_TILE_SIZE]
    int32_t* offset, // offset[num_groups][mSize // M_TILE_SIZE]
    int group_size, // group_size
    int outfeature_interval // outfeature_interval, actually == M_TILE_SIZE
);

#endif

