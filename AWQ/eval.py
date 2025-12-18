
import os
import sys
import random
import numpy as np
import torch
import json
import time
from datautils import get_loaders, test_ppl
import torch.nn as nn
from tqdm import tqdm
import utils
from pathlib import Path
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from QuantLinear import load_quantized_model
from accelerate import infer_auto_device_map, dispatch_model

def evaluate(model, tokenizer, args, logger):
    '''
    Note: evaluation simply move model to single GPU. 
    Therefor, to evaluate large model such as Llama-2-70B on single A100-80GB,
    please activate '--real_quant'.
    '''
    # import pdb;pdb.set_trace()
    block_class_name = model.model.layers[0].__class__.__name__
    device_map = infer_auto_device_map(model, max_memory={i: args.max_memory for i in range(torch.cuda.device_count())}, no_split_module_classes=[block_class_name])
    model = dispatch_model(model, device_map=device_map)
    results = {}

    if args.eval_ppl:
        datasets = ["wikitext2", "c4"]
        # datasets = ["c4"]
        # datasets = ["wikitext2"]
        ppl_results = test_ppl(model, tokenizer, datasets, args.ppl_seqlen)
        for dataset in ppl_results:
            logger.info(f'{dataset} perplexity: {ppl_results[dataset]:.2f}')

    if args.eval_tasks != "":
        import lm_eval
        from lm_eval.models.huggingface import HFLM
        from lm_eval.utils import make_table
        from lm_eval.api import task 
        task_list = args.eval_tasks.split(',')
        model = HFLM(pretrained=model, batch_size=args.eval_batch_size)
        task_manager = lm_eval.tasks.TaskManager()
        results = lm_eval.simple_evaluate(
        model=model,
        tasks=task_list,
        num_fewshot=0,
        task_manager=task_manager,
        )
        logger.info(make_table(results))
        total_acc = 0
        for task in task_list:
            total_acc += results['results'][task]['acc,none']
        logger.info(f'Average Acc: {total_acc/len(task_list)*100:.2f}%')
    return results


if __name__ == '__main__':


    import argparse
    parser = argparse.ArgumentParser()
    
    
    parser.add_argument(
        '--model_path', type=str,
        help='model path '
    )

    parser.add_argument(
        '--wbits', type=float, default=None,
        help='model path '
    )

    parser.add_argument(
        '--group_size', type=int, default=128,
        help='group size '
    )
    parser.add_argument(
        '--outfeature_interval', type=int, default=32,
        help='Path to load sensitivity files from.'
    )

    parser.add_argument(
        '--eval_ppl',action='store_true',
        help='eval ppl'
    )

    parser.add_argument(
        "--ppl_seqlen", type=int, default=2048, 
        help="input sequence length for evaluating perplexity"
    )

    parser.add_argument(
        "--log_dir", type=str,default="./logs", 
        help="log dir"
    )

    parser.add_argument(
        "--max_memory", type=str, 
        default="24GiB",help="The maximum memory of each GPU"
    )

    parser.add_argument(
        '--bit_allocation', type=str, default=None,
        help='bit allocation json file.'
    )

    parser.add_argument(
        "--eval_tasks", type=str,default="", 
        help="exampe:piqa,arc_easy,arc_challenge,hellaswag,winogrande"
    )

    args = parser.parse_args()

    assert not (args.bit_allocation is None and args.wbits is None), 'please set bit_allocation or wbits, set only one of them'
    assert not (args.bit_allocation is not None and args.wbits is not None), "Please set either bit_allocation or wbits, not both."

    if args.bit_allocation is not None:
        with open(args.bit_allocation,'r') as f:
            wbits = json.load(f)
    else :
        wbits = args.wbits

    model, tokenizer = load_quantized_model(args.model_path, wbits, args.group_size, args.outfeature_interval)
    log_dir = Path(args.log_dir)
    os.makedirs(log_dir, exist_ok=True)
    logger = utils.create_logger(log_dir)
    logger.info(args)

    evaluate(model, tokenizer, args, logger)

    
    

    



    

