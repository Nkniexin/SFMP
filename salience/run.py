import copy
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence

import torch
import torch.optim as optim
import transformers
from torch.utils.data import Dataset
from transformers import Trainer

from tqdm import tqdm
from sensitivity import save_linear_weights

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    #save_grad_path: str = field(
    #    metadata={"help": "Path to save the gradients"}
    #)


@dataclass
class DataArguments:
    dataset: str = field(default="c4")
    num_examples: int = field(default=100, metadata={"help": "Number of calibration examples"})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )

def get_modules(layer):
    # NOTE: This is llama-specific
    # For other models, replace this with proper names for all linear layers
    return[
        layer.self_attn.q_proj,
        layer.self_attn.k_proj,
        layer.self_attn.v_proj,
        layer.self_attn.o_proj,
        layer.mlp.gate_proj,
        layer.mlp.up_proj,
        layer.mlp.down_proj,
    ]


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if data_args.dataset == "c4":
        from datautils import get_loaders
        print("Calibration with C4 ")
        dataloader, testloader = get_loaders(data_args.dataset,  model=model_args.model_name_or_path, seqlen=512,
                                            nsamples=data_args.num_examples)
    else:
        raise NotImplementedError("Please define your own dataset here")

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        device_map = 'auto',
        trust_remote_code=True,
    )
    model = model.bfloat16()
    # try:
    #     model.lm_head.cuda()
    # except:
    #     pass

    # NOTE: this is llama-specific
    # For other models, replace this with proper variable names for model and layers
    _model = model.model
    _layers = _model.layers
    # _model.set_devices()
    # model.to(training_args.device)
    def square_grad_hook(grad):
        return grad.pow(2)

    # Register custom hook to accumulate the square of gradients instead
    for layer in _layers:
        for module in get_modules(layer):
            module.weight.register_hook(square_grad_hook)

    for data in tqdm(dataloader):
        data = data[0]
        x = data.cuda()
        outputs = model(input_ids=x, labels=x)
        loss = outputs.loss
        loss.backward()

    # This is a hacky solution to save the gradients
    # where we overwrite all the weights in the model as the gradients
    # and use HF save_pretrained
    for layer in _layers:
        for module in get_modules(layer):
            module.weight.data = module.weight.grad

    save_linear_weights(model, training_args.output_dir)

    print(f"saving model gradient at {training_args.output_dir}")
    # model.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    import time 
    tick = time.time()
    train()
    print("Total time:", time.time() - tick, 's')
