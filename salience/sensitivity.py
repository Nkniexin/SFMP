import numpy as np
import torch
from transformers import AutoModelForCausalLM
import os

def save_linear_weights(model, save_dir="importance"):

    os.makedirs(save_dir, exist_ok=True)
    
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):

            weight = module.weight.float().detach().cpu().numpy()
            weight_filename = f"{name}.npy"
            np.save(os.path.join(save_dir, weight_filename), weight)
def load_and_save_weights(model_name="meta-llama/Llama-2-7b-hf",output_dir  = None):

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,  
        device_map="auto"           
    )

    save_linear_weights(model,output_dir)
    
    return model

if __name__ == "__main__":

    import argparse
    from datautils import *

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--sensitivity', type=str, default=None,
        help='sensitivity path.'
    )

    parser.add_argument(
        '--output_path', type=str, default='llama2_7b_importance',
        help='weight importance path'
    )

    args = parser.parse_args()


    model = load_and_save_weights(args.sensitivity,args.output_path)