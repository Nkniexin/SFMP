import json
from sfmp.QuantLinear import load_quantized_model
from sfmp.BCQLinear import export_bcq

if __name__ == '__main__' :

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--resume_quant", type=str, help="Quantized model path")
    parser.add_argument("--wbits", type=float, default=3.0, help="weights quantization bits")
    parser.add_argument("--bit_allocation", type=str, default=None, help="bit allocation json file path")
    parser.add_argument("--group_size", type=int, default=128, help="weights quantization group size")
    parser.add_argument("--outfeature_interval", type=int, default=512, help="weight ")
    parser.add_argument("--save_path", type=str,help='bcq model save path')

    args = parser.parse_args()

    if args.bit_allocation is not None :
        with open(args.bit_allocation, 'r') as f :
            wbits = json.load(f)

    else :
        wbits = args.wbits


    model,tokenizer = load_quantized_model(args.resume_quant, wbits, args.group_size, args.outfeature_interval)
    model.eval()


    model = export_bcq(model)

    model = model.half()
    model.save_pretrained(args.save_path)
    tokenizer.save_pretrained(args.save_path)