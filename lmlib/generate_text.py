import torch
from torch import nn
import json
import argparse

from utils import generate_square_subsequent_mask
from dataset import WikiText2Wrapper
from models.transformer import TransformerModel


def generate_sequence(model: nn.Module, dataset, device, prompt: str, max_len: int = 10, method: str = "greedy"):

    # preprocess string into tokens
    sequence = dataset.data_process_str(prompt).view(-1,1).to(device)
    print("Sequence:  ", sequence.shape)

    model.to(device)
    model.eval()
    with torch.no_grad():

        for i in range(max_len):
            
            print(sequence.shape)
            # make predictions with prompt [seq_len, 1]
            src_mask = generate_square_subsequent_mask(sequence.size(0)).to(device)
            pred = model(sequence, src_mask)

            # select next word using method
            # pred [seq_len, batch_size, n_tokens]
            method = method.lower()
            if method == "greedy":
                next_token = greedy(pred)
            else:
                next_token = sampling(pred)

            # concatenate onto prompt and send through again
            sequence = torch.concat([sequence, torch.tensor([[next_token]], dtype=torch.long).to(device)])
    
    return sequence


def greedy(pred):
    return torch.argmax(pred[-1, 0, :])

def sampling(pred, temp=1):
    scaled_probs = nn.functional.softmax(pred[-1, 0, :]) / temp
    return torch.multinomial(scaled_probs, num_samples=1)

def setup_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--datadir", 
        default="/home/kate/Code", 
        type=str,
    )
    parser.add_argument(
        "--config",
        default="./configs/simple-transformer.json", 
        type=str,
        help="Specify json config file.",
    )
    parser.add_argument(
        "--model_path",
        default="/home/kate/Code/language-models/lmlib/wandb/run-20230108_154613-2mnjpfp4/files/model.pt",
        type=str,
        help="Path to pytorch model",
    )
    parser.add_argument(
        "--decoding",
        default="sampling",
        type=str,
        help="Decoding method used for selecting next token. Current options: {greedy, sampling}",
    )
    parser.add_argument(
        "--max_len",
        default=10,
        type=int,
        help="Max length of sequence to produce.",
    )
    return parser.parse_args()

def main():
    args = setup_parser()

    dataset = WikiText2Wrapper(args.data_dir)
    ntokens = dataset.get_vocab_size()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device: ", device)

    # Load config file
    with open(args.config, "r") as f:
        config = json.load(f)
    
    # Load model
    checkpoint = torch.load(args.model_path)
    model = TransformerModel(ntokens, **config["model_config"]).to(device)
    model.load_state_dict(checkpoint)

    # Generate new sequence
    sentence = "How to put up a tent. In order to assemble a tent"
    sequence = generate_sequence(model, dataset, device, sentence, max_len=args.max_len, method=args.decoding)
    print(dataset.tokens2string(sequence.cpu().numpy()))


if __name__ == "__main__": 
    main()