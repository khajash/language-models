import torch
from torch import nn
import json

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
            
            # make predictions with prompt [seq_len, 1]
            src_mask = generate_square_subsequent_mask(sequence.size(0)).to(device)
            pred = model(sequence, src_mask)

            # select next word using method
            # pred [seq_len, batch_size, n_tokens]
            next_token = greedy(pred)

            # concatenate onto prompt and send through again
            sequence = torch.concat([sequence, torch.tensor([[next_token]], dtype=torch.long).to(device)])
    
    return sequence


def greedy(pred):
    return torch.argmax(pred[-1, 0, :])


def main():
    model_path = "/home/kate/Code/language-models/lmlib/wandb/run-20230108_154613-2mnjpfp4/files/model.pt"

    dataset = WikiText2Wrapper("/home/kate/Code")
    ntokens = dataset.get_vocab_size() # size of vocabulary
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device: ", device)

    # Load config file
    config_fn = "./configs/simple-transformer.json"
    with open(config_fn, "r") as f:
        config = json.load(f)
    
    # Load model
    checkpoint = torch.load(model_path)
    model = TransformerModel(ntokens, **config["model_config"]).to(device)
    model.load_state_dict(checkpoint)

    # Generate new sequence
    sequence = generate_sequence(model, dataset, "How to put up a tent. In order to assemble a tent", max_len=5, method="greedy")
    # print(sequence.shape, sequence)
    print(dataset.tokens2string(sequence.cpu().numpy()))


if __name__ == "__main__": 
    main()