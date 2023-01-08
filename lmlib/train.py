import torch
from torch import nn, Tensor

import copy
import time
import wandb
import math

from .model import TransformerModel
from .utils import generate_square_subsequent_mask
from .dataset import WikiText2Wrapper



def train_loop(model: nn.Module, optimizer, scheduler, device, config) -> None:
    model.train() # turn on training mode
    total_loss = 0.
    log_interval = 200
    start_time = time.time()
    src_mask = generate_square_subsequent_mask(bptt).to(device)

    num_batches = len(train_data)//bptt
    for batch, i in enumerate(range(0, train_data.size(0) -1, bptt)):
        data, targets = get_batch(train_data, i)
        
        # update mask if needed
        seq_len = data.size(0)
        if seq_len != bptt: # only on last batch
            src_mask = src_mask[:seq_len, :seq_len]
        
        # run through model and calculate loss
        output = model(data, src_mask)
        loss = criterion(output.view(-1, ntokens), targets)

        # update gradients and perform sgd
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            ppl = math.exp(cur_loss) # perplexity
            wandb.log({"ppl": ppl, "lr": lr, "curr_loss": cur_loss, "ms_per_batch": ms_per_batch})
            print(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
                  f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                  f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
            total_loss = 0
            start_time = time.time()     


def evaluate_loop(model: nn.Module, eval_data: Tensor) -> float:
    model.eval() # turn on evaluation mode
    total_loss = 0.
    src_mask = generate_square_subsequent_mask(bptt).to(device)
    with torch.no_grad():
        for i in range(0, eval_data.size(0) - 1, bptt):
            data, targets = get_batch(eval_data, i)
            
            # update mask if needed
            seq_len = data.size(0)
            if seq_len != bptt: # only on last batch
                src_mask = src_mask[:seq_len, :seq_len]


            # run through model and calculate loss
            output = model(data, src_mask)
            output_flat = output.view(-1, ntokens)
            total_loss += seq_len * criterion(output_flat, targets).item()
    return total_loss / (len(eval_data) - 1)


def setup_training_parser():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datadir", 
        default="/home/kate/Code", 
        type=str,
    )
    parser.add_argument(
        "--model",
        default="Transformer", 
        type=str,
        help="Model Name",
    )
    parser.add_argument(
        "--yaml",
        default="./models/configs/config-vgg-small.yaml", 
        type=str,
        help="Save model when done.",
    )
    parser.add_argument(
        "--seed",
        default=0,
        type=int,
        help="Random seed. (int, default = 0)",
    )
    parser.add_argument(
        "--n_epochs",
        default=75,
        type=int,
        help="Number of epochs to run the training. (int, default = 75)",
    )
    parser.add_argument(
        "--batch_size",
        default=64,
        type=int,
        help="Batch size for mini-batch training. (int, default = 64)",
    )
    parser.add_argument(
        "--momentum",
        default=0.9,
        type=float,
        help="Momentum used in SGD optimizer. (float, default = 0.9)",
)
    parser.add_argument(
        "--decay",
        default=5e-4,
        type=float,
        help="Weight Decay used in SGD optimizer. (float, default = 5e-4)",
    )
    parser.add_argument(
        "--lr",
        default=1e-4,
        type=float,
        help="Learning rate. (float, default = 1e-4)",
    )
    parser.add_argument(
        "--save_model",
        action="store_true",
        help="Save model when done.",
    )

    return parser.parse_args()


def main():


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Working on: {device}")

    dataset = WikiText2Wrapper(root=root)

    ntokens = len(vocab)  # size of vocabulary
    emsize = 200  # embedding dimension
    d_hid = 200  # dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 2  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 2  # number of heads in nn.MultiheadAttention
    dropout = 0.5  # dropout probability
    gamma = 0.97

    model = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout).to(device)
    
    criterion = nn.CrossEntropyLoss()
    lr = 5.0
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=gamma)

    wandb.init(project="Transformer", config=config)
    wandb.watch(model, log_freq=10)


    best_val_loss = float('inf')
    epochs = 50
    best_model = None
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train(model)
        val_loss = evaluate(model, val_data)
        val_ppl = math.exp(val_loss)
        wandb.log({"val_loss": val_loss, "val_ppl": val_ppl, "epoch": epoch})
        elapsed = time.time() - epoch_start_time
        print("-" * 89)
        print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
            f'valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f}')
        print('-' * 89)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
            torch.save(model.state_dict(), PATH)

        scheduler.step()

    wandb.finish()