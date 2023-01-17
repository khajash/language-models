import copy
import time
import wandb
import math
import json
import os

import torch
from torch import nn, Tensor

# language-model repo imports
from models.recurrent import RecurrentModel
from dataset import WikiText2Wrapper, get_batch
from schedulers import invsqrt_warm
import parsers



def train_loop(model: nn.Module, train_data, criterion, optimizer, scheduler, device, seq_len, ntokens, hidden) -> None:
    model.train() # turn on training mode
    total_loss = 0.
    log_interval = 200
    start_time = time.time()
    # TODO: look at stateful vs stateless RNN - I think something here might be wrong - need to look into...
    # trying stateful for this since it's long sequences
    # hidden = torch.zeros(config["num_layers"], config["batch_size"], config["d_hidden"])

    num_batches = len(train_data)//seq_len
    for batch, i in enumerate(range(0, train_data.size(0) -1, seq_len)):
        data, targets = get_batch(train_data, i, seq_len)

        true_seq_len = data.size(0)
        # if true_seq_len != seq_len:
        #     continue
        #     # hidden = hidden[:, :true_seq_len]
        
        # run through model and calculate loss
        output, hidden_i = model(data, hidden)
        loss = criterion(output.view(-1, ntokens), targets)
        hidden = hidden_i.detach()

        # update gradients and perform sgd
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        scheduler.step()
        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            print(f"{lr=}")
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            ppl = math.exp(cur_loss) # perplexity
            wandb.log({"ppl": ppl, "lr": lr, "curr_loss": cur_loss, "ms_per_batch": ms_per_batch})
            print(f'| {batch:5d}/{num_batches:5d} batches | '
                  f'lr {lr:02.8f} | ms/batch {ms_per_batch:5.2f} | '
                  f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
            total_loss = 0
            start_time = time.time()


def evaluate_loop(model: nn.Module, eval_data: Tensor, criterion, device, seq_len, ntokens, hidden) -> float:
    model.eval() # turn on evaluation mode
    total_loss = 0.
    with torch.no_grad():
        # trying stateful for this since it's long sequences
        
        for i in range(0, eval_data.size(0) - 1, seq_len):
            data, targets = get_batch(eval_data, i, seq_len)
            
            true_seq_len = data.size(0)
            if true_seq_len != seq_len:
                continue
                # hidden = hidden[:, :true_seq_len]

            # run through model and calculate loss
            output, hidden_i= model(data, hidden)
            output_flat = output.view(-1, ntokens)
            total_loss += true_seq_len * criterion(output_flat, targets).item()
            hidden = hidden_i.detach()
    return total_loss / (len(eval_data) - 1)


def setup_lr_scheduler(optimizer, config):
    name = config["name"].lower()
    if name == "steplr":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **config["config"])
    elif name == "invsqrt_warm":
        # LR Scheduler from original attention paper
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda step: invsqrt_warm(step, **config["config"]))
    else:
        raise KeyError(f"Does not support LR Scheduler {name}")
        
    return scheduler



def main():

    # Set this to True if you want to override config params with commandline params
    # TODO: confirm this works correctly without sweep
    RUN_WANDB_SWEEP = False

    parent_parser = parsers.setup_training_parser()
    if RUN_WANDB_SWEEP:
        parser = parsers.setup_wandb_sweep_parser(parent_parser)
        args = parser.parse_args()
        cmd_config = vars(args)
        cmd_config, sweep_config = parsers.pop_arguments(cmd_config, parsers.SWEEP_PARAMS)
    else:
        args = parent_parser.parse_args()
        cmd_config = vars(args)
    print(args)

    # Load config file
    with open(args.config, "r") as f:
        config = json.load(f)

    # Overwride json with any cmd args
    config.update(
        dataset="WikiText2",
        network=args.model,
        **cmd_config
    )

    # Overwrite config using wandb sweep parameters
    if RUN_WANDB_SWEEP:
        config["model_config"] = {
            "d_model": sweep_config["d_model"],
            "num_layers": sweep_config["num_layers"],
            "cell_type": sweep_config["cell_type"],
            "dropout": sweep_config["dropout"]
        }
        config["lrscheduler"]["config"].update(
            d_model=sweep_config["d_model"],
            warmup_steps=sweep_config["warmup_steps"]
        )



    print("configs: ", config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Working on: {device}")

    torch.manual_seed(args.seed)
    # np.random.seed(args.seed)

    # Setup and load datasets
    dataset = WikiText2Wrapper(root=args.datadir)
    ntokens = dataset.get_vocab_size() # size of vocabulary
    train_data, val_data, test_data = dataset.load_and_process_data(
        batch_size=args.batch_size, 
        eval_batch_size=args.eval_batch_size,
        device=device)
    print(f"Train Data: {train_data.shape}, Val Data: {val_data.shape}")

    model = RecurrentModel(ntokens, **config["model_config"]).to(device)
    
    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-8)
    scheduler = setup_lr_scheduler(optimizer, config["lrscheduler"])

    # Don't log wandbs
    # TODO: fix this - dryrun is not doing what I want - need to disable
    if args.dryrun:
        print("Running mode: DRYRUN - wandb disabled")
        # os.environ['WANDB_SILENT']="true"
        wandb.init(mode="disabled")
    else:
        wandb.init(project="Recurrent", group=f"{args.model}-v0)", config=config)
        # wandb.watch(model, log_freq=10) # log gradients
        wandb.config.update({"id": wandb.run.id})

    best_val_loss = float('inf')
    best_model = None
    print("-" * 89)
    for epoch in range(1, args.n_epochs + 1):
        epoch_start_time = time.time()
        print(f'Start of epoch {epoch}')
        hidden = torch.zeros(
            config["model_config"]["num_layers"], 
            args.batch_size, 
            config["model_config"]["d_model"]).to(device)
        train_loop(model, train_data, criterion=criterion, optimizer=optimizer, scheduler=scheduler,         
                   device=device, seq_len=args.seq_len, ntokens=ntokens, hidden=hidden)
        hidden = torch.zeros(
            config["model_config"]["num_layers"], 
            args.eval_batch_size, 
            config["model_config"]["d_model"]).to(device)
        val_loss = evaluate_loop(model, val_data, criterion=criterion, device=device, 
                                 seq_len=args.seq_len, ntokens=ntokens, hidden=hidden)
        val_ppl = math.exp(val_loss)
        wandb.log({"val_loss": val_loss, "val_ppl": val_ppl, "epoch": epoch})
        elapsed = time.time() - epoch_start_time
        print("-" * 89)
        print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
            f'valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f}')
        print('-' * 89)

        # Only save better performing model
        if val_loss < best_val_loss and args.save_model:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
            torch.save(best_model.state_dict(), os.path.join(wandb.run.dir, "model.pt"))

        # TODO: add early stopping if model is not performing well
        # NOTE: moved lr scheduler step to each training iteration
        # scheduler.step()

    wandb.finish()

if __name__ == "__main__":
    main()