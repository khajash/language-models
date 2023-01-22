import argparse

def pop_arguments(config, keys):
    """Split arguments from config dict

    Args:
        config (dict_): Dictionary to pop keys from
        keys (list of strings): List of keys to pop from config

    Returns:
        (dict, dict): dictionary minus keys, dictionary of keys
    """
    alg_config = {}
    for hparam in keys:
        if hparam in config:
            val = config.pop(hparam)
            alg_config[hparam] = val
    return config, alg_config

def setup_training_parser():

    parser = argparse.ArgumentParser(add_help=False)
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
        "--config",
        default="./configs/simple-transformer-invsqrt.json", 
        type=str,
        help="Specify json config file.",
    )
    parser.add_argument(
        "--seed",
        default=0,
        type=int,
        help="Random seed. (int, default = 0)",
    )
    parser.add_argument(
        "--n_epochs",
        default=50,
        type=int,
        help="Number of epochs to run the training. (int, default = 50)",
    )
    parser.add_argument(
        "--batch_size",
        default=20,
        type=int,
        help="Batch size for mini-batch training. (int, default = 20)",
    )
    parser.add_argument(
        "--eval_batch_size",
        default=10,
        type=int,
        help="Batch size for mini-batch training. (int, default = 20)",
    )
    parser.add_argument(
        "--seq_len",
        default=35,
        type=int,
        help="Max length of a sequence. (int, default = 35)",
    )
    # parser.add_argument(
    #     "--lr",
    #     default=5.0,
    #     type=float,
    #     help="Learning rate. (float, default = 1e-4)",
    # )
    parser.add_argument(
        "--save_model",
        action="store_true",
        help="Save model when done.",
    )
    parser.add_argument(
        "--dryrun",
        action="store_true",
        help="Run in dryrun mode without wandb.",
    )

    return parser


def setup_wandb_sweep_parser(parent_parser):
    """Setup parser including wandb sweep hparams. Use this when using sweep from wandb, otherwise
    this will override the config file settings.

    Args:
        parent_parser (ArgumentParser): Parent parser to add onto.

    Returns:
        ArgumentParser: parser with arguments for wandb sweep added
    """
    parser = argparse.ArgumentParser(parents=[parent_parser])

    parser.add_argument("--warmup_steps", default=4000, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--num_layers", default=2, type=int)
    parser.add_argument("--nhead", default=2, type=int)
    parser.add_argument("--d_model", default=200, type=int)
    parser.add_argument("--dim_feedforward", default=200, type=int)

    return parser


SWEEP_PARAMS = [
    "warmup_steps",
    "dropout",
    "num_layers",
    "nhead",
    "d_model",
    "dim_feedforward"
]