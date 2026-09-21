import os
import torch
import yaml
from argparse import ArgumentParser
from cryograph.train import train

def main():
    """Parse the command line, load the config, apply any overrides and train."""
    parser = ArgumentParser()
    parser.add_argument('-c', '--config', default=None)
    parser.add_argument('-g', '--group', default=None)
    parser.add_argument('-r', '--run', default=None)
    parser.add_argument('-s', '--seed', default=None)
    parser.add_argument('-d', '--devices', default=None)
    parser.add_argument('-m', '--model', default=None)
    args = parser.parse_args()

    assert args.config is not None and os.path.exists(args.config), (
        "Must set --config to a valid path.",
    )
    assert args.config[-5:] == ".yaml", (
        "Must set --config to a path to some .yaml file.",
    )
    
    with open(args.config) as stream:
        try:
            config = yaml.safe_load(stream)
            dtype = config['data']['dtype']
            if dtype == "torch.float32":
                config['data']['dtype'] = torch.float32
        except yaml.YAMLError as exc:
            print(exc)

    if args.group is not None:
        config['wandb']['group'] = args.group
    if args.run is not None:
        config['wandb']['run'] = args.run
    if args.seed is not None:
        config['training']['seed'] = int(args.seed)
    if args.devices is not None:
        if args.devices == "auto":
            config['training']['devices'] = "auto"
        elif args.devices == "-1":
            config['training']['devices'] = -1
        else:
            tmp = args.devices.split(',')
            for idx, item in enumerate(tmp):
                tmp[idx] = int(item.strip())
            config['training']['devices'] = tmp
    if args.model is not None:
        config['network']['decoder']['model'] = args.model
    train(config)

if __name__ == "__main__":
    main()