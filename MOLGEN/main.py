import argparse
import time
from parsers.parser import Parser
from parsers.config import get_config, print_cfg
import six, sys
sys.modules['rdkit.six'] = six
from trainer import Trainer
from sampler import Sampler_mol


def main(work_type_args):
    args = Parser().parse()
    config = get_config(args.config, args.seed, args.ckpt)
    if args.print_cfg:
        print_cfg(config)

    # -------- Train --------
    if work_type_args.type == 'train':
        trainer = Trainer(config) 
        config.ckpt = trainer.train(time.strftime('%b%d-%H:%M:%S', time.gmtime()))

    # -------- Generation --------
    elif work_type_args.type == 'sample':
        if config.data.data != 'ZINC250k':
            raise ValueError(f"Unsupported dataset {config.data.data}; only 'ZINC250k' is allowed.")
        Sampler_mol(config).sample()
    else:
        raise ValueError(f'Wrong type : {work_type_args.type}')

if __name__ == '__main__':
    work_type_parser = argparse.ArgumentParser()
    work_type_parser.add_argument('--type', type=str, required=True)
    main(work_type_parser.parse_known_args()[0])
