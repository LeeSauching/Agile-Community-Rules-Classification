import argparse
import yaml
import torch
from src.utils import seed_everything,, init_logger
from src.data import load_and_expand, train_val_split, build_dataset
#from src.model import load_model, create_trainer

def train(fold):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to yaml config")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
      cfg = yaml.safe_load(f)

    seed_everything(cfg['seed'])
    logger = init_looger(f"{cfg['run_name'].log}")

    logger.info(f"Loading model: {cfg['model']['path']}")

    df, test_df = load_and_expand(cfg['data']['comp_dir'])
	
