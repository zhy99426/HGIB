import os
import torch
from data import load_data
from model import Trainer, HGIB
from parser import parse_args
from utils import set_seed, print_args
from loguru import logger

def main(args):
    print_args(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    if args.seed is not None:
        set_seed(args.seed)
    
    # Load data
    data = load_data(args.data_dir, args.dataset, args.device, args.batch_size)
    
    # Build model
    model = HGIB(data, args.emb_dim, args.threshold, args.beta, args.sigma, args.alpha).to(args.device)
        
    trainer = Trainer(model, data, args)
    
    if args.load_checkpoint:
        # Load Checkpoint
        trainer.evaluate()
    else:
        # Train model
        logger.info("Start training the model")
        trainer.train_model()
    
    
if __name__ == '__main__':
    args = parse_args()
    
    main(args)