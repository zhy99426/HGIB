import torch
import numpy as np
import os

from loguru import logger
from tqdm import tqdm
from .metrics import ndcg, hit
from torch.utils.tensorboard import SummaryWriter

class Trainer:
    def __init__(self, model, data, args):
        self.model = model
        self.data = data
        self.args = args
        self.topk = args.topk
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.best_hr = 0.0 
        self.best_ndcg = 0.0 
        self.best_epoch = 0 
        self.early_stop_patience = args.early_stop  
        self.early_stop_counter = 0
        
        log_dir = os.path.join("./log/", args.dataset, args.log_dir)
        self.writer = SummaryWriter(log_dir=log_dir)

    def train_epoch(self, epoch, pretrain=False):
        self.model.train()
        total_loss = 0
        train_loader = self.data['train_loader']
        with tqdm(total=len(train_loader), desc='Training', unit='batch', leave=False) as pbar:
            for user_indices, pos_indices, neg_indices in train_loader:
                user_indices, pos_indices, neg_indices = user_indices.to(self.args.device), pos_indices.to(self.args.device), neg_indices.to(self.args.device)
                loss = self.model.loss(user_indices, pos_indices, neg_indices) if not pretrain else self.model.pretrain_loss(user_indices, pos_indices, neg_indices)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                pbar.set_description(f'Batch Loss: {loss.item():.4f}')
                pbar.update()
                total_loss += loss.item()
        return total_loss / len(pbar)
    
    def train_model(self):
        num_epochs = self.args.num_epochs
        pbar = tqdm(range(num_epochs), desc='Epoch', unit='epoch', leave=False)
        for epoch in pbar:
            loss = self.train_epoch(epoch)
            self.writer.add_scalar('Loss/Train', loss, epoch)
            pbar.set_description(f'Epoch {epoch+1} total loss: {loss:.4f}')
            pbar.update()
            hr, ndcg = self.evaluate()
            self.writer.add_scalar('HR/Test', hr, epoch)
            self.writer.add_scalar('NDCG/Test', ndcg, epoch)
            
            
            improved = self.save_best_model(epoch, hr, ndcg)
            if improved:
                self.early_stop_counter = 0 
            else:
                self.early_stop_counter += 1 
                if self.early_stop_counter >= self.early_stop_patience:
                    logger.info(f"Early stopping triggered after {self.early_stop_patience} epochs without improvement.")
                    break
            self.writer.add_scalar('Best/HR', self.best_hr, epoch)
            self.writer.add_scalar('Best/NDCG', self.best_ndcg, epoch)
            logger.info(f"Test HR: {hr:.4f}, NDCG: {ndcg:.4f}")

        checkpoint_dir = os.path.join(self.args.checkpoint_dir, self.args.dataset)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        torch.save(self.model.state_dict(), f'{checkpoint_dir}/model.pt')
        self.writer.close()

        self.print_best_test_result()

    def save_best_model(self, epoch, hr, ndcg):

        if  ndcg > self.best_ndcg:
            self.best_hr = hr
            self.best_ndcg = ndcg
            self.best_epoch = epoch + 1

            checkpoint_dir = os.path.join(self.args.checkpoint_dir, self.args.dataset, 'best_model')
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            torch.save(self.model.state_dict(), f'{checkpoint_dir}/best_model.pt')
            logger.info(f"Best model saved at epoch {self.best_epoch} with HR: {self.best_hr:.4f}, NDCG: {self.best_ndcg:.4f}")
            return True 
        return False 

    def print_best_test_result(self):
        logger.info(f"Best Test Result - Epoch: {self.best_epoch}, HR: {self.best_hr:.4f}, NDCG: {self.best_ndcg:.4f}")

    def evaluate(self):
        device = self.args.device
        
        self.model.eval()
        topk_list = []
        with torch.no_grad():
            for user_indices, pos_indices, neg_indices in tqdm(self.data['test_loader'], desc='Test', leave=False):
                user_indices, pos_indices, neg_indices = user_indices.to(device), pos_indices.to(device), neg_indices.to(device)
                scores = self.model.predict(user_indices)
                
                for user_idx in range(user_indices.size(0)):
                    user = user_indices[user_idx].item()
                    train_items = self.data['train_gt'].get(str(user), [])
                    scores[user_idx, train_items] = -np.inf
                
                _, topk_indices = torch.topk(scores, self.topk, dim=1)
                
                for idx, user in enumerate(user_indices):
                    gt_items = np.array(self.data['test_gt'][str(user.item())])
                    topk_items = topk_indices[idx].cpu().numpy()
                    mask = np.isin(topk_items, gt_items)
                    topk_list.append(mask)
        
        topk_list = np.vstack(topk_list) 
        hr_res = hit(topk_list, self.data['test_gt_length']).mean(axis=0)
        ndcg_res = ndcg(topk_list, self.data['test_gt_length']).mean(axis=0)
        hr_topk = hr_res[self.topk-1]
        ndcg_topk = ndcg_res[self.topk-1]
        
        return hr_topk, ndcg_topk