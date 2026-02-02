import os
import time
from tqdm import trange
import numpy as np
import torch
import gc
from collections import defaultdict
#from torch.utils.tensorboard import SummaryWriter

from utils.loader import load_seed, load_device, load_data, load_batch, load_model_params, \
                         load_model_optimizer, load_ema, load_mix_loss_fn, \
                         load_ckpt, load_model_from_ckpt, load_ema_from_ckpt, load_opt_from_ckpt
from utils.logger import Logger, set_log, start_log, train_log, resume_log
from utils.graph_utils import rand_perm
from utils.print_colors import red
from losses import cknna



class Trainer(object):
    def __init__(self, config):
        super(Trainer, self).__init__()
        self.config = config
        self._validate_config()
        self.log_folder_name, self.log_dir, self.ckpt_dir = set_log(self.config)
        self.seed = load_seed(self.config.seed)
        self.device = load_device()

        if self.config.data.data != 'ZINC250k':
            raise ValueError(f"Unsupported dataset {self.config.data.data}; only 'ZINC250k' is allowed.")
        self.train_loader, self.test_loader = load_data(self.config)
        self.params = load_model_params(self.config)

        
    def train(self, ts):
        self.config.exp_name = ts
        self.ckpt = f'{ts}'+f'_{self.config.seed}_' + self.config.train.repa_schedule
        print(red(f'{self.ckpt}'))

        # -------- Load models, optimizers, ema --------
        self.model, self.optimizer, self.scheduler = load_model_optimizer(self.params, self.config.train, self.device)
        self.ema = load_ema(self.model, decay=self.config.train.ema)

        logger = Logger(str(os.path.join(self.log_dir, f'{self.ckpt}.log')), mode='a')
        logger.log(f'{self.ckpt}', verbose=False)
        start_log(logger, self.config)
        train_log(logger, self.config, self.params)

        self.loss_fn = load_mix_loss_fn(self.config) 

        # -------- Training --------
        for epoch in trange(0, (self.config.train.num_epochs), desc = '[Epoch]', position = 1, leave=False):
            train_loss_x = []
            train_repa_x = []
            train_loss_adj = []
            train_repa_adj = []
            
                            
            self.model.train()
            for step_idx, train_b in enumerate(self.train_loader, start=1):
                x, adj, emb, emb_x = load_batch(train_b, self.device)
                self.optimizer.zero_grad()

                loss_subject = (x, adj, emb, emb_x) if not self.config.data.perm_mix else rand_perm(x, adj, emb, emb_x)
                loss, loss_x, loss_adj, repa_x, repa_adj = self.loss_fn(self.model, *loss_subject, 
                                                                        repa_on=self.config.train.repa)    

                
                if self.config.train.repa: 
                    (loss + self.config.train.repa_coef*repa_x + self.config.train.repa_coef*repa_adj).backward()
                else:
                    loss.backward()
                if self.config.train.grad_norm > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.train.grad_norm)
                else:
                    grad_norm = 0
                
                
                self.optimizer.step()
                # -------- EMA update --------
                self.ema.update(self.model.parameters())
                train_loss_x.append(loss_x.item())
                train_loss_adj.append(loss_adj.item())
                train_repa_x.append(repa_x.item())
                train_repa_adj.append(repa_adj.item())
                t_load_start = time.time()

                
            if self.config.train.lr_schedule:
                self.scheduler.step()

        
            mean_train_x = np.mean(train_loss_x)
            mean_train_adj = np.mean(train_loss_adj)
            mean_repa_x = np.mean(train_repa_x)
            mean_repa_adj = np.mean(train_repa_adj)

            # -------- Log losses --------
            logger.log(f'{epoch+1:03d} | {time.time()-t_start:.2f}s | '
                        f'train x: {mean_train_x:.3e} | train adj: {mean_train_adj:.3e} | '
                        f'repa x: {mean_repa_x:.3e} | repa adj: {mean_repa_adj:.3e} | '
                        f'grad_norm: {grad_norm:.2e} |', verbose=False)

            # -------- Save checkpoints --------
            if epoch % self.config.train.save_interval == self.config.train.save_interval-1:
                save_name = f'_{epoch+1}' if epoch < self.config.train.num_epochs - 1 else ''
                torch.save({ 
                    'epoch': epoch,
                    'config': self.config,
                    'params' : self.params,
                    'state_dict': self.model.state_dict(), 
                    'optimizer': self.optimizer.state_dict(),
                    'ema': self.ema.state_dict(),
                    }, f'checkpoints/{self.config.data.data}/{self.ckpt + save_name}.pth')
        print(' ')
        return self.ckpt

    def _validate_config(self):
        """
        Error checking for critical training options to fail fast on bad configs.
        """
        if not hasattr(self.config, 'train'):
            raise ValueError('Config is missing train section.')

        allowed_schedules = {'two-stage', 'exponent', 'None'}
        schedule = getattr(self.config.train, 'repa_schedule', None)
        if schedule not in allowed_schedules:
            raise ValueError(f"Unsupported repa_schedule '{schedule}'. Expected one of {sorted(allowed_schedules)}.")

        save_interval = getattr(self.config.train, 'save_interval', None)
        if save_interval is None or save_interval <= 0:
            raise ValueError('train.save_interval must be a positive integer.')









