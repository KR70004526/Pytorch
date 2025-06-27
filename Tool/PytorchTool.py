# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 16:29:58 2025

@author: USER
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from EarlyStopping import EarlyStopping

class MAETraining:
    def __init__(self, model, criterion, optimizer):
        self.model     = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device    = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        
    # Model Training Structure
    def train_epoch(self, train_loader):
        self.model.train()
        running_mae = 0.0
        for x, y in train_loader:
            x, y    = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            y_pred  = self.model(x)
            loss    = self.criterion(y_pred, y)
            loss.backward()
            self.optimizer.step()
            running_mae += loss.item()*x.size(0)
        return running_mae / len(train_loader.dataset)
        
    # Model Validation Structure
    def validate_epoch(self, val_loader):
        self.model.eval()
        running_mae  = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(self.device), y.to(self.device)
                y_pred = self.model(x)
                loss = self.criterion(y_pred, y)
                running_mae += loss.item()*x.size(0)
        return running_mae / len(val_loader.dataset)
                
    # Begin Model Train & Validation
    def fit(self, num_epochs, train_loader, val_loader=None, verbose=True, early_stopping=None):
        history = {"train_mae": []}
        if val_loader is not None:
            history["val_mae"] = []
            
        for epoch in range(num_epochs):
            train_mae = self.train_epoch(train_loader)
            history["train_mae"].append(train_mae)
            
            if val_loader is not None:
                val_mae = self.validate_epoch(val_loader)
                history["val_mae"].append(val_mae)
                
            if verbose:
                log = f"[Epoch {epoch:03d}] Train MAE: {train_mae:.4f}"
                if val_loader is not None:
                    log += f" Val Mae: {val_mae:.4f}"
                print(log)
                
            if early_stopping is not None:
                early_stopping(val_mae, self.model)
                if early_stopping.early_stop:
                    if verbose: print("-> Early Stopping Triggered")
                    break
                
        return history

class MAETest:
    def __init__(self, model, criterion):
        self.model     = model
        self.criterion = criterion
        self.device    = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
    
    # Model Evaluation
    def evaluate(self, test_loader):
        self.model.eval()
        running_mae = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(self.device), y.to(self.device)
                y_pred = self.model(x)
                loss = self.criterion(y_pred, y)
                running_mae += loss.item()*x.size(0)
        return running_mae / len(test_loader.dataset)
    
    # Model Prediction
    def predict(self, data_loader, option=None):
        self.model.eval()
        all_preds = []
        with torch.no_grad():
            for batch in data_loader:
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                x = x.to(self.device)
                y_pred = self.model(x)
                all_preds.append(y_pred.detach().cpu())
        
        preds = torch.cat(all_preds, dim=0)
        array = preds.numpy()
        
        if option is None:
            return preds
        elif option == "np":
            return array
        elif option == "pd":
            return pd.DataFrame(array)
        
        raise ValueError(f"Unsupported Option Type: {option!r}")
