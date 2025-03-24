import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import random_split
from tqdm.auto import tqdm
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def train_sentiment_model(model, train_loader, val_loader, device, 
                         epochs=5, learning_rate=2e-5, weight_decay=0.01):
    
    # Define optimizer and loss function
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
        weight_decay=weight_decay
    )
    criterion = nn.MSELoss()  # Mean Squared Error for regression
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_losses = []
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch in progress_bar:
            # Get inputs and targets
            inputs, targets = batch
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(input_ids, attention_mask)
            
            # Calculate loss
            loss = criterion(outputs, targets)
            train_losses.append(loss.item())
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        avg_train_loss = sum(train_losses) / len(train_losses)
        
        # Validation phase
        model.eval()
        val_losses = []
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            for batch in progress_bar:
                # Get inputs and targets
                inputs, targets = batch
                input_ids = inputs['input_ids'].to(device)
                attention_mask = inputs['attention_mask'].to(device)
                targets = targets.to(device)
                
                # Forward pass
                outputs = model(input_ids, attention_mask)
                
                # Calculate loss
                loss = criterion(outputs, targets)
                val_losses.append(loss.item())
                
                # Store predictions and targets for metrics
                all_preds.extend(outputs.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
                # Update progress bar
                progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        avg_val_loss = sum(val_losses) / len(val_losses)
        
        # Calculate evaluation metrics
        mse = mean_squared_error(all_targets, all_preds)
        mae = mean_absolute_error(all_targets, all_preds)
        
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}")
        
        """
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_sentiment_model.pt")
            print(f"Model saved with Val Loss: {best_val_loss:.4f}")
        """

    return model