# Add the dedpendancies to the system path
import sys
from pathlib import Path

# Add the project root directory to sys.path
root_path = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_path))

# Add the xLSTM_TS_Model directory to sys.path
xLSTM_path = Path(__file__).resolve().parent.parent / "xLSTM_TS_Model"
sys.path.insert(0, str(xLSTM_path))

from dependency_checker import *
from xLSTM_TS import xLSTM_TS_Model, SequenceDataset, directional_loss, TrainingProgressTracker

sequence_length=60 # was 10, lopez uses 150

def prepare_data(df, target_col='BTC/USD', sequence_length=sequence_length, split_ratios=(0.7, 0.15, 0.15)):
    """
    Prepare data for training the xLSTM-TS model
    
    Args:
        df: Pandas DataFrame with time series data
        target_col: Column name of the target variable
        sequence_length: Length of input sequences
        split_ratios: Ratios for train/val/test splits
        
    Returns:
        Prepared datasets and scalers
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty")
    
    # Basic checks
    print(f"Input DataFrame shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Create a simple feature if needed
    if len(df.columns) <= 1:
        df['prev_price'] = df[target_col].shift(1)
        df = df.dropna()
        print("Added previous price as a feature")
    
    # Calculate split sizes
    total_rows = len(df)
    train_size = int(total_rows * split_ratios[0])
    val_size = int(total_rows * split_ratios[1])
    
    # Split while maintaining temporal order
    train_df = df[:train_size]
    val_df = df[train_size:train_size + val_size]
    test_df = df[train_size + val_size:]
    
    print(f"Train shape: {train_df.shape}")
    
    # Store original test values
    original_test_actuals = test_df[target_col].values
    
    # Get feature columns
    # feature_columns = [col for col in df.columns if col != target_col]
    feature_columns = [col for col in df.columns if col not in [target_col, 'Unnamed: 0']]
    print(f"Feature columns: {feature_columns}")
    
    # Initialize scalers
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Fit scalers on training data
    feature_scaler.fit(train_df[feature_columns])
    target_scaler.fit(train_df[[target_col]])
    
    # Transform data
    X_train = pd.DataFrame(
        feature_scaler.transform(train_df[feature_columns]), 
        columns=feature_columns, 
        index=train_df.index
    )
    y_train = pd.Series(
        target_scaler.transform(train_df[[target_col]]).ravel(), 
        index=train_df.index
    )
    
    X_val = pd.DataFrame(
        feature_scaler.transform(val_df[feature_columns]), 
        columns=feature_columns, 
        index=val_df.index
    )
    y_val = pd.Series(
        target_scaler.transform(val_df[[target_col]]).ravel(), 
        index=val_df.index
    )
    
    X_test = pd.DataFrame(
        feature_scaler.transform(test_df[feature_columns]), 
        columns=feature_columns, 
        index=test_df.index
    )
    y_test = pd.Series(
        target_scaler.transform(test_df[[target_col]]).ravel(), 
        index=test_df.index
    )
    
    # Store original training data for MASE calculation
    y_train_original = train_df[target_col].values
    
    # Create sequences
    print("\nCreating sequences...")
    sequence_progress = tqdm(total=3, desc="Sequence preparation")
    
    # Training sequences
    X_seq, y_seq = [], []
    for i in range(len(X_train) - sequence_length - 6):
        X_seq.append(X_train.iloc[i:i + sequence_length].values)
        y_seq.append(y_train.iloc[i + sequence_length:i + sequence_length + 7].values.reshape(-1))
    X_train_seq, y_train_seq = np.array(X_seq), np.array(y_seq)
    sequence_progress.update(1)
    
    # Validation sequences
    X_seq, y_seq = [], []
    for i in range(len(X_val) - sequence_length - 6):
        X_seq.append(X_val.iloc[i:i + sequence_length].values)
        y_seq.append(y_val.iloc[i + sequence_length:i + sequence_length + 7].values.reshape(-1))
    X_val_seq, y_val_seq = np.array(X_seq), np.array(y_seq)
    sequence_progress.update(1)
    
    # Test sequences
    X_seq, y_seq = [], []
    for i in range(len(X_test) - sequence_length - 6):
        X_seq.append(X_test.iloc[i:i + sequence_length].values)
        y_seq.append(y_test.iloc[i + sequence_length:i + sequence_length + 7].values.reshape(-1))
    X_test_seq, y_test_seq = np.array(X_seq), np.array(y_seq)
    sequence_progress.update(1)
    sequence_progress.close()
    
    # Create PyTorch datasets
    train_dataset = SequenceDataset(X_train_seq, y_train_seq)
    val_dataset = SequenceDataset(X_val_seq, y_val_seq)
    test_dataset = SequenceDataset(X_test_seq, y_test_seq)
    
    return {
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'test_dataset': test_dataset,
        'feature_scaler': feature_scaler,
        'target_scaler': target_scaler,
        'y_train_original': y_train_original,
        'original_test_actuals': original_test_actuals,
        'n_features': X_train.shape[1]
    }


def setup_model(n_features, sequence_length=sequence_length, embedding_dim=64, learning_rate=1e-4):
    """
    Set up the xLSTM-TS model and training components
    
    Args:
        n_features: Number of input features
        sequence_length: Length of input sequences
        embedding_dim: Dimension of embedding layers
        learning_rate: Initial learning rate
        
    Returns:
        Configured model, optimizer, and scheduler
    """
    # Set device (GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model
    print("\nInitializing model...")
    model = xLSTM_TS_Model(
        input_shape=(sequence_length, n_features),
        embedding_dim=embedding_dim,
        output_size=7  # 7-day forecast
    ).to(device)
    
    # Configure optimizer with paper's parameters
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-7
    )
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10,
        min_lr=1e-7,
        verbose=True
    )
    
    return {
        'model': model,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'device': device
    }


def create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=16):
    """
    Create DataLoader objects for the datasets
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        batch_size: Batch size for training
        
    Returns:
        Dictionary of DataLoaders
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,  # Maintain temporal order
        drop_last=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )
    
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader
    }


def train_model(model, train_loader, val_loader, optimizer, scheduler, device, 
                        epochs=200, early_stopping_patience=30):

    """
    Training loop for the xLSTM-TS model
    
    Args:
        model: The PyTorch model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        optimizer: Optimizer for training
        scheduler: Learning rate scheduler
        device: Device to train on (cuda/cpu)
        epochs: Number of epochs to train
        early_stopping_patience: Number of epochs without improvement before stopping
    """
    # Initialize progress tracker
    progress_tracker = TrainingProgressTracker(epochs, len(train_loader))
    
    # Early stopping variables
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_model_state = None
    
    # Gradient clipping max norm
    max_grad_norm = 1.0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_mae = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            # Move data to device
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            output = model(data)
            loss = directional_loss(target, output)  # <--- Use directional_loss
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            optimizer.step()
            
            # Track metrics
            train_loss += loss.item()
            train_mae += F.l1_loss(output, target).item()
            
            # Update progress tracker
            progress_tracker.update_batch(loss.item(), F.l1_loss(output, target).item())
        
        # Calculate average training metrics
        train_loss /= len(train_loader)
        train_mae /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += directional_loss(target, output).item()  # <--- Use directional_loss
                val_mae += F.l1_loss(output, target).item()
        
        val_loss /= len(val_loader)
        val_mae /= len(val_loader)
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            best_model_state = model.state_dict().copy()
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_patience:
                print(f"\nEarly stopping after {epoch+1} epochs")
                break
        
        # Update learning rate scheduler
        scheduler.step(val_loss)
        
        # Update progress tracker for epoch
        progress_tracker.update_epoch(train_loss, val_loss, train_mae)
    
    # Close progress tracker
    progress_tracker.close()
    
    # Load best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\nLoaded best model with validation loss: {best_val_loss:.6f}")
    
    return model


def use_xLSTM_TS_model(df, target_col='BTC/USD', sequence_length=sequence_length, batch_size=16, epochs=2):
    """
    Main function to set up and prepare the xLSTM-TS model (without execution)
    
    Args:
        df: DataFrame with time series data
        target_col: Target column name
        sequence_length: Length of input sequences
        batch_size: Batch size for training
        epochs: Number of epochs for training
        
    Returns:
        Prepared model and data
    """
    print("\nInitializing xLSTM-TS model preparation...")
    
    # Prepare data
    data_prep = prepare_data(df, target_col, sequence_length)
    
    # Create dataloaders
    dataloaders = create_dataloaders(
        data_prep['train_dataset'],
        data_prep['val_dataset'],
        data_prep['test_dataset'],
        batch_size
    )
    
    # Set up model
    model_setup = setup_model(
        data_prep['n_features'],
        sequence_length
    )
    
    print("\nModel preparation complete. Ready for training.")
    
    return {
        'model': model_setup['model'],
        'optimizer': model_setup['optimizer'],
        'scheduler': model_setup['scheduler'],
        'device': model_setup['device'],
        'dataloaders': dataloaders,
        'feature_scaler': data_prep['feature_scaler'],
        'target_scaler': data_prep['target_scaler'],
        'y_train_original': data_prep['y_train_original'],
        'original_test_actuals': data_prep['original_test_actuals']
    }
