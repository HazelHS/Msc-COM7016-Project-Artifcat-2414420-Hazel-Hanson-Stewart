import sys
from pathlib import Path

# Add the project root directory to sys.path
root_path = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_path))

# Add the Run_Model directory to sys.path
run_model_path = Path(__file__).resolve().parent.parent / "Run_Model"
sys.path.insert(0, str(run_model_path))

from Run_Model import sequence_length, prepare_data, setup_model, create_dataloaders, train_model

import pandas as pd

def train_and_return_model():
    # Load the dataset
    data_path = Path(root_path) / "Dataset_Modules" / "dataset_output" / "2015-2025_dataset_denoised.csv"
    df = pd.read_csv(data_path)
    print(f"Loaded data shape: {df.shape}")

    # Set parameters
    target_col = 'BTC/USD'
    sequence_length = 10
    batch_size = 16
    epochs = 200  # <--- CHANGE: Increase from 10 to 200

    # Prepare data
    print("Preparing data...")
    prepared = prepare_data(df, target_col=target_col, sequence_length=sequence_length)
    if prepared is None:
        return None

    train_dataset = prepared['train_dataset']
    val_dataset = prepared['val_dataset']
    test_dataset = prepared['test_dataset']
    feature_scaler = prepared['feature_scaler']
    target_scaler = prepared['target_scaler']
    original_test_actuals = prepared['original_test_actuals']
    n_features = prepared['n_features']

    # Setup model
    print("Setting up model...")
    model_setup = setup_model(n_features, sequence_length=sequence_length)
    model = model_setup['model']
    device = model_setup['device']
    optimizer = model_setup['optimizer']
    scheduler = model_setup['scheduler']

    # Create dataloaders
    print("Creating dataloaders...")
    loaders = create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=batch_size)
    train_loader = loaders['train_loader']
    val_loader = loaders['val_loader']
    test_loader = loaders['test_loader']

    # Train model
    print("Training model...")
    train_model(  # <--- CHANGE: Use train_model instead of train_model_skeleton
        model, train_loader, val_loader, optimizer, scheduler, device,
        epochs=epochs,  # <--- CHANGE: Pass epochs parameter
        early_stopping_patience=30  # <--- CHANGE: Increase patience to 30
    )

    print("Training complete.")
    return model, test_loader, target_scaler, device

if __name__ == "__main__":
    train_and_return_model()