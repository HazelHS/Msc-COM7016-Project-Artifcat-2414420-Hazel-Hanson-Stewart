
# Add the dedpendancies to the system path
import sys
from pathlib import Path

# Add the project root directory to sys.path
root_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_path))

# Add the xLSTM_TS_Model directory to sys.path
train_path = Path(__file__).resolve().parent.parent / "AI_Modules" / "xLSTM_TS_Model"
sys.path.insert(0, str(train_path))

import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, precision_score, recall_score, f1_score
from Train import train_and_return_model
import numpy as np

def calculate_mase(y_true, y_pred, y_train):
    mae = mean_absolute_error(y_true, y_pred)
    if y_train is not None and len(y_train) > 1:
        naive_errors = np.abs(np.diff(y_train))
        naive_mae = np.mean(naive_errors)
        if naive_mae < 1e-10:
            return mae
        return mae / naive_mae
    else:
        if len(y_true) > 1:
            naive_errors = np.abs(np.diff(y_true))
            naive_mae = np.mean(naive_errors)
            if naive_mae < 1e-10:
                return mae
            return mae / naive_mae
        else:
            return mae

def evaluate_and_plot(model, test_loader, target_scaler, device, y_train=None, dataset_name="2015-2025_dataset_denoised.csv"):
    model.eval()
    predictions = []
    actuals = []
    import torch
    with torch.no_grad():
        for batch in test_loader:
            X, y = batch
            X = X.to(device)
            y = y.to(device)
            output = model(X)
            predictions.append(output.cpu().numpy())
            actuals.append(y.cpu().numpy())
    predictions = np.concatenate(predictions)
    actuals = np.concatenate(actuals)
    predictions_1d = predictions[:, 0]
    actuals_1d = actuals[:, 0]
    predictions_1d = target_scaler.inverse_transform(predictions_1d.reshape(-1, 1)).flatten()
    actuals_1d = target_scaler.inverse_transform(actuals_1d.reshape(-1, 1)).flatten()

    # Classification metrics (directional)
    binary_pred = np.diff(predictions_1d) > 0
    binary_true = np.diff(actuals_1d) > 0
    accuracy = accuracy_score(binary_true, binary_pred)
    precision = precision_score(binary_true, binary_pred, zero_division=0)
    recall = recall_score(binary_true, binary_pred, zero_division=0)
    f1 = f1_score(binary_true, binary_pred, zero_division=0)

    # Regression metrics
    mae = mean_absolute_error(actuals_1d, predictions_1d)
    rmse = np.sqrt(mean_squared_error(actuals_1d, predictions_1d))
    mase = calculate_mase(actuals_1d, predictions_1d, y_train)

    # Print metrics
    print(f"\nEvaluation Metrics for {dataset_name}:")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MASE: {mase:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # --- Plotting in Evaluate.py style ---
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(2, 3)
    ax_class = fig.add_subplot(gs[0, :2])
    ax_error = fig.add_subplot(gs[0, 2])
    ax_forecast = fig.add_subplot(gs[1, :2])
    ax_mase = fig.add_subplot(gs[1, 2])

    # 1. Classification Metrics
    class_metrics = {
        'Accuracy': accuracy * 100,
        'Precision': precision * 100,
        'Recall': recall * 100,
        'F1 Score': f1 * 100
    }
    ax_class.bar(class_metrics.keys(), class_metrics.values(), color='royalblue')
    ax_class.set_ylim(0, 100)
    ax_class.set_title('Classification Metrics')
    ax_class.set_ylabel('Score')
    ax_class.grid(axis='y', alpha=0.3)
    for i, value in enumerate(class_metrics.values()):
        ax_class.annotate(f"{value:.1f}%", xy=(i, value + 1), ha='center', fontsize=9)

    # 2. Regression Metrics
    reg_metrics = {'MAE': mae, 'RMSE': rmse}
    bars = ax_error.bar(list(reg_metrics.keys()), list(reg_metrics.values()), color='green')
    max_error = max(list(reg_metrics.values()))
    ax_error.set_ylim(0, max_error * 1.2)
    ax_error.set_title('Regression Metrics')
    ax_error.set_ylabel('Error Rate')
    ax_error.grid(axis='y', alpha=0.3)
    for i, value in enumerate(reg_metrics.values()):
        ax_error.annotate(f"{value:.4f}", xy=(i, value + max_error * 0.05), ha='center', fontsize=9)

    # 3. Predictions vs Actuals
    sample_size = min(100, len(actuals_1d))
    sample_indices = np.linspace(0, len(actuals_1d)-1, sample_size, dtype=int)
    ax_forecast.plot(sample_indices, actuals_1d[sample_indices], 'b-', label='Actual')
    ax_forecast.plot(sample_indices, predictions_1d[sample_indices], 'r--', label='Predicted')
    ax_forecast.set_title(f'Predictions vs Actuals (Sample) - {dataset_name}')
    ax_forecast.set_xlabel('Time')
    ax_forecast.set_ylabel('Value')
    ax_forecast.legend()
    ax_forecast.grid(alpha=0.3)

    # 4. MASE bar
    safe_mase = min(mase, 10)
    ax_mase.bar(['MASE'], [safe_mase], color='orange' if safe_mase > 1 else 'green')
    ax_mase.set_ylim(0, max(2.5, safe_mase * 1.2))
    ax_mase.axhspan(0, 1, alpha=0.2, color='green', label='Good (MASE < 1)')
    ax_mase.axhspan(1, 2, alpha=0.2, color='yellow', label='Fair (1 < MASE < 2)')
    ax_mase.axhspan(2, 10, alpha=0.2, color='red', label='Poor (MASE > 2)')
    ax_mase.annotate(f"{mase:.4f}", xy=(0, safe_mase + 0.1), ha='center', fontsize=9)
    ax_mase.set_title('Time Series Specific Metrics')
    ax_mase.set_ylabel('Score')
    ax_mase.legend()

    fig.text(0.5, 0.02, 'Note: For classification metrics, higher is better. For error metrics, lower is better. MASE < 1 means better than naive forecast.',
             ha='center', fontsize=10)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    model, test_loader, target_scaler, device = train_and_return_model()
    # If you have y_train available, pass it here; otherwise, set y_train=None
    evaluate_and_plot(model, test_loader, target_scaler, device, y_train=None)