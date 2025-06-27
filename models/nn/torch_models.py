#!/Users/jerald/opt/miniconda3/envs/llm/bin/python

import pandas as pd 
import numpy as np 
import sys 

# Torch Imports 
import torch
from torch import nn 
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

device = ("cuda" if torch.cuda.is_available() else "cpu")

# Basic Neural Network model
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim=43, hidden_dim=128):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.linear_relu_stack(x)

# LSTM model
class LSTM(nn.Module):
    def __init__(self, input_dim=43, hidden_dim=64, seq_len=5):
        super().__init__()
        self.seq_len = seq_len
        self.feature_dim = input_dim // seq_len  # Split features into sequence
        self.lstm = nn.LSTM(input_size=self.feature_dim, hidden_size=hidden_dim, 
                           num_layers=1, batch_first=True, dropout=0.0)  # Simplify to 1 layer
        self.linear = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Reshape: (batch, 43) -> (batch, seq_len, feature_dim)
        if len(x.shape) == 2:
            batch_size = x.size(0)
            x = x[:, :self.seq_len * self.feature_dim].view(batch_size, self.seq_len, self.feature_dim)
        lstm_out, _ = self.lstm(x)
        return self.linear(lstm_out[:, -1, :])

# Basic Recurrent Neural Network model
class RNN(nn.Module):
    def __init__(self, input_dim=43, hidden_dim=64, seq_len=5):
        super().__init__()
        self.seq_len = seq_len
        self.feature_dim = input_dim // seq_len
        self.rnn = nn.RNN(input_size=self.feature_dim, hidden_size=hidden_dim, 
                         num_layers=1, batch_first=True, dropout=0.0)
        self.linear = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        if len(x.shape) == 2:
            batch_size = x.size(0)
            x = x[:, :self.seq_len * self.feature_dim].view(batch_size, self.seq_len, self.feature_dim)
        rnn_out, _ = self.rnn(x)
        return self.linear(rnn_out[:, -1, :])

# Simplified Convolutional Neural Network model
class CNN(nn.Module):
    def __init__(self, input_dim=43):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(4)  # Remove aggressive pooling
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 4, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# Ensemble Model with Independent Training
class TorchModels(nn.Module):
    def __init__(self, input_dim=43):
        super().__init__()
        self.neural_network = NeuralNetwork(input_dim=input_dim)
        self.lstm = LSTM(input_dim=input_dim)
        self.rnn = RNN(input_dim=input_dim)
        self.cnn = CNN(input_dim=input_dim)

    def forward(self, x):
        nn_out = self.neural_network(x)
        lstm_out = self.lstm(x)
        rnn_out = self.rnn(x)
        cnn_out = self.cnn(x)
        combined = (nn_out + lstm_out + rnn_out + cnn_out) / 4
        return combined, nn_out, lstm_out, rnn_out, cnn_out

    def fit(self, x_train, y_train, x_val, y_val, epochs=100, batch_size=128, patience=10):
        self.train()
        dataset = torch.utils.data.TensorDataset(x_train, y_train)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Class weights: boost Class 0
        class_counts = np.bincount(y_train.cpu().numpy().astype(int).flatten())
        class_weights = torch.tensor([class_counts[1] / sum(class_counts) * 2.0, class_counts[0] / sum(class_counts)], dtype=torch.float32).to(device)

        criterion = nn.BCELoss(reduction='mean')
        optimizers = {
            'nn': torch.optim.Adam(self.neural_network.parameters(), lr=0.001, weight_decay=1e-4),
            'lstm': torch.optim.Adam(self.lstm.parameters(), lr=0.001, weight_decay=1e-4),
            'rnn': torch.optim.Adam(self.rnn.parameters(), lr=0.001, weight_decay=1e-4),
            'cnn': torch.optim.Adam(self.cnn.parameters(), lr=0.001, weight_decay=1e-4)
        }

        best_val_f1 = 0
        epochs_no_improve = 0
        best_model_state = None

        for epoch in range(epochs):
            total_loss = {'nn': 0, 'lstm': 0, 'rnn': 0, 'cnn': 0, 'combined': 0}
            num_batches = 0
            
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                weights = torch.ones_like(batch_y)
                weights[batch_y == 0] = class_weights[0]
                weights[batch_y == 1] = class_weights[1]

                # Train each model independently
                for model_name, optimizer in optimizers.items():
                    optimizer.zero_grad()
                    combined, nn_out, lstm_out, rnn_out, cnn_out = self(batch_x)
                    outputs = {
                        'nn': nn_out,
                        'lstm': lstm_out,
                        'rnn': rnn_out,
                        'cnn': cnn_out,
                        'combined': combined
                    }[model_name]
                    
                    loss = criterion(outputs, batch_y) * weights
                    loss = loss.mean()
                    loss.backward()
                    optimizer.step()
                    
                    total_loss[model_name] += loss.item()
                
                num_batches += 1

            avg_loss = {k: v / num_batches for k, v in total_loss.items()}

            # Validation
            self.eval()
            with torch.no_grad():
                val_combined, val_nn, val_lstm, val_rnn, val_cnn = self(x_val)
                val_pred_labels = (val_combined.cpu().numpy() > 0.5).astype(int)
                val_f1 = f1_score(y_val.cpu().numpy(), val_pred_labels)
                val_acc = accuracy_score(y_val.cpu().numpy(), val_pred_labels)

                nn_pred = (val_nn.cpu().numpy() > 0.5).astype(int)
                lstm_pred = (val_lstm.cpu().numpy() > 0.5).astype(int)
                rnn_pred = (val_rnn.cpu().numpy() > 0.5).astype(int)
                cnn_pred = (val_cnn.cpu().numpy() > 0.5).astype(int)

            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss['combined']:.4f}, Val F1: {val_f1:.4f}, Val Acc: {val_acc:.4f}")
            print(f"NN Loss: {avg_loss['nn']:.4f}, Class Dist: {np.bincount(nn_pred.flatten())}")
            print(f"LSTM Loss: {avg_loss['lstm']:.4f}, Class Dist: {np.bincount(lstm_pred.flatten())}")
            print(f"RNN Loss: {avg_loss['rnn']:.4f}, Class Dist: {np.bincount(rnn_pred.flatten())}")
            print(f"CNN Loss: {avg_loss['cnn']:.4f}, Class Dist: {np.bincount(cnn_pred.flatten())}")

            # Early stopping
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_model_state = self.state_dict()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    self.load_state_dict(best_model_state)
                    break

        self.eval()

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            x = x.to(device)
            combined, nn_out, lstm_out, rnn_out, cnn_out = self(x)
        return {
            'combined': combined.cpu().numpy(),
            'nn': nn_out.cpu().numpy(),
            'lstm': lstm_out.cpu().numpy(),
            'rnn': rnn_out.cpu().numpy(),
            'cnn': cnn_out.cpu().numpy()
        }

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[3]))
    from bin.min import Manager
    from bin.stockData import StockData
    import numpy as np
    from sklearn.preprocessing import StandardScaler

    print("Connecting to database...")
    sd = StockData(stock="spy", manager=Manager(), cache_dir="data_cache")
    print("Success. ")
    df = sd.get_features()
    df['y'] = df['close'].pct_change().shift(-1)
    df['y'] = pd.Series(np.where(df['y'] > 0, 1, 0), index=df.index)
    df = df.dropna()
    
    n = 100
    test_ind = df.index[-n:]
    
    x = df.drop(columns=["close", "open", "high", "low", "y"])
    y = df["y"]
    xtest = x.loc[test_ind]
    ytest = y.loc[test_ind]
    x = x.loc[~x.index.isin(test_ind)]
    y = y.loc[~y.index.isin(test_ind)]
    print("X shape:", x.shape, "XTEST shape:", xtest.shape)
    print("Y shape:", y.shape, "YTEST shape:", ytest.shape)
    
    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )
    print("X_train shape:", x_train.shape, "X_val shape:", x_val.shape)
    
    scaler = StandardScaler()
    x_train = torch.tensor(scaler.fit_transform(x_train.values), dtype=torch.float32)
    x_val = torch.tensor(scaler.transform(x_val.values), dtype=torch.float32)
    x_test = torch.tensor(scaler.transform(xtest.values), dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    y_val = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)
    y_test = torch.tensor(ytest.values, dtype=torch.float32).unsqueeze(1)
    
    print("Label distribution (train):", np.bincount(y_train.numpy().astype(int).flatten()))
    
    input_dim = x_train.shape[1]
    model = TorchModels(input_dim=input_dim).to(device)
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_val = x_val.to(device)
    y_val = y_val.to(device)
    x_test = x_test.to(device)
    y_test = y_test.to(device)
    
    model.fit(x_train, y_train, x_val, y_val, epochs=100, batch_size=128, patience=10)
    
    predictions = model.predict(x_test)
    print("Predictions shape (combined):", predictions['combined'].shape)
    
    for model_name in ['combined', 'nn', 'lstm', 'rnn', 'cnn']:
        pred = predictions[model_name]
        pred_labels = (pred > 0.5).astype(int)
        print(f"{model_name.upper()} Predictions:")
        print(f"Raw predictions (min, max, mean): {pred.min():.4f}, {pred.max():.4f}, {pred.mean():.4f}")
        print(f"Class distribution: {np.bincount(pred_labels.flatten())}")
        print(f"Accuracy: {accuracy_score(y_test.cpu().numpy(), pred_labels):.4f}")
        print(f"F1-score: {f1_score(y_test.cpu().numpy(), pred_labels):.4f}\n")
    
    combined_pred = predictions['combined']
    print("Combined Predictions:")
    print(combined_pred)
    combined_labels = np.where(combined_pred > 0.54, 1, -1)
    print("Combined Labels:")
    print(combined_labels)
    from bin.utils.tools import encode_orders
    orders = encode_orders(predictions=combined_labels, test_index=test_ind, stock='spy', shares=10, name="spy_model")
    print(orders.Order.value_counts())
    orders.to_csv("orders.csv", index=True)