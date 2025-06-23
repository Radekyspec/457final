import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

# === Parameters ===
SEQ_LEN = 8
BATCH_SIZE = 8
EPOCHS = 800
LEARNING_RATE = 1e-3
COUNTRY = "United States"
ROLL_YEARS = 5


def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len, :])
        y.append(data[i + seq_len, :])
    return np.array(X), np.array(y)


f = ['co2', 'gdp', 'population']
df = pd.read_csv("owid-co2-with-gdp-updated.csv")
china_df = df[df['country'] == COUNTRY][
    ['year'] + f].dropna().sort_values(
    'year').reset_index(drop=True)

scaler = StandardScaler()
features = china_df[f].values
data_scaled = scaler.fit_transform(features)

X, y = create_sequences(data_scaled, SEQ_LEN)
split_idx = int(len(X) * 0.8)
X_train, y_train = X[:split_idx], y[:split_idx]
X_test, y_test = X[split_idx:], y[split_idx:]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_t = torch.tensor(y_test, dtype=torch.float32).to(device)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t),
                          batch_size=BATCH_SIZE, shuffle=False)


class LSTMModel(nn.Module):
    def __init__(self, input_size=len(f), hidden_size=128, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True)
        self.linear = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.linear(lstm_out[:, -1, :])


model = LSTMModel().to(device)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# all_loss = []
# for epoch in range(EPOCHS):
#     model.train()
#     total_loss = 0
#     for X_b, y_b in train_loader:
#         optimizer.zero_grad()
#         pred = model(X_b)
#         loss = loss_fn(pred, y_b)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#     all_loss.append(total_loss)
#     if (epoch + 1) % 20 == 0:
#         print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.4f}")
# torch.save(model.state_dict(), "model_multivar.pth")
#
# plt.figure(figsize=(10, 5))
# plt.plot(range(1, EPOCHS + 1), all_loss, label="Training Loss")
# plt.title("LSTM Training Loss Over Epoches")
# plt.xlabel("Epoch")
# plt.ylabel("MSE Loss")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

model.load_state_dict(torch.load("model_multivar.pth", map_location=device))
model.eval()

with torch.no_grad():
    y_test_pred_scaled = model(X_test_t).cpu().numpy()

y_test_real = scaler.inverse_transform(y_test)
y_pred_real = scaler.inverse_transform(y_test_pred_scaled)

test_years = china_df['year'].iloc[
             SEQ_LEN + split_idx: SEQ_LEN + split_idx + len(y_test)].values

rmse_per_feature = np.sqrt(
    mean_squared_error(y_test_real, y_pred_real, multioutput='raw_values')
)
mae_per_feature = mean_absolute_error(
    y_test_real, y_pred_real, multioutput='raw_values'
)
mape_per_feature = mean_absolute_percentage_error(
    y_test_real, y_pred_real, multioutput='raw_values'
) * 100

for idx, label in enumerate(["CO₂", 'GDP', 'Population']):
    print(f"{label} RMSE: {rmse_per_feature[idx]:.4f}")
    print(f"{label} MAE: {mae_per_feature[idx]:.4f}")
    print(f"{label} MAPE: {mape_per_feature[idx]:.2f}%")
    plt.figure(figsize=(10, 5))
    plt.plot(test_years, y_test_real[:, idx], label=f'True {label}')
    plt.plot(test_years, y_pred_real[:, idx], '--',
             label=f'Predicted {label}')
    plt.title(f"{COUNTRY} {label} Prediction on Test Set")
    plt.xlabel("Year")
    plt.ylabel(label)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

last_sequence = X_test[-1].copy()  # shape (SEQ_LEN, 3)
rolled_preds = []
with torch.no_grad():
    for _ in range(ROLL_YEARS):
        inp = torch.tensor(last_sequence.reshape(1, SEQ_LEN, -1),
                           dtype=torch.float32).to(device)
        pred_scaled = model(inp).cpu().numpy().flatten()
        rolled_preds.append(pred_scaled)
        last_sequence = np.vstack([last_sequence[1:], pred_scaled])

rolled_preds = np.array(rolled_preds)
rolled_real = scaler.inverse_transform(rolled_preds)
roll_years = np.arange(test_years[-1] + 1, test_years[-1] + 1 + ROLL_YEARS)

plt.figure(figsize=(10, 5))
plt.plot(china_df['year'], china_df['co2'], label='Historical CO₂')
plt.plot(roll_years, rolled_real[:, 0], '--o', label='5-Year Forecast CO₂')
plt.title(f"{COUNTRY} 5-Year Rolling CO₂ Forecast from {test_years[-1]}")
plt.xlabel("Year")
plt.ylabel("CO₂ Emissions")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
