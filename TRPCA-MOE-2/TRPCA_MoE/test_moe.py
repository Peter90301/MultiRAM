"""
MoE 模型測試 - 與 Transformer 公平對比
使用相同的數據劃分和預處理
"""
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time
import json
import sys
sys.path.append('TRPCA')

from TRPCA.trpca import NormalizedTransformer

# 設置設備
device = 'cpu'  # 使用 CPU 避免 CUDA 兼容性問題
print(f"使用設備: {device}")

# 載入統一劃分的數據
print("\n📂 載入數據...")
data = np.load('data/train_test_split.npz')
X_train = data['X_train']
X_test = data['X_test']
y_train = data['y_train']
y_test = data['y_test']

print(f"   訓練集: {X_train.shape}")
print(f"   測試集: {X_test.shape}")

# 轉換為 PyTorch tensor
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)

# 創建 DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


def count_parameters(model):
    """計算模型參數量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_model(model, train_loader, num_epochs=1000, lr=0.001, patience=20):
    """訓練模型"""
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    best_loss = float('inf')
    patience_counter = 0
    train_losses = []
    
    print("\n🚀 開始訓練 MoE...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            if isinstance(outputs, dict):
                outputs = outputs['regression_output']
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'moe_best.pth')
        else:
            patience_counter += 1
        
        if (epoch + 1) % 100 == 0:
            print(f"   Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
        
        if patience_counter >= patience:
            print(f"   Early stopping at epoch {epoch+1}")
            break
    
    training_time = time.time() - start_time
    print(f"   訓練完成！耗時: {training_time:.2f} 秒")
    
    # 載入最佳模型
    model.load_state_dict(torch.load('moe_best.pth'))
    return model, training_time


def evaluate_model(model, data_loader, dataset_name=""):
    """評估模型"""
    model.eval()
    predictions = []
    targets = []
    
    start_time = time.time()
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            if isinstance(outputs, dict):
                outputs = outputs['regression_output']
            predictions.extend(outputs.cpu().numpy().flatten())
            targets.extend(y_batch.numpy().flatten())
    
    inference_time = time.time() - start_time
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    mae = mean_absolute_error(targets, predictions)
    mse = mean_squared_error(targets, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(targets, predictions)
    
    print(f"\n📊 {dataset_name} 評估結果:")
    print(f"   MAE:  {mae:.4f}")
    print(f"   MSE:  {mse:.4f}")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   R²:   {r2:.4f}")
    if dataset_name == "測試集":
        print(f"   推理時間: {inference_time:.4f} 秒")
    
    return {
        'mae': float(mae),
        'mse': float(mse),
        'rmse': float(rmse),
        'r2': float(r2),
        'inference_time': float(inference_time)
    }


# 創建模型
print("\n🏗️  創建 MoE 模型...")
model = NormalizedTransformer(
    input_dim=X_train.shape[1],
    hidden_dim=128,
    num_layers=3,
    output_dim=1,
    projection_dim=8,
    num_experts=4,
    top_k=1,
    dropout=0.2
)

n_params = count_parameters(model)
print(f"   模型參數量: {n_params:,}")

# 訓練模型
model, training_time = train_model(model, train_loader, num_epochs=300, patience=20)

# 評估模型
print("\n" + "=" * 70)
print("📈 模型評估")
print("=" * 70)

train_metrics = evaluate_model(model, train_loader, "訓練集")
test_metrics = evaluate_model(model, test_loader, "測試集")

# 計算過擬合程度
overfit_gap = train_metrics['r2'] - test_metrics['r2']

print("\n" + "=" * 70)
print("🔍 過擬合分析")
print("=" * 70)
print(f"   訓練集 R²: {train_metrics['r2']:.4f}")
print(f"   測試集 R²: {test_metrics['r2']:.4f}")
print(f"   差距:      {overfit_gap:.4f}")

if overfit_gap > 0.15:
    print("   ⚠️  警告: 可能存在過擬合！")
elif overfit_gap > 0.10:
    print("   ⚠️  注意: 存在一定程度的過擬合")
else:
    print("   ✅ 泛化能力良好")

# 保存結果
results = {
    'model': 'MoE',
    'parameters': n_params,
    'training_time': training_time,
    'train_metrics': train_metrics,
    'test_metrics': test_metrics,
    'overfit_gap': float(overfit_gap),
    'config': {
        'hidden_dim': 128,
        'num_layers': 3,
        'num_experts': 4,
        'top_k': 1,
        'dropout': 0.2
    }
}

with open('moe_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n✅ 結果已保存到 moe_results.json")
print("=" * 70)
