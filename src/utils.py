import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# --- SEED & EARLY STOPPING ---
def set_seed(seed):
    """Thiết lập seed cho tính tái lập."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class EarlyStopper:
    """Dừng sớm quá trình huấn luyện khi loss không cải thiện."""
    def __init__(self, patience=10, min_delta=0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')

    def __call__(self, loss):
        if self.best_loss - loss > self.min_delta:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"🛑 Dừng sớm! Loss không cải thiện trong {self.patience} epochs.")
                return True
        return False

# --- T MATRIX ESTIMATION & EVALUATION ---
def calculate_ground_truth_T(true_labels, noisy_labels, num_classes):
    """Tính toán ma trận T dựa trên các nhãn đã cho."""
    T_matrix = np.zeros((num_classes, num_classes))
    for i in range(num_classes):
        indices = np.where(true_labels == i)[0]
        if len(indices) == 0:
            T_matrix[i, i] = 1.0
            continue
        noisy_subset = noisy_labels[indices]
        for j in range(num_classes):
            T_matrix[i, j] = np.sum(noisy_subset == j) / len(indices)
    return T_matrix

def estimate_T_soft(robust_model_probs, noisy_labels_one_hot, num_classes):
    """Ước tính ma trận chuyển đổi T bằng soft labels."""
    device = robust_model_probs.device
    p_clean = robust_model_probs
    p_noisy_obs = noisy_labels_one_hot
    numerator = torch.matmul(p_clean.T, p_noisy_obs)
    denominator = torch.sum(p_clean, dim=0)
    T_estimated = numerator / (denominator.unsqueeze(1) + 1e-8)
    return T_estimated.cpu().numpy()

# --- VISUALIZATION ---
def evaluate_T_matrix(T_estimated, T_true):
    """Đánh giá ma trận T ước tính bằng MAE và vẽ heatmap."""
    mae = np.mean(np.abs(T_true - T_estimated))
    print(f"📊 Sai số tuyệt đối trung bình (MAE) giữa T_estimated và T_true: {mae:.4f}")
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    sns.heatmap(T_estimated, annot=True, fmt='.2f', cmap='Blues', ax=axes[0])
    axes[0].set_title('Ma trận T Ước tính', fontsize=14)
    axes[0].set_xlabel('Nhãn Nhiễu')
    axes[0].set_ylabel('Nhãn Sạch (Proxy)')
    
    sns.heatmap(T_true, annot=True, fmt='.2f', cmap='Blues', ax=axes[1])
    axes[1].set_title('Ma trận T Ground Truth', fontsize=14)
    axes[1].set_xlabel('Nhãn Nhiễu')
    axes[1].set_ylabel('Nhãn Sạch (Thực tế)')
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, classes, title='Ma trận nhầm lẫn'):
    """Vẽ ma trận nhầm lẫn."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title, fontsize=16)
    plt.ylabel('Nhãn Thật', fontsize=12)
    plt.xlabel('Nhãn Dự đoán/Sửa', fontsize=12)
    plt.show()