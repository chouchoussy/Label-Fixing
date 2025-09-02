import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from .models import MLP
from .utils import EarlyStopper
from .losses import ForwardCorrectionLoss

class ACTTrainer:
    """
    Triển khai thuật toán Asymmetric Co-Training (ACT).
    """
    def __init__(self, input_dim, num_classes, hidden_dims=[512, 256], lr_rtm=1e-4, lr_ntm=1e-3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes
        
        self.rtm = MLP(input_dim, num_classes, hidden_dims).to(self.device)
        self.ntm = MLP(input_dim, num_classes, hidden_dims).to(self.device)
        
        self.optimizer_rtm = optim.Adam(self.rtm.parameters(), lr=lr_rtm)
        self.optimizer_ntm = optim.Adam(self.ntm.parameters(), lr=lr_ntm)
        
        self.criterion = nn.CrossEntropyLoss()

    def train(self, dataloader, epochs=100, warmup_epochs=20, early_stopper=None):
        print("Bắt đầu huấn luyện ACT...")
        full_dataset = dataloader.dataset
        X_full = full_dataset.tensors[0].to(self.device)
        y_noisy_full_soft = full_dataset.tensors[1].to(self.device)

        for epoch in range(epochs):
            self.rtm.train()
            self.ntm.train()
            
            if epoch < warmup_epochs:
                for features, noisy_labels_soft, _ in dataloader:
                    features, noisy_labels_soft = features.to(self.device), noisy_labels_soft.to(self.device)
                    # Train RTM
                    self.optimizer_rtm.zero_grad()
                    loss_rtm = self.criterion(self.rtm(features), noisy_labels_soft)
                    loss_rtm.backward()
                    self.optimizer_rtm.step()
                    # Train NTM
                    self.optimizer_ntm.zero_grad()
                    loss_ntm = self.criterion(self.ntm(features), noisy_labels_soft)
                    loss_ntm.backward()
                    self.optimizer_ntm.step()
                
                if (epoch + 1) % 5 == 0:
                    print(f"Warmup Epoch [{epoch+1}/{warmup_epochs}]")
            else:
                self.rtm.eval()
                self.ntm.eval()
                with torch.no_grad():
                    preds_rtm_hard = torch.argmax(self.rtm(X_full), dim=1)
                    preds_ntm_hard = torch.argmax(self.ntm(X_full), dim=1)
                
                y_noisy_hard = torch.argmax(y_noisy_full_soft, dim=1)
                agree_mask = (preds_rtm_hard == y_noisy_hard) & (preds_ntm_hard == y_noisy_hard)
                
                mine_mask = torch.zeros_like(agree_mask)
                if epoch < warmup_epochs + (epochs - warmup_epochs) / 2:
                    mine_mask = (preds_rtm_hard != y_noisy_hard) & (preds_ntm_hard == y_noisy_hard)
                
                clean_indices_mask = agree_mask | mine_mask
                num_clean = clean_indices_mask.sum().item()

                if num_clean == 0:
                    print(f"Epoch [{epoch+1}/{epochs}]: Không tìm thấy mẫu sạch, bỏ qua cập nhật RTM.")
                    self.ntm.train()
                    for features, noisy_labels_soft, _ in dataloader:
                        features, noisy_labels_soft = features.to(self.device), noisy_labels_soft.to(self.device)
                        self.optimizer_ntm.zero_grad()
                        loss_ntm = self.criterion(self.ntm(features), noisy_labels_soft)
                        loss_ntm.backward()
                        self.optimizer_ntm.step()
                    continue
                
                clean_dataset = TensorDataset(X_full[clean_indices_mask], y_noisy_full_soft[clean_indices_mask])
                clean_dataloader = DataLoader(clean_dataset, batch_size=dataloader.batch_size, shuffle=True, drop_last=True)
                
                self.rtm.train()
                self.ntm.train()
                
                clean_iter = iter(clean_dataloader)
                for features, noisy_labels_soft, _ in dataloader:
                    features, noisy_labels_soft = features.to(self.device), noisy_labels_soft.to(self.device)
                    try:
                        clean_features, clean_labels_soft = next(clean_iter)
                        self.optimizer_rtm.zero_grad()
                        loss_rtm = self.criterion(self.rtm(clean_features), clean_labels_soft)
                        loss_rtm.backward()
                        self.optimizer_rtm.step()
                    except StopIteration:
                        pass 

                    self.optimizer_ntm.zero_grad()
                    loss_ntm = self.criterion(self.ntm(features), noisy_labels_soft)
                    loss_ntm.backward()
                    self.optimizer_ntm.step()

                if (epoch + 1) % 10 == 0:
                    print(f"ACT Epoch [{epoch+1}/{epochs}], Số mẫu sạch được chọn: {num_clean}/{len(X_full)}")

            if early_stopper:
                self.ntm.eval()
                total_ntm_loss = 0
                with torch.no_grad():
                    eval_loader = DataLoader(full_dataset, batch_size=dataloader.batch_size, shuffle=False)
                    for features, noisy_labels_soft, _ in eval_loader:
                        features, noisy_labels_soft = features.to(self.device), noisy_labels_soft.to(self.device)
                        loss_ntm = self.criterion(self.ntm(features), noisy_labels_soft)
                        total_ntm_loss += loss_ntm.item()
                avg_ntm_loss = total_ntm_loss / len(eval_loader)
                
                print(f"Epoch [{epoch+1}/{epochs}], NTM Loss (for early stopping): {avg_ntm_loss:.4f}")
                if early_stopper(avg_ntm_loss):
                    break
        
        print("Hoàn thành huấn luyện ACT.")
        return self.rtm

class CorrectionTrainer:
    """Lớp chuyên huấn luyện mô hình cuối cùng với Forward Correction."""
    def __init__(self, model, optimizer, loss_fn, device):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device

    def train(self, dataloader, epochs=80, early_stopper=None):
        print(f"Bắt đầu huấn luyện mô hình cuối cùng trong tối đa {epochs} epochs...")
        self.model.train()
        final_avg_loss = 0
        for epoch in range(epochs):
            total_loss = 0
            for features, noisy_labels, _ in dataloader:
                features, noisy_labels = features.to(self.device), noisy_labels.to(self.device)
                self.optimizer.zero_grad()
                clean_logits = self.model(features)
                loss = self.loss_fn(clean_logits, noisy_labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            final_avg_loss = avg_loss
            print(f"Epoch [{epoch+1}/{epochs}], Correction Loss: {avg_loss:.4f}")
            if early_stopper and early_stopper(avg_loss):
                break
        print("✅ Huấn luyện mô hình cuối cùng hoàn tất.")
        return self.model, final_avg_loss