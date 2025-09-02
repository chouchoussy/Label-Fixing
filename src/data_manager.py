import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

class DataManager:
    """
    Quản lý việc tải, xử lý và tạo DataLoader cho dữ liệu.
    Hỗ trợ cả nhãn cứng ban đầu và nhãn mềm trong quá trình lặp lại.
    """
    def __init__(self, ground_truth_path: str, features_path: str, batch_size=64):
        self.ground_truth_path = ground_truth_path
        self.features_path = features_path
        self.batch_size = batch_size
        
        self.embeddings = None
        self.noisy_labels = None
        self.true_labels = None
        self.num_classes = None
        
        self.X_tensor = None
        self.y_noisy_tensor = None # Sẽ là soft labels (N, C)
        self.y_true_tensor = None

        self._load_and_process_data()
        self._prepare_pytorch_tensors()

    def _load_and_process_data(self):
        """
        Tải, hợp nhất và trích xuất dữ liệu từ các nguồn.
        """
        print("Bắt đầu quá trình tải và hợp nhất dữ liệu...")
        
        df_csv = pd.read_csv(self.ground_truth_path)
        df_csv.rename(columns={'label': 'true_label'}, inplace=True)
        df_feather = pd.read_feather(self.features_path)
        
        df_aligned = pd.concat([df_feather, df_csv[['true_label']]], axis=1)
        print("✅ Dữ liệu đã được hợp nhất.")

        self.noisy_labels = df_aligned['label'].values.astype(int)
        self.true_labels = df_aligned['true_label'].values.astype(int)
        
        embedding_df = df_aligned.drop(columns=['label', 'true_label'])
        self.embeddings = embedding_df.values
        
        print(f"✅ Đã trích xuất dữ liệu thành công.")
        print(f"Tổng số mẫu xử lý: {len(self.embeddings)}")
        print(f"Kích thước embedding (số chiều): {self.embeddings.shape[1]}")

    def _prepare_pytorch_tensors(self):
        """
        Chuẩn bị dữ liệu và chuyển nhãn nhiễu ban đầu sang dạng soft (one-hot).
        """
        if self.num_classes is None:
             self.num_classes = len(np.unique(self.true_labels))
             print(f"Số lượng lớp: {self.num_classes}")
        
        self.X_tensor = torch.tensor(self.embeddings, dtype=torch.float32)
        self.y_true_tensor = torch.tensor(self.true_labels, dtype=torch.long)
        
        noisy_labels_one_hot = np.eye(self.num_classes)[self.noisy_labels]
        self.y_noisy_tensor = torch.tensor(noisy_labels_one_hot, dtype=torch.float32)
        print("✅ Nhãn nhiễu ban đầu đã được chuyển sang dạng soft (one-hot).")

    def update_noisy_soft_labels(self, new_soft_labels: np.ndarray):
        """
        Cập nhật nhãn nhiễu bằng "soft labels" mới cho vòng lặp tiếp theo.
        """
        print("\n🔄 Cập nhật nhãn mềm (soft labels) cho vòng lặp tiếp theo...")
        if new_soft_labels.shape != (len(self.embeddings), self.num_classes):
            raise ValueError(f"Shape của nhãn mềm không chính xác! Expecting {(len(self.embeddings), self.num_classes)}, got {new_soft_labels.shape}")
        
        self.y_noisy_tensor = torch.tensor(new_soft_labels, dtype=torch.float32)
        self.noisy_labels = np.argmax(new_soft_labels, axis=1)
        print("✅ Nhãn mềm đã được cập nhật.")

    def get_full_dataset(self):
        return TensorDataset(self.X_tensor, self.y_noisy_tensor, self.y_true_tensor)

    def get_full_dataloader(self, shuffle=True):
        dataset = self.get_full_dataset()
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, drop_last=True)