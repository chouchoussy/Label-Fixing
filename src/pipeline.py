import torch
import numpy as np

from .data_manager import DataManager
from .trainers import ACTTrainer, CorrectionTrainer
from .losses import ForwardCorrectionLoss
from .utils import EarlyStopper, estimate_T_soft, calculate_ground_truth_T, evaluate_T_matrix

class NoiseCorrectionPipeline:
    """
    Lớp chính điều phối toàn bộ quy trình sửa lỗi nhãn theo phương pháp lặp lại.
    """
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("--- Bước 0: Khởi tạo và Tải dữ liệu ---\n")
        self.data_manager = DataManager(
            ground_truth_path=config['GROUND_TRUTH_PATH'],
            features_path=config['FEATURES_PATH'],
            batch_size=config['BATCH_SIZE']
        )
        self.input_dim = self.data_manager.embeddings.shape[1]
        self.num_classes = self.data_manager.num_classes
        self.noisy_labels_initial = self.data_manager.noisy_labels.copy()

    def _run_single_iteration(self):
        """
        Thực hiện một vòng lặp và trả về nhãn mềm đã sửa CÙNG VỚI loss cuối cùng.
        """
        print("\n--- Bắt đầu huấn luyện ACT và Ước tính T ---")
        act_trainer = ACTTrainer(
            input_dim=self.input_dim,
            num_classes=self.num_classes,
            hidden_dims=self.config['MODEL_DIMS'],
            lr_rtm=self.config.get('ACT_LR_RTM', 1e-4),
            lr_ntm=self.config.get('ACT_LR_NTM', 1e-3)
        )
        act_stopper = EarlyStopper(patience=self.config['ACT_PATIENCE'])
        
        robust_model = act_trainer.train(
            self.data_manager.get_full_dataloader(shuffle=True),
            epochs=self.config['ACT_EPOCHS'],
            warmup_epochs=self.config['ACT_WARMUP'],
            early_stopper=act_stopper
        )
        robust_model.eval()
        all_features = self.data_manager.get_full_dataset().tensors[0].to(self.device)
        current_noisy_soft_labels = self.data_manager.get_full_dataset().tensors[1].to(self.device)
        with torch.no_grad():
            proxy_clean_probs = torch.nn.functional.softmax(robust_model(all_features), dim=1)
        T_estimated = estimate_T_soft(proxy_clean_probs, current_noisy_soft_labels, self.num_classes)
        
        print("\n--- Đánh giá ma trận T trung gian (chỉ để quan sát) ---")
        current_noisy_hard_labels = self.data_manager.noisy_labels
        T_true = calculate_ground_truth_T(self.data_manager.true_labels, current_noisy_hard_labels, self.num_classes)
        evaluate_T_matrix(T_estimated, T_true)
        
        print("\n--- Bắt đầu Fine-tune mô hình với Forward Correction ---")
        final_classifier, final_loss = self._finetune_with_correction(T_estimated, robust_model)
        
        print("\n--- Lấy nhãn mềm đã sửa từ vòng lặp hiện tại ---")
        corrected_soft_labels = self._predict_soft(final_classifier)
        
        return corrected_soft_labels, final_loss

    def _finetune_with_correction(self, T_estimated, model_to_finetune):
        optimizer = torch.optim.Adam(model_to_finetune.parameters(), lr=self.config['FINETUNE_LR'])
        correction_loss_fn = ForwardCorrectionLoss(T=torch.tensor(T_estimated, dtype=torch.float32))
        
        correction_trainer = CorrectionTrainer(model_to_finetune, optimizer, correction_loss_fn, self.device)
        finetune_stopper = EarlyStopper(patience=self.config['FINETUNE_PATIENCE'])
        
        trained_classifier, final_avg_loss = correction_trainer.train(
            self.data_manager.get_full_dataloader(shuffle=True),
            epochs=self.config['FINETUNE_EPOCHS'],
            early_stopper=finetune_stopper
        )
        return trained_classifier, final_avg_loss

    def _predict_soft(self, model):
        model.eval()
        all_features = self.data_manager.get_full_dataset().tensors[0].to(self.device)
        with torch.no_grad():
            final_logits = model(all_features)
            corrected_probs = torch.nn.functional.softmax(final_logits, dim=1).cpu().numpy()
        return corrected_probs

    def run(self):
        """
        Thực thi pipeline học lặp lại để sửa lỗi nhãn.
        """
        best_loss_so_far = float('inf')
        patience_counter = 0
        best_soft_labels_so_far = None 
        current_soft_labels = self.data_manager.y_noisy_tensor.cpu().numpy()

        for i in range(self.config['NUM_ITERATIONS']):
            print("\n" + "="*60)
            print(f"🚀 BẮT ĐẦU VÒNG LẶP SỬA LỖI THỨ {i+1}/{self.config['NUM_ITERATIONS']}")
            print("="*60)

            newly_corrected_soft_labels, iteration_loss = self._run_single_iteration()
            
            print(f"\n📊 Đánh giá cuối vòng lặp {i+1}:")
            print(f"   - Correction Loss tổng thể: {iteration_loss:.4f}")

            alpha = self.config['MOMENTUM_ALPHA']
            updated_soft_labels_for_next_iteration = alpha * newly_corrected_soft_labels + (1 - alpha) * current_soft_labels
            
            if iteration_loss < best_loss_so_far:
                best_loss_so_far = iteration_loss
                best_soft_labels_so_far = updated_soft_labels_for_next_iteration.copy()
                patience_counter = 0
                print(f"🎉 Cải thiện mới! Loss tốt nhất hiện tại: {best_loss_so_far:.4f}. Đã lưu lại bộ nhãn.")
            else:
                patience_counter += 1
                print(f"📉 Không cải thiện loss. Patience: {patience_counter}/{self.config['ITERATION_PATIENCE']}")

            if patience_counter >= self.config['ITERATION_PATIENCE']:
                print(f"\n🛑 Dừng vòng lặp sớm do loss không cải thiện trong {self.config['ITERATION_PATIENCE']} vòng lặp.")
                break
            
            self.data_manager.update_noisy_soft_labels(updated_soft_labels_for_next_iteration)
            current_soft_labels = updated_soft_labels_for_next_iteration

        print("\n" + "="*60)
        print("🎉 ĐÃ HOÀN THÀNH TOÀN BỘ QUY TRÌNH HỌC LẶP LẠI! 🎉")
        print(f"🏆 Correction Loss thấp nhất đạt được: {best_loss_so_far:.4f}")

        print("\n--- Áp dụng bộ lọc Hybrid vào kết quả tốt nhất cuối cùng ---")
        
        final_soft_labels_to_filter = best_soft_labels_so_far if best_soft_labels_so_far is not None else current_soft_labels
        
        confidence_scores = np.max(final_soft_labels_to_filter, axis=1)
        confidence_threshold = np.percentile(confidence_scores, 100 - self.config['CONFIDENCE_PERCENTILE'])
        
        confident_mask = confidence_scores >= confidence_threshold
        num_confident = np.sum(confident_mask)
        
        print(f"   - Ngưỡng tin cậy (dựa trên top {self.config['CONFIDENCE_PERCENTILE']}%): {confidence_threshold:.4f}")
        print(f"   - Số mẫu được sửa đổi cuối cùng: {num_confident}/{len(final_soft_labels_to_filter)}")

        final_corrected_labels = self.noisy_labels_initial.copy()
        confident_hard_labels = np.argmax(final_soft_labels_to_filter[confident_mask], axis=1)
        final_corrected_labels[confident_mask] = confident_hard_labels
        
        print("✅ Đã tạo xong bộ nhãn cuối cùng sau khi lọc.")
        print("="*60)

        return final_corrected_labels, self.data_manager.true_labels, self.noisy_labels_initial