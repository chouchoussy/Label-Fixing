import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from src.pipeline import NoiseCorrectionPipeline
from src.utils import set_seed, plot_confusion_matrix

def main():
    # --- Cấu hình toàn bộ dự án ---
    config = {
        'SEED': 42,
        'BATCH_SIZE': 128,
        'GROUND_TRUTH_PATH': "data/agnews/train/agnews_groundtruth.csv",
        'FEATURES_PATH': "data/agnews/train/agnews_noise/agnews_llm.feather",
        'MODEL_DIMS': [512, 256],
        
        # --- Tham số cho vòng lặp ---
        'NUM_ITERATIONS': 30,
        'MOMENTUM_ALPHA': 0.8,
        'ITERATION_PATIENCE': 3, 

        # --- Tham số cho hybird filtering/correction ---
        'CONFIDENCE_PERCENTILE': 80,
        
        # --- Tham số cho ACT ---
        'ACT_EPOCHS': 150,
        'ACT_WARMUP': 20,
        'ACT_PATIENCE': 15,
        'ACT_LR_RTM': 1e-4,
        'ACT_LR_NTM': 1e-3,
        
        # --- Tham số cho fine-tuning ---
        'FINETUNE_EPOCHS': 100,
        'FINETUNE_LR': 1e-5,
        'FINETUNE_PATIENCE': 7
    }

    # --- Bắt đầu quy trình ---
    set_seed(config['SEED'])

    # 1. Khởi tạo và chạy pipeline
    pipeline = NoiseCorrectionPipeline(config)
    corrected_labels, true_labels, noisy_labels = pipeline.run()

    # 2. Đánh giá kết quả cuối cùng
    print("\n" + "="*50)
    print("--- BẮT ĐẦU ĐÁNH GIÁ KẾT QUẢ CUỐI CÙNG ---")
    print("="*50 + "\n")

    class_names = [str(i) for i in range(len(np.unique(true_labels)))]

    # 2.1. Hiệu suất Phát hiện Nhãn Nhiễu
    print("--- 2.1. Hiệu suất Phát hiện Nhãn Nhiễu (so với nhãn gốc) ---")
    is_noisy_ground_truth = (noisy_labels != true_labels)
    is_corrected_by_model = (noisy_labels != corrected_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        is_noisy_ground_truth, is_corrected_by_model, average='binary', zero_division=0
    )
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("-" * 50)

    # 2.2. Chất lượng bộ dữ liệu
    print("\n--- 2.2. Chất lượng bộ dữ liệu TRƯỚC và SAU khi sửa lỗi ---")
    acc_original = accuracy_score(true_labels, noisy_labels)
    noise_rate_original = 1 - acc_original
    acc_corrected = accuracy_score(true_labels, corrected_labels)
    noise_rate_corrected = 1 - acc_corrected
    print(f"📈 Độ chính xác của bộ dữ liệu GỐC: {acc_original*100:.2f}%")
    print(f"🔥 Tỷ lệ nhiễu (Noise Rate) GỐC: {noise_rate_original*100:.2f}%")
    print("-" * 20)
    print(f"📉 Độ chính xác của bộ dữ liệu SAU KHI SỬA (CUỐI CÙNG): {acc_corrected*100:.2f}%")
    print(f"💧 Tỷ lệ nhiễu (Noise Rate) CÒN LẠI: {noise_rate_corrected*100:.2f}%")
    print("-" * 50)

    # 2.3. Vẽ ma trận nhầm lẫn
    print("\n--- 2.3. Trực quan hóa bằng Ma trận nhầm lẫn ---")
    plot_confusion_matrix(
        true_labels,
        noisy_labels,
        class_names,
        title='Ma trận nhầm lẫn (Nhãn nhiễu GỐC so với Nhãn thật)'
    )
    plot_confusion_matrix(
        true_labels,
        corrected_labels,
        class_names,
        title='Ma trận nhầm lẫn (Nhãn ĐÃ SỬA CUỐI CÙNG so với Nhãn thật)'
    )

    # 3. Tạo và lưu file CSV kết quả
    print("\n" + "="*50)
    print("--- XUẤT KẾT QUẢ RA FILE CSV ---")
    print("="*50 + "\n")
    
    results_df = pd.DataFrame({
        'noisy_label': noisy_labels,
        'fixed_label': corrected_labels,
        'true_label': true_labels
    })
    
    output_filename = 'output/agnews_semi_corrected.csv'
    results_df.to_csv(output_filename, index=True)
    print(f"✅ Đã xuất file '{output_filename}' thành công!")

if __name__ == '__main__':
    main()