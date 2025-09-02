import torch
import torch.nn as nn

class ForwardCorrectionLoss(nn.Module):
    """
    Hàm loss cho Forward Correction, sử dụng KLDivLoss để hoạt động với soft labels.
    """
    def __init__(self, T):
        super(ForwardCorrectionLoss, self).__init__()
        self.T = T.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.loss_fn = nn.KLDivLoss(reduction='batchmean')

    def forward(self, logits, target_soft_labels):
        p_clean = nn.functional.softmax(logits, dim=1)
        p_noisy = torch.matmul(p_clean, self.T)
        log_p_noisy = torch.log(p_noisy.clamp(min=1e-7))
        return self.loss_fn(log_p_noisy, target_soft_labels)