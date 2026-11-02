import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseMoEAdapter(nn.Module):
    def __init__(self, dim, num_experts=4, top_k=2):
        """
        Args:
            dim (int): Số chiều của đặc trưng (channel size) đi vào.
            num_experts (int): Tổng số lượng "chuyên gia" (ở đây mặc định là 4 chuyên gia).
            top_k (int): Số lượng chuyên gia được chọn để xử lý cùng một lúc cho mỗi phần dữ liệu (chọn 2).
        """
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        # 1. Mạng định tuyến (Gating Network / Lễ tân): Quyết định chọn chuyên gia nào
        self.gate = nn.Linear(dim, num_experts)
        
        # 2. Danh sách các "chuyên gia" (Experts): Mỗi chuyên gia là một mạng nhỏ (MLP) độc lập
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim * 2),
                nn.GELU(),
                nn.Linear(dim * 2, dim)
            ) for _ in range(num_experts)
        ])

    def forward(self, x):
        # x shape: [Batch, Tokens, Dim] (Đặc trưng đầu vào từ Encoder)
        batch_size, seq_len, dim = x.shape
        
        # Bước 1: Ông lễ tân (Gate) đánh giá xem token nào nên giao cho chuyên gia nào
        gate_logits = self.gate(x)
        weights = F.softmax(gate_logits, dim=-1)
        
        # Bước 2: Chọn ra top_k chuyên gia có điểm số cao nhất cho từng token
        topk_weights, topk_indices = torch.topk(weights, self.top_k, dim=-1)
        
        # Chuẩn hóa lại tổng trọng số của các chuyên gia được chọn cho bằng 1
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-6)
        
        # Bước 3: Tổng hợp kết quả từ các chuyên gia được chỉ định
        out = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            # Kiểm tra xem chuyên gia thứ i có được chọn không
            mask = (topk_indices == i)
            if mask.any():
                expert_out = expert(x)
                for k in range(self.top_k):
                    k_mask = mask[..., k].unsqueeze(-1)
                    out += expert_out * k_mask * topk_weights[..., k:k+1]
                    
        return out