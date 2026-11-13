"""Mô hình Ordinal Hybrid Matrix Factorization.

    u = m_u * p_u + E(x_u)                                     (user: cộng tác + cơ thể)
    v = q_(item,size) + q_item [+ c_category + c_context + size_z * w + gap * w_gap]   (item [+ sản phẩm])
    t = <u, v> + m_u * b_u + b_(item,size) + b_item + g(x_u) [+ beta * gap]
    gap = (size đặt - size ước lượng từ số đo cơ thể theo danh mục) / sd   (xem data.SizeRegressor)
    P(y <= k) = sigmoid(theta_k - t),  y: small(0) < fit(1) < large(2)

t lớn  -> sản phẩm "rộng" so với người dùng (large); t nhỏ -> "chật" (small).
m_u = 0 với người dùng mới (cold-start) hoặc khi bị ID dropout lúc huấn luyện.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ModelSpec:
    use_id: bool = True        # latent factor cộng tác của người dùng (user_id)
    use_body: bool = True      # số đo cơ thể người dùng tự khai
    use_product: bool = True   # đặc trưng sản phẩm tường minh (category, size, rented_for)
    id_drop: float = 0.1       # xác suất ẩn user_id khi train (chỉ có tác dụng khi use_id và use_body)

    def to_dict(self):
        return asdict(self)


# Các cấu hình dùng trong thực nghiệm. "mf_body_product" là mô hình đề xuất.
SPECS = {
    "mf": ModelSpec(use_id=True, use_body=False, use_product=False, id_drop=0.0),
    "mf_body": ModelSpec(use_id=True, use_body=True, use_product=False),
    "mf_body_product": ModelSpec(use_id=True, use_body=True, use_product=True),
    "body_product": ModelSpec(use_id=False, use_body=True, use_product=True, id_drop=0.0),
    "mf_body_product_nodrop": ModelSpec(use_id=True, use_body=True, use_product=True, id_drop=0.0),
}
PROPOSED = "mf_body_product"
DESCRIPTIONS = {
    "majority": "Luôn dự đoán 'fit' (mốc dưới)",
    "body_lr": "Logistic Regression trên số đo + sản phẩm (thuộc tính tường minh, tuyến tính)",
    "mf": "MF thứ bậc thuần: chỉ latent factor cộng tác (user, item, item-size)",
    "mf_body": "MF + số đo cơ thể tích hợp vào vector ẩn người dùng",
    "mf_body_product": "MF + số đo cơ thể + đặc trưng sản phẩm (MÔ HÌNH ĐỀ XUẤT)",
    "body_product": "Chỉ số đo + sản phẩm, không có latent của người dùng (phi tuyến)",
    "mf_body_product_nodrop": "Mô hình đề xuất nhưng không ID dropout (ablation)",
}


class OrdinalHybridMF(nn.Module):
    def __init__(self, n: dict, d_body: int, spec: ModelSpec, dim: int = 16, hidden: int = 64):
        super().__init__()
        self.spec, self.dim = spec, dim
        emb = lambda k, d: nn.Embedding(n[k] + 1, d, padding_idx=0)   # chỉ số 0 = chưa biết
        self.P, self.bu = emb("user_id", dim), emb("user_id", 1)
        self.Q_is, self.b_is = emb("item_size", dim), emb("item_size", 1)
        self.Q_i, self.b_i = emb("item_id", dim), emb("item_id", 1)
        self.C_cat = emb("category", dim)
        self.C_ctx = emb("context", dim)
        self.w_size = nn.Parameter(torch.zeros(dim))
        self.w_gap = nn.Parameter(torch.zeros(dim))      # tương tác size_gap với vector người dùng
        self.beta_gap = nn.Parameter(torch.tensor(0.0))  # ảnh hưởng trực tiếp của size_gap
        for e in [self.P, self.Q_is, self.Q_i, self.C_cat, self.C_ctx]:
            nn.init.normal_(e.weight, std=0.05)
        for e in [self.bu, self.b_is, self.b_i]:
            nn.init.zeros_(e.weight)
        with torch.no_grad():
            for e in [self.P, self.Q_is, self.Q_i, self.C_cat, self.C_ctx, self.bu, self.b_is, self.b_i]:
                e.weight[0].zero_()
        self.E = nn.Sequential(nn.Linear(d_body, hidden), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden, dim))
        self.g = nn.Linear(d_body, 1)
        self.theta0 = nn.Parameter(torch.tensor(-1.0))
        self.dtheta = nn.Parameter(torch.tensor(1.0))

    # ------------------------------------------------------------------ core
    def thresholds(self):
        return self.theta0, self.theta0 + F.softplus(self.dtheta)

    def terms(self, b: dict) -> dict:
        """Tách điểm t thành các thành phần (phục vụ giải thích)."""
        s = self.spec
        v = self.Q_is(b["item_size"]) + self.Q_i(b["item_id"])
        if s.use_product:
            v = v + self.C_cat(b["category"]) + self.C_ctx(b["context"]) + b["size_z"][:, None] * self.w_size
            if s.use_body:
                v = v + b["gap"][:, None] * self.w_gap
        zero = torch.zeros(v.shape[0], device=v.device)
        out = {"item_bias": self.b_is(b["item_size"])[:, 0] + self.b_i(b["item_id"])[:, 0],
               "user_collab": zero, "body": zero, "size_gap": zero}
        if s.use_product and s.use_body:
            out["size_gap"] = self.beta_gap * b["gap"]
        if s.use_id:
            out["user_collab"] = (self.P(b["user_id"]) * v).sum(-1) + self.bu(b["user_id"])[:, 0]
        if s.use_body:
            out["body"] = (self.E(b["x"]) * v).sum(-1) + self.g(b["x"])[:, 0]
        return out

    def forward(self, b: dict) -> torch.Tensor:
        t = sum(self.terms(b).values())
        th0, th1 = self.thresholds()
        c0, c1 = torch.sigmoid(th0 - t), torch.sigmoid(th1 - t)
        return torch.stack([c0, c1 - c0, 1 - c1], -1).clamp_min(1e-7)

    def l2_terms(self, b: dict) -> torch.Tensor:
        pairs = [(self.Q_is, b["item_size"]), (self.Q_i, b["item_id"]), (self.b_is, b["item_size"]),
                 (self.b_i, b["item_id"])]
        if self.spec.use_id:
            pairs += [(self.P, b["user_id"]), (self.bu, b["user_id"])]
        return sum((e(k) ** 2).sum() for e, k in pairs) / len(b["y"])
