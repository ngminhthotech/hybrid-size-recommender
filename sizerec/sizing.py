"""Chọn size cụ thể để đề xuất.

Vì sao cần bước này: dữ liệu fit là dữ liệu quan sát, người mua TỰ chọn size gần với cơ thể mình.
Một mô hình dự đoán fit (dù AUC tốt) vì thế không đáng tin khi so sánh các size xa nhau cho cùng một
người (ví dụ người nhỏ mặc size 20 hầu như không có trong dữ liệu). Do đó việc đề xuất size dùng một
mô hình kích cỡ có tính đơn điệu, diễn giải được, kết hợp cùng hai nguồn thông tin như mô hình hybrid:

    base_body(x)     : size "nền" ước lượng từ số đo cơ thể (Ridge, data.SizeRegressor)
    cat_coef(c)      : chênh lệch theo danh mục
    item_offset(i)   : sản phẩm i "chạy" to/nhỏ bao nhiêu, học từ các đánh giá 'fit'
    user_base(u)     : size nền của người dùng học từ lịch sử (size đã mua - item_offset, hiệu chỉnh theo nhãn)

    base   = (n_u * user_base(u) + k * base_body(x)) / (n_u + k)       (người dùng mới: n_u = 0)
    target = base + cat_coef(c) + item_offset(i)
    size đề xuất = size có sẵn của sản phẩm gần target nhất.

Mô hình hybrid MF dùng để ước lượng rủi ro không vừa (P(chật), P(vừa), P(rộng)) cho size được chọn.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import SizeRegressor


class SizeModel:
    def fit(self, train: pd.DataFrame, X: np.ndarray, reg: SizeRegressor, k_item: float = 5.0,
            k_user: float = 2.0) -> "SizeModel":
        self.k_user = k_user
        coef = np.array(reg.coef)
        n_body = X.shape[1]
        self.intercept, self.w_body = float(coef[0]), coef[1:1 + n_body].tolist()
        self.cat_coef = {c: float(v) for c, v in zip(reg.cats, coef[1 + n_body:])}
        self.step = float(reg.sd) * 0.5          # hiệu chỉnh khi nhãn small/large
        base = self.base_body(X)
        cc = train["category"].map(self.cat_coef).fillna(0.0).values
        resid = train["size"].values - base - cc
        fit = train["y"].values == 1
        d = pd.DataFrame({"item": train["item_id"].values[fit], "r": resid[fit]})
        g = d.groupby("item")["r"].agg(["sum", "count"])
        self.item_offset = ((g["sum"]) / (g["count"] + k_item)).astype(float).to_dict()   # co về 0 khi ít dữ liệu
        adj = np.select([train["y"].values == 0, train["y"].values == 2], [self.step, -self.step], 0.0)
        io = train["item_id"].map(self.item_offset).fillna(0.0).values
        u = pd.DataFrame({"u": train["user_id"].values, "b": train["size"].values - cc - io + adj})
        gu = u.groupby("u")["b"].agg(["mean", "count"])
        self.user_base = gu["mean"].astype(float).to_dict()
        self.user_n = gu["count"].astype(int).to_dict()
        self.global_base = float(np.median(train["size"].values - cc - io))
        return self

    def base_body(self, X: np.ndarray) -> np.ndarray:
        return self.intercept + X @ np.array(self.w_body)

    def target(self, item_id: str, category: str, X_row: np.ndarray | None, user_id: str | None) -> dict:
        has_body = X_row is not None
        b_body = float(self.base_body(X_row[None, :])[0]) if has_body else self.global_base
        n = self.user_n.get(str(user_id), 0) if user_id else 0
        if n:
            w = n / (n + self.k_user) if has_body else 1.0
            base = w * self.user_base[str(user_id)] + (1 - w) * b_body
        else:
            w, base = 0.0, b_body
        cat, off = self.cat_coef.get(category, 0.0), self.item_offset.get(item_id, 0.0)
        return {"base_body": b_body, "user_base": self.user_base.get(str(user_id)) if n else None,
                "user_weight": w, "category_offset": cat, "item_offset": off, "target": base + cat + off}

    def state(self):
        return {k: getattr(self, k) for k in ["k_user", "intercept", "w_body", "cat_coef", "step", "item_offset",
                                              "user_base", "user_n", "global_base"]}

    @classmethod
    def from_state(cls, s):
        m = cls()
        for k, v in s.items():
            setattr(m, k, v)
        return m


def nearest(sizes, target):
    return min(sizes, key=lambda s: (abs(s - target), s))


def evaluate_size_rec(train: pd.DataFrame, test: pd.DataFrame, X_test: np.ndarray, item_sizes: dict,
                      sm: SizeModel) -> dict:
    """Trên các giao dịch test có nhãn 'fit' (khách đã mặc vừa), kiểm tra size đề xuất có trùng size đó không.
    Đây là chỉ số gần nhất với mục tiêu giảm trả hàng do lệch size.
      popular : size được mua nhiều nhất của sản phẩm (mốc)
      body    : chỉ dùng số đo cơ thể (+ danh mục, độ lệch sản phẩm)
      hybrid  : số đo cơ thể + lịch sử của người dùng (nếu có)"""
    popular = train.groupby("item_id")["size"].agg(lambda x: x.value_counts().index[0]).to_dict()
    fit = np.where(test["y"].values == 1)[0]
    stats = {m: [0, 0, 0] for m in ["popular", "body", "hybrid"]}   # n, exact, within_1
    for j, (uid, item, cat, size) in zip(fit, test[["user_id", "item_id", "category", "size"]].values[fit]):
        sizes = sorted(item_sizes.get(item, []))
        if len(sizes) < 2 or size not in sizes:
            continue
        recs = {"popular": popular.get(item, sizes[len(sizes) // 2]),
                "body": nearest(sizes, sm.target(item, cat, X_test[j], None)["target"]),
                "hybrid": nearest(sizes, sm.target(item, cat, X_test[j], uid)["target"])}
        pos = sizes.index(size)
        for m, rec in recs.items():
            st = stats[m]
            st[0] += 1
            st[1] += rec == size
            st[2] += abs(sizes.index(rec) - pos) <= 1 if rec in sizes else 0
    return {m: {"n": n, "exact": e / max(n, 1), "within_1": w / max(n, 1)} for m, (n, e, w) in stats.items()}
