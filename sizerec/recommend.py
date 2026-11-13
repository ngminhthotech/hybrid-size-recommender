"""Gợi ý size cho một sản phẩm và giải thích dự đoán."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from .data import CUP_ORD, FIT_NAMES, body_frame
from .sizing import nearest
from .trainer import Bundle

BODY_ARGS = {  # tên tham số CLI -> trường dữ liệu gốc
    "height": "height", "weight": "weight", "age": "age", "bust_size": "bust size", "bra_size": "bra size",
    "cup": "cup size", "bust": "bust", "waist": "waist", "hips": "hips", "shoe": "shoe size",
}


def body_from_inputs(values: dict) -> tuple[pd.DataFrame, pd.Series]:
    raw = pd.DataFrame([{BODY_ARGS[k]: v for k, v in values.items() if k in BODY_ARGS and v is not None}])
    if "cup size" in raw and str(raw.loc[0, "cup size"]).lower() not in CUP_ORD:
        raise ValueError(f"cup không hợp lệ: {raw.loc[0, 'cup size']} (ví dụ hợp lệ: a, b, c, d, dd/e)")
    body = body_frame(raw)
    bt = pd.Series([values.get("body_type") or "unknown"])
    return body, bt


def recommend(bundle: Bundle, item_id: str, user_id: str | None = None, body_inputs: dict | None = None,
              context: str = "unknown") -> dict:
    item_id = str(item_id)
    if item_id not in bundle.item_sizes:
        raise KeyError(f"Không tìm thấy item_id '{item_id}' trong dữ liệu huấn luyện của mô hình.")
    sizes = bundle.item_sizes[item_id]
    known_user = user_id is not None and str(user_id) in bundle.indexer.maps["user_id"]
    body, bt = body_from_inputs(body_inputs or {})
    rows = pd.DataFrame({
        "user_id": [str(user_id) if user_id else "__new__"] * len(sizes),
        "item_id": [item_id] * len(sizes),
        "size": sizes,
        "category": [bundle.item_category.get(item_id, "unknown")] * len(sizes),
        "context": [context] * len(sizes),
        "body_type": list(bt) * len(sizes),
    })
    rows["item_size"] = rows["item_id"] + "|" + rows["size"].map(lambda s: f"{s:g}")
    body_rep = pd.concat([body] * len(sizes), ignore_index=True)
    b = bundle.batch(rows, body_rep)
    with torch.no_grad():
        probs = bundle.model(b).numpy()
        terms = {k: v.numpy() for k, v in bundle.model.terms(b).items()}
        th0, th1 = [float(t) for t in bundle.model.thresholds()]
    has_body = any(body[c].notna().iloc[0] for c in body.columns) or bt.iloc[0] != "unknown"
    X1 = bundle.body_enc.transform(body, bt)[0]
    tgt = bundle.size_model.target(item_id, rows["category"].iloc[0], X1 if has_body else None,
                                   str(user_id) if known_user else None)
    rec_size = nearest(sizes, tgt["target"])
    ranking = []
    for i, s in enumerate(sizes):
        ranking.append({"size": s, "distance": abs(s - tgt["target"]), "p_small": float(probs[i, 0]),
                        "p_fit": float(probs[i, 1]), "p_large": float(probs[i, 2]),
                        "score_t": float(sum(t[i] for t in terms.values())),
                        "terms": {k: float(v[i]) for k, v in terms.items()}})
    ranking.sort(key=lambda r: (r["distance"], r["size"]))
    best = ranking[0]
    advice = _advice(best, ranking)
    filled = [c for c in body.columns if body[c].notna().iloc[0]]
    return {
        "item_id": item_id, "category": rows["category"].iloc[0],
        "user_id": user_id, "user_status": "đã có lịch sử" if known_user else "người dùng mới (cold-start)",
        "user_history": int(bundle.user_counts.get(str(user_id), 0)) if user_id else 0,
        "body_used": {c: float(body[c].iloc[0]) for c in filled},
        "recommended_size": rec_size, "fit_probability": best["p_fit"],
        "risk_small": best["p_small"], "risk_large": best["p_large"], "advice": advice,
        "size_estimate": tgt,
        "thresholds": {"theta_small": th0, "theta_large": th1}, "ranking": ranking,
    }


def _advice(best: dict, ranking: list) -> str | None:
    """Chỉ khuyên đổi size khi size kế bên (theo hướng khuyên) thật sự an toàn hơn.

    Rủi ro chật ở size đang chọn -> xét size lớn hơn kế tiếp; rủi ro rộng -> xét size nhỏ hơn kế tiếp.
    Size kế bên phải có rủi ro theo hướng đó thấp hơn, và rủi ro lệch tổng (chật + rộng) không cao hơn."""
    by_size = sorted(ranking, key=lambda r: r["size"])
    i = next(k for k, r in enumerate(by_size) if r["size"] == best["size"])
    miss = lambda r: r["p_small"] + r["p_large"]
    if best["p_small"] - best["p_large"] > 0.05 and best["p_small"] > 0.25:
        risk, j, word, side = "p_small", i + 1, "CHẬT", "lớn hơn"
    elif best["p_large"] - best["p_small"] > 0.05 and best["p_large"] > 0.25:
        risk, j, word, side = "p_large", i - 1, "RỘNG", "nhỏ hơn"
    else:
        return None
    level = "khá cao" if best[risk] >= 0.4 else "hơi cao"
    msg = f"Mô hình hybrid thấy rủi ro {word} ở size này {level} ({best[risk]:.0%})."
    if not 0 <= j < len(by_size):
        return msg + f" Sản phẩm không có size {side}, nên giữ size đề xuất."
    nb = by_size[j]
    if nb[risk] < best[risk] and miss(nb) <= miss(best):
        return msg + (f" Nếu phân vân, cân nhắc size {nb['size']:g}: rủi ro {word.lower()} {nb[risk]:.0%}, "
                      f"P(vừa) {nb['p_fit']:.0%}.")
    return msg + (f" Tuy vậy size {side} kế tiếp ({nb['size']:g}) không an toàn hơn "
                  f"(P(vừa) {nb['p_fit']:.0%}), nên vẫn giữ size {best['size']:g}.")


def body_contributions(bundle: Bundle, body_inputs: dict, top: int = 5) -> list:
    """Đóng góp tuyến tính của từng số đo vào điểm t (phần g(x)): dương -> đẩy về 'rộng', âm -> 'chật'."""
    body, bt = body_from_inputs(body_inputs or {})
    x = bundle.body_enc.transform(body, bt)[0]
    w = bundle.model.g.weight.detach().numpy()[0]
    contrib = x * w
    names = bundle.body_enc.feature_names()
    order = np.argsort(-np.abs(contrib))
    return [{"feature": names[i], "value_z": float(x[i]), "contribution": float(contrib[i])}
            for i in order[:top] if abs(contrib[i]) > 1e-6]


TERM_LABELS = {
    "item_bias": "Sản phẩm & size (độ lệch chung của item-size)",
    "user_collab": "Lịch sử cộng tác của người dùng (p_u, b_u)",
    "body": "Số đo cơ thể (E(x), g(x))",
    "size_gap": "Chênh lệch size so với size ước lượng từ số đo",
}
FIT_LABELS = dict(zip(FIT_NAMES, ["chật (small)", "vừa (fit)", "rộng (large)"]))
