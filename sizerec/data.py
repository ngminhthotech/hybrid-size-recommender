"""Đọc, làm sạch và mã hoá dữ liệu ModCloth / RentTheRunway.

Ba nhóm thông tin được tách riêng để đúng thiết kế thực nghiệm MF / MF+Body / MF+Body+Product:
  * Cộng tác (collaborative): user_id, item_id, (item_id, size)
  * Cơ thể (body, người dùng tự khai): height, weight, age, bust/bra, cup, waist, hips, shoe, BMI, body type
  * Sản phẩm (product): category, size (số), rented_for (chỉ RTR, ngữ cảnh thuê)

Các trường chỉ có SAU khi mua (review_text, review_summary, rating, quality, length) bị loại bỏ
để tránh rò rỉ thông tin (data leakage).
"""

from __future__ import annotations

import gzip
import json
import os
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

FIT_MAP = {"small": 0, "fit": 1, "large": 2}
FIT_NAMES = ["small", "fit", "large"]
CUP_ORD = {"aa": 0, "a": 1, "b": 2, "c": 3, "d": 4, "d+": 4.5, "dd": 5, "dd/e": 5, "ddd/e": 5.5,
           "ddd/f": 6, "f": 6, "dddd/g": 7, "g": 7, "h": 8, "i": 9, "j": 10, "k": 11}
LEAKY_FIELDS = ["review_text", "review_summary", "rating", "quality", "length"]


# ----------------------------------------------------------------------------- parse helpers
def _isnull(v) -> bool:
    return v is None or (isinstance(v, float) and np.isnan(v))


def parse_height(v) -> float:
    """'5ft 6in', 5' 6", '165cm', 66 -> inch."""
    if _isnull(v):
        return np.nan
    s = str(v).lower().strip()
    m = re.search(r"(\d+(?:\.\d+)?)\s*cm", s)
    if m:
        return float(m.group(1)) / 2.54
    m = re.search(r"(\d+)\s*(?:ft|')\s*(\d+(?:\.\d+)?)?", s)
    if m:
        return 12.0 * int(m.group(1)) + (float(m.group(2)) if m.group(2) else 0.0)
    m = re.search(r"\d+(?:\.\d+)?", s)
    return float(m.group(0)) if m else np.nan


def parse_weight(v) -> float:
    """'137lbs', '62kg', 137 -> lbs."""
    if _isnull(v):
        return np.nan
    s = str(v).lower()
    m = re.search(r"\d+(?:\.\d+)?", s)
    if not m:
        return np.nan
    x = float(m.group(0))
    return x * 2.20462 if "kg" in s else x


def parse_num(v) -> float:
    if _isnull(v):
        return np.nan
    m = re.search(r"\d+(?:\.\d+)?", str(v))
    return float(m.group(0)) if m else np.nan


def parse_bust_size(v):
    """'34d' -> (34, 4)."""
    if _isnull(v):
        return np.nan, np.nan
    s = str(v).lower().strip()
    m = re.match(r"^(\d+)\s*([a-z/+]*)$", s)
    if not m:
        return np.nan, np.nan
    return float(m.group(1)), CUP_ORD.get(m.group(2), np.nan)


def detect_dataset(path: str) -> str:
    p = os.path.basename(path).lower()
    if "modcloth" in p:
        return "modcloth"
    if "runway" in p or "rtr" in p:
        return "rtr"
    return os.path.splitext(os.path.splitext(p)[0])[0]


# ----------------------------------------------------------------------------- loading
def read_json_lines(path: str) -> pd.DataFrame:
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as f:
        return pd.DataFrame([json.loads(line) for line in f if line.strip()])


def body_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Trích các số đo cơ thể dạng số từ bản ghi thô (dùng chung cho train và cho khách mới)."""
    col = lambda c: raw[c] if c in raw else pd.Series(np.nan, index=raw.index)
    num = pd.DataFrame(index=raw.index)
    num["height"] = col("height").map(parse_height)
    num["weight"] = col("weight").map(parse_weight)
    num["age"] = col("age").map(parse_num)
    band_cup = col("bust size").map(parse_bust_size)
    num["bust_band"] = [b for b, _ in band_cup]
    num["cup"] = [c for _, c in band_cup]
    if "bra size" in raw:
        num["bust_band"] = num["bust_band"].fillna(raw["bra size"].map(parse_num))
    if "cup size" in raw:
        num["cup"] = num["cup"].fillna(raw["cup size"].map(lambda x: CUP_ORD.get(str(x).lower(), np.nan)))
    num["bust"] = col("bust").map(parse_num)
    num["waist"] = col("waist").map(parse_num)
    num["hips"] = col("hips").map(parse_num)
    num["shoe"] = col("shoe size").map(parse_num)
    num["bmi"] = num["weight"] * 703.0 / num["height"] ** 2
    num.loc[(num["height"] < 48) | (num["height"] > 84), "height"] = np.nan
    num.loc[~num["bmi"].between(12, 60), "bmi"] = np.nan
    num.loc[num["shoe"] > 15, "shoe"] = np.nan
    return num.astype(float)


@dataclass
class FitData:
    name: str
    df: pd.DataFrame            # cột chuẩn hoá: user_id, item_id, item_size, size, category, context, body_type, y
    body: pd.DataFrame          # số đo cơ thể dạng số
    body_cols: list = field(default_factory=list)

    def __len__(self):
        return len(self.df)


def load_dataset(path: str) -> FitData:
    raw = read_json_lines(path)
    raw = raw[raw["fit"].isin(FIT_MAP)].reset_index(drop=True)
    df = pd.DataFrame({
        "user_id": raw["user_id"].astype(str),
        "item_id": raw["item_id"].astype(str),
        "size": pd.to_numeric(raw["size"], errors="coerce").fillna(0.0),
        "category": raw["category"].fillna("unknown").astype(str) if "category" in raw else "unknown",
        "context": raw["rented for"].fillna("unknown").astype(str) if "rented for" in raw else "unknown",
        "body_type": raw["body type"].fillna("unknown").astype(str) if "body type" in raw else "unknown",
        "y": raw["fit"].map(FIT_MAP).astype(int),
    })
    df["item_size"] = df["item_id"] + "|" + df["size"].map(lambda s: f"{s:g}")
    body = body_frame(raw)
    keep = [c for c in body.columns if body[c].notna().mean() > 0.01]   # bỏ cột gần như trống
    return FitData(detect_dataset(path), df, body[keep], keep)


# ----------------------------------------------------------------------------- encoders
class BodyEncoder:
    """z-score + chỉ báo thiếu dữ liệu + one-hot body type. Fit chỉ trên tập train."""

    def fit(self, body: pd.DataFrame, body_type: pd.Series) -> "BodyEncoder":
        self.cols = list(body.columns)
        self.mu = {c: float(body[c].mean()) if body[c].notna().any() else 0.0 for c in self.cols}
        self.sd = {c: float(body[c].std()) if body[c].notna().sum() > 1 and body[c].std() > 0 else 1.0
                   for c in self.cols}
        # chỉ báo thiếu chỉ cho cột thực sự hay thiếu (>= 2%); cột hiếm khi thiếu -> thiếu thì coi như giá trị trung bình
        self.miss_cols = [c for c in self.cols if body[c].isna().mean() >= 0.02]
        cats = sorted(body_type.unique())
        self.types = cats if len(cats) > 1 else []
        return self

    @property
    def dim(self) -> int:
        return len(self.cols) + len(self.miss_cols) + len(self.types)

    def feature_names(self) -> list:
        return [f"{c}" for c in self.cols] + [f"{c}_missing" for c in self.miss_cols] + \
               [f"body_type={t}" for t in self.types]

    def transform(self, body: pd.DataFrame, body_type: pd.Series) -> np.ndarray:
        body = body.reindex(columns=self.cols)
        z = np.stack([((body[c] - self.mu[c]) / self.sd[c]).fillna(0.0).clip(-5, 5).values
                      for c in self.cols], 1) if self.cols else np.zeros((len(body), 0))
        miss = body[self.miss_cols].isna().values.astype(float) if self.miss_cols else np.zeros((len(body), 0))
        oh = np.stack([(body_type == t).values for t in self.types], 1).astype(float) if self.types \
            else np.zeros((len(body), 0))
        return np.hstack([z, miss, oh]).astype(np.float32)

    def state(self) -> dict:
        return {"cols": self.cols, "miss_cols": self.miss_cols, "mu": self.mu, "sd": self.sd, "types": self.types}

    @classmethod
    def from_state(cls, s: dict) -> "BodyEncoder":
        e = cls()
        e.cols, e.mu, e.sd, e.types = list(s["cols"]), dict(s["mu"]), dict(s["sd"]), list(s["types"])
        e.miss_cols = list(s["miss_cols"])
        return e


class SizeRegressor:
    """Ước lượng size "tự nhiên" của người dùng từ số đo cơ thể (theo từng danh mục): Ridge regression.
    Đặc trưng size_gap = (size đặt - size ước lượng) / sd cho mô hình biết size này to hay nhỏ so với cơ thể,
    tương tự ý tưởng "true size" của Misra et al. (2018) nhưng suy ra trực tiếp từ số đo tự khai
    -> dùng được cả cho người dùng mới."""

    def fit(self, X: np.ndarray, category: pd.Series, size: pd.Series, alpha: float = 1.0) -> "SizeRegressor":
        vc = category.value_counts()
        self.cats = list(vc[vc >= 30].index)
        A = self._design(X, category)
        y = size.values.astype(float)
        reg = alpha * np.eye(A.shape[1])
        reg[0, 0] = 0.0
        self.coef = np.linalg.solve(A.T @ A + reg, A.T @ y).tolist()
        self.sd = float(np.std(y - A @ np.array(self.coef))) or 1.0
        return self

    def _design(self, X, category):
        oh = np.stack([(category == c).values for c in self.cats], 1).astype(float) if self.cats \
            else np.zeros((len(X), 0))
        return np.hstack([np.ones((len(X), 1)), X, oh])

    def predict(self, X, category) -> np.ndarray:
        return self._design(X, category) @ np.array(self.coef)

    def gap(self, X, category, size) -> np.ndarray:
        return ((size.values - self.predict(X, category)) / self.sd).clip(-5, 5).astype(np.float32)

    def state(self):
        return {"cats": self.cats, "coef": self.coef, "sd": self.sd}

    @classmethod
    def from_state(cls, s):
        r = cls()
        r.cats, r.coef, r.sd = list(s["cats"]), list(s["coef"]), s["sd"]
        return r


INDEX_COLS = ["user_id", "item_size", "item_id", "category", "context"]


class Indexer:
    """Ánh xạ id -> chỉ số (0 = chưa biết / người dùng mới)."""

    def fit(self, df: pd.DataFrame) -> "Indexer":
        self.maps = {c: {k: i + 1 for i, k in enumerate(df[c].unique())} for c in INDEX_COLS}
        self.size_mu = float(df["size"].mean())
        self.size_sd = float(df["size"].std()) or 1.0
        return self

    def sizes(self) -> dict:
        return {c: len(m) for c, m in self.maps.items()}

    def transform(self, df: pd.DataFrame) -> dict:
        out = {c: df[c].map(self.maps[c]).fillna(0).astype(np.int64).values for c in INDEX_COLS}
        out["size_z"] = ((df["size"] - self.size_mu) / self.size_sd).values.astype(np.float32)
        return out

    def state(self) -> dict:
        return {"maps": self.maps, "size_mu": self.size_mu, "size_sd": self.size_sd}

    @classmethod
    def from_state(cls, s: dict) -> "Indexer":
        ix = cls()
        ix.maps, ix.size_mu, ix.size_sd = s["maps"], s["size_mu"], s["size_sd"]
        return ix


def split(df: pd.DataFrame, protocol: str, seed: int):
    """cold: chia theo user (user test không có trong train). warm: chia ngẫu nhiên theo tương tác."""
    rng = np.random.RandomState(seed)
    if protocol == "warm":
        r = rng.rand(len(df))
        return np.where(r < .8)[0], np.where((r >= .8) & (r < .9))[0], np.where(r >= .9)[0]
    if protocol != "cold":
        raise ValueError(f"giao thức không hợp lệ: {protocol}")
    users = np.array(df["user_id"].unique(), dtype=object)
    rng.shuffle(users)
    n = len(users)
    tr, va = set(users[:int(.8 * n)]), set(users[int(.8 * n):int(.9 * n)])
    m_tr = df["user_id"].isin(tr).values
    m_va = df["user_id"].isin(va).values
    return np.where(m_tr)[0], np.where(m_va)[0], np.where(~m_tr & ~m_va)[0]


def history_bucket(n_hist: np.ndarray) -> np.ndarray:
    return np.select([n_hist == 0, n_hist == 1, n_hist <= 4], ["0", "1", "2-4"], "5+")


def describe(data: FitData) -> dict:
    df, c = data.df, data.df["user_id"].value_counts()
    return {
        "dataset": data.name,
        "interactions": len(df),
        "users": int(df["user_id"].nunique()),
        "items": int(df["item_id"].nunique()),
        "item_sizes": int(df["item_size"].nunique()),
        "categories": int(df["category"].nunique()),
        "users_with_1_interaction": float((c == 1).mean()),
        "fit_ratio": {k: float((df["y"] == v).mean()) for k, v in FIT_MAP.items()},
        "body_missing": {k: float(data.body[k].isna().mean()) for k in data.body_cols},
    }
