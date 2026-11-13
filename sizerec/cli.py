"""Giao diện dòng lệnh `sizerec`.

    sizerec info                                   Thông tin phiên bản, môi trường, danh sách mô hình
    sizerec data        DATA                       Thống kê bộ dữ liệu
    sizerec experiment  DATA [tuỳ chọn]            Thực nghiệm so sánh + kiểm định thống kê
    sizerec train       DATA [tuỳ chọn]            Huấn luyện và lưu mô hình đề xuất
    sizerec inspect     MODEL                      Xem thông tin mô hình đã lưu
    sizerec recommend   MODEL --item ID [...]      Gợi ý size (người dùng cũ hoặc khách mới theo số đo)

Gõ `sizerec <lệnh> -h` để xem chi tiết từng lệnh.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime

from . import __version__
from .ui import UI

EPILOG = """Gõ `sizerec` (không tham số) để mở chế độ tương tác dạng menu.

Ví dụ:
  sizerec data dataset/renttherunway_final_data.json.gz
  sizerec experiment dataset/modcloth_final_data.json.gz --quick
  sizerec experiment dataset/renttherunway_final_data.json.gz --seeds 1 2 3 4 5 6 7 8 9 10
  sizerec train dataset/renttherunway_final_data.json.gz -o models/rtr.pt
  sizerec recommend models/rtr.pt --item 2260466 --height "5'6\\"" --weight 130lbs --bust-size 34b --explain
"""


# ============================================================================ helpers
def _add_hparams(p):
    g = p.add_argument_group("siêu tham số")
    g.add_argument("--dim", type=int, default=16, help="số chiều không gian ẩn (mặc định: 16)")
    g.add_argument("--lr", type=float, default=3e-3, help="learning rate Adam (mặc định: 3e-3)")
    g.add_argument("--l2", type=float, default=1e-2, help="hệ số L2 cho embedding (mặc định: 1e-2)")
    g.add_argument("--batch", type=int, default=1024, help="batch size (mặc định: 1024)")
    g.add_argument("--epochs", type=int, default=30, help="số epoch tối đa (mặc định: 30)")
    g.add_argument("--patience", type=int, default=3, help="early stopping theo AUC val (mặc định: 3)")
    g.add_argument("--id-drop", type=float, default=0.1, help="xác suất ID dropout của mô hình đề xuất (mặc định: 0.1)")


def _hparams(a):
    from .trainer import HParams
    return HParams(dim=a.dim, lr=a.lr, l2=a.l2, batch=a.batch, epochs=a.epochs, patience=a.patience)


def _apply_id_drop(value):
    from dataclasses import replace
    from . import model as M
    for k in ["mf_body", "mf_body_product"]:
        M.SPECS[k] = replace(M.SPECS[k], id_drop=value)


def _check_file(ui, path):
    if not os.path.isfile(path):
        ui.error(f"Không tìm thấy file: {path}")
        sys.exit(2)


def _load(ui, path):
    from .data import load_dataset
    _check_file(ui, path)
    t0 = time.time()
    data = load_dataset(path)
    ui.ok(f"Đã nạp [bold]{data.name}[/bold]: {len(data):,} tương tác, "
          f"{data.df.user_id.nunique():,} người dùng, {data.df.item_id.nunique():,} sản phẩm ({time.time() - t0:.1f}s)")
    return data


def _fmt(x, nd=4):
    return "—" if x != x else f"{x:.{nd}f}"


# ============================================================================ commands
def cmd_info(a, ui):
    import numpy, pandas, sklearn, torch
    from .model import DESCRIPTIONS, PROPOSED
    ui.header(f"sizerec {__version__}", "Hybrid Matrix Factorization cho gợi ý size thời trang")
    ui.kv([("Python", platform.python_version()), ("PyTorch", torch.__version__),
           ("NumPy / pandas / scikit-learn", f"{numpy.__version__} / {pandas.__version__} / {sklearn.__version__}"),
           ("Thiết bị", "CUDA" if torch.cuda.is_available() else f"CPU ({torch.get_num_threads()} luồng)")],
          title="Môi trường")
    ui.table(["Mô hình", "Mô tả"], [(k + (" ★" if k == PROPOSED else ""), v) for k, v in DESCRIPTIONS.items()],
             title="Các mô hình có sẵn", justify=["left", "left"])


def cmd_data(a, ui):
    from .data import LEAKY_FIELDS, describe
    data = _load(ui, a.data)
    d = describe(data)
    if a.json:
        print(json.dumps(d, ensure_ascii=False, indent=2))
        return
    ui.kv([("Tương tác", f"{d['interactions']:,}"), ("Người dùng", f"{d['users']:,}"),
           ("Sản phẩm / cặp sản phẩm-size", f"{d['items']:,} / {d['item_sizes']:,}"),
           ("Danh mục", d["categories"]),
           ("User chỉ có 1 tương tác", f"{d['users_with_1_interaction']:.1%}"),
           ("Tỉ lệ nhãn small / fit / large", " / ".join(f"{d['fit_ratio'][k]:.1%}" for k in ["small", "fit", "large"]))],
          title=f"Bộ dữ liệu {d['dataset']}")
    ui.table(["Số đo cơ thể", "Tỉ lệ thiếu"], [(k, f"{v:.1%}") for k, v in d["body_missing"].items()])
    ui.log(f"Các trường bị loại để tránh rò rỉ thông tin sau mua: {', '.join(LEAKY_FIELDS)}")


def cmd_experiment(a, ui):
    import torch
    from .experiment import DEFAULT_MODELS, run_experiment, write_outputs
    from .data import describe
    from .model import SPECS
    if a.quick:
        a.protocols, a.seeds, a.boot = ["cold"], [1], 50
    bad = [m for m in a.models if m not in SPECS and m not in ("majority", "body_lr")]
    if bad:
        ui.error(f"Mô hình không hợp lệ: {', '.join(bad)}. Xem `sizerec info`.")
        sys.exit(2)
    _apply_id_drop(a.id_drop)
    data = _load(ui, a.data)
    hp = _hparams(a)
    outdir = a.out_dir or os.path.join("results", f"{data.name}_{datetime.now():%Y%m%d_%H%M%S}")
    ui.header(f"Thực nghiệm trên {data.name}",
              f"giao thức: {', '.join(a.protocols)} · seeds: {a.seeds} · {len(a.models)} mô hình · "
              f"bootstrap: {a.boot} · CPU {torch.get_num_threads()} luồng")
    t0 = time.time()
    res, sig, size_df = run_experiment(data, a.protocols, a.seeds, a.models, hp, log=ui.log, n_boot=a.boot)
    config = {"dataset": data.name, "data_path": os.path.abspath(a.data), "protocols": a.protocols, "seeds": a.seeds,
              "models": a.models, "hparams": hp.to_dict(), "id_drop": a.id_drop, "bootstrap": a.boot,
              "version": __version__, "finished": datetime.now().isoformat(timespec="seconds")}
    summ = write_outputs(outdir, describe(data), res, sig, config, size_df)
    _print_summary(ui, summ, sig)
    _print_size(ui, size_df)
    ui.ok(f"Hoàn tất sau {time.time() - t0:.0f}s. Kết quả lưu tại [bold]{outdir}[/bold] "
          f"(summary.csv, significance.csv, metrics_per_seed.csv, report.md)")


def _print_summary(ui, summ, sig):
    from .model import PROPOSED
    for (proto, bucket), g in summ.groupby(["protocol", "bucket"], sort=False):
        label = "cold-start: người dùng test chưa từng xuất hiện" if proto == "cold" else \
            f"warm: người dùng có {bucket} lịch sử trong train" if bucket != "all" else "warm: toàn bộ tập test"
        best = g["auc_mean"].max()
        rows = [(r.model + (" ★" if r.model == PROPOSED else ""), f"{_fmt(r.auc_mean)} ± {_fmt(r.auc_std if r.auc_std == r.auc_std else 0.0)}",
                 _fmt(r.macro_f1_mean), _fmt(r.acc_mean)) for r in g.itertuples()]
        ui.table(["Mô hình", "AUC (mean ± std)", "Macro-F1", "Acc"], rows,
                 title=f"{label} (n≈{int(g['n'].iloc[0]):,}, {int(g['seeds'].iloc[0])} seed)",
                 highlight_row=lambda i, r: abs(g.iloc[i]["auc_mean"] - best) < 1e-12)
    if len(sig):
        rows = [(f"{r.protocol}/{r.bucket}", r.vs, f"{r.delta_auc:+.4f}", f"[{r.ci95_low:+.3f}, {r.ci95_high:+.3f}]",
                 (f"<{1 / r.n_boot:.3f}" if r.p_bootstrap == 0 else _fmt(r.p_bootstrap, 3)), _fmt(r.p_ttest, 3), "có" if r.significant else "không")
                for r in sig.itertuples()]
        ui.table(["Nhóm", "So với", "ΔAUC", "CI 95%", "p boot", "p t-test", "p<0.05"], rows,
                 title=f"Kiểm định: {PROPOSED} ★ so với từng mô hình (paired bootstrap + paired t-test theo seed)",
                 justify=["left", "left", "right", "right", "right", "right", "left"])


def _print_size(ui, size_df):
    from .experiment import summarize_size
    if size_df is None or not len(size_df):
        return
    labels = {"popular": "Size bán chạy nhất của sản phẩm (mốc)", "body": "Chỉ số đo cơ thể",
              "hybrid": "Số đo + lịch sử người dùng (hybrid) ★"}
    s = summarize_size(size_df)
    rows = [(f"{r.protocol}", labels.get(r.method, r.method), f"{r.exact_mean:.1%}", f"{r.within_1_mean:.1%}", f"{r.n:,}")
            for r in s.itertuples()]
    ui.table(["Giao thức", "Phương pháp", "Trùng đúng size", "Lệch ≤ 1 size", "n"], rows,
             title="Đề xuất size: so với size khách thực sự mặc vừa (giao dịch 'fit' trong tập test)",
             justify=["left", "left", "right", "right", "right"])


def cmd_train(a, ui):
    import numpy as np
    from .model import SPECS
    from .trainer import Prepared, metrics, predict, save_bundle, train_model, tune_decision
    _apply_id_drop(a.id_drop)
    if a.model not in SPECS:
        ui.error(f"Mô hình không hợp lệ: {a.model}. Chọn một trong: {', '.join(SPECS)}")
        sys.exit(2)
    data = _load(ui, a.data)
    rng = np.random.RandomState(a.seed)
    r = rng.rand(len(data))
    tr, va = np.where(r >= a.val_frac)[0], np.where(r < a.val_frac)[0]
    ui.step(f"Huấn luyện [bold]{a.model}[/bold] trên {len(tr):,} tương tác, kiểm định trên {len(va):,}")
    prep = Prepared(data, tr, va)
    hp = _hparams(a)
    m = train_model(prep, SPECS[a.model], hp, a.seed,
                    on_epoch=lambda h: ui.log(f"  epoch {h['epoch']:2d}  train NLL={h['train_nll']:.4f}  "
                                              f"val AUC={h['val_auc']:.4f}  ({h['sec']:.1f}s)"))
    p_va = predict(m, prep.tensors["val"])
    w = tune_decision(p_va, prep.y("val"))
    met = metrics(prep.y("val"), p_va, w)
    out = a.output or os.path.join("models", f"{data.name}_{a.model}.pt")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    save_bundle(out, m, prep, SPECS[a.model], hp, extra={
        "dataset": data.name, "model_name": a.model, "val_metrics": met, "decision_weights": w.tolist(),
        "trained_at": datetime.now().isoformat(timespec="seconds"), "version": __version__,
        "n_train": int(len(tr)), "seed": a.seed})
    ui.ok(f"Val AUC={met['auc']:.4f}, Macro-F1={met['macro_f1']:.4f}. Đã lưu mô hình: [bold]{out}[/bold]")


def cmd_inspect(a, ui):
    from .trainer import Bundle
    _check_file(ui, a.model)
    b = Bundle(a.model)
    e = b.extra
    sizes = b.indexer.sizes()
    ui.kv([("Bộ dữ liệu", e.get("dataset")), ("Cấu hình", e.get("model_name")), ("Spec", b.spec.to_dict()),
           ("Siêu tham số", b.hp.to_dict()), ("Huấn luyện lúc", e.get("trained_at")),
           ("Số tương tác train", f"{e.get('n_train', 0):,}"),
           ("Người dùng / sản phẩm / cặp sản phẩm-size", f"{sizes['user_id']:,} / {sizes['item_id']:,} / {sizes['item_size']:,}"),
           ("Số đo cơ thể dùng", ", ".join(b.body_enc.cols) + (f"; body type: {len(b.body_enc.types)} loại" if b.body_enc.types else "")),
           ("Val AUC / Macro-F1", f"{e['val_metrics']['auc']:.4f} / {e['val_metrics']['macro_f1']:.4f}")],
          title=f"Mô hình {a.model}")


def cmd_recommend(a, ui):
    from .recommend import FIT_LABELS, TERM_LABELS, body_contributions, recommend
    from .trainer import Bundle
    _check_file(ui, a.model)
    b = Bundle(a.model)
    body = {k: getattr(a, k) for k in ["height", "weight", "age", "bust_size", "bra_size", "cup", "bust",
                                        "waist", "hips", "shoe", "body_type"]}
    if a.user is None and not any(v for k, v in body.items()):
        ui.warn("Không có user_id và không có số đo nào: mô hình chỉ dựa vào thông tin sản phẩm.")
    try:
        r = recommend(b, a.item, a.user, body, context=a.occasion or "unknown")
    except (KeyError, ValueError) as ex:
        ui.error(str(ex).strip("'\""))
        sys.exit(2)
    if a.json:
        if a.explain:
            r["body_contributions"] = body_contributions(b, body)
        print(json.dumps(r, ensure_ascii=False, indent=2))
        return
    who = f"user {a.user} ({r['user_status']}, {r['user_history']} lịch sử)" if a.user else r["user_status"]
    ui.header(f"Gợi ý size cho sản phẩm {r['item_id']} ({r['category']})", who)
    if r["body_used"]:
        ui.log("Số đo dùng (đơn vị inch / lbs): " + ", ".join(f"{k}={v:g}" for k, v in r["body_used"].items()))
    rows = [(f"{x['size']:g}", f"{x['distance']:.1f}", f"{x['p_fit']:.1%}", f"{x['p_small']:.1%}", f"{x['p_large']:.1%}")
            for x in r["ranking"][:a.top]]
    ui.table(["Size", "Cách size mục tiêu", "P(vừa)", "P(chật)", "P(rộng)"], rows,
             title=f"Các size của sản phẩm (size mục tiêu ≈ {r['size_estimate']['target']:.1f})",
             highlight_row=lambda i, _: i == 0)
    ui.ok(f"Size đề xuất: [bold]{r['recommended_size']:g}[/bold] · mô hình hybrid: P(vừa) {r['fit_probability']:.1%}, "
          f"P(chật) {r['risk_small']:.1%}, P(rộng) {r['risk_large']:.1%}")
    if r["advice"]:
        ui.warn(r["advice"])
    if a.explain:
        e = r["size_estimate"]
        ui.table(["Cách tính size mục tiêu", "Giá trị"], [
            ("Size nền theo số đo cơ thể", f"{e['base_body']:.2f}"),
            ("Size nền theo lịch sử mua", "—" if e["user_base"] is None else f"{e['user_base']:.2f}"),
            ("Trọng số lịch sử (n / (n + 2))", f"{e['user_weight']:.2f}"),
            (f"Điều chỉnh theo danh mục '{r['category']}'", f"{e['category_offset']:+.2f}"),
            ("Sản phẩm chạy to (-) / nhỏ (+)", f"{e['item_offset']:+.2f}"),
            ("Size mục tiêu", f"{e['target']:.2f}")], title="Giải thích: chọn size", justify=["left", "right"])
        best = r["ranking"][0]
        th = r["thresholds"]
        rows = [(TERM_LABELS[k], f"{v:+.3f}") for k, v in best["terms"].items()]
        rows.append(("Tổng điểm t", f"{best['score_t']:+.3f}"))
        ui.table(["Thành phần của điểm t (size đề xuất)", "Giá trị"], rows, title="Giải thích: rủi ro không vừa của size đề xuất (mô hình hybrid MF)")
        ui.print(f"  Quy tắc: t < {th['theta_small']:.3f} → {FIT_LABELS['small']}; "
                 f"{th['theta_small']:.3f} ≤ t ≤ {th['theta_large']:.3f} → {FIT_LABELS['fit']}; "
                 f"t > {th['theta_large']:.3f} → {FIT_LABELS['large']}")
        ui.print("  Thành phần dương đẩy về phía 'rộng', âm đẩy về phía 'chật'.")
        contrib = body_contributions(b, body)
        if contrib:
            ui.table(["Đặc trưng cơ thể (phần tuyến tính g)", "z-score", "Đóng góp"],
                     [(c["feature"], f"{c['value_z']:+.2f}", f"{c['contribution']:+.3f}") for c in contrib])


# ============================================================================ parser
def build_parser():
    from .experiment import DEFAULT_MODELS
    from .model import PROPOSED, SPECS
    p = argparse.ArgumentParser(prog="sizerec", description="Hybrid Matrix Factorization cho gợi ý size (ModCloth / RentTheRunway).",
                                epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-V", "--version", action="version", version=f"sizerec {__version__}")
    p.add_argument("-q", "--quiet", action="store_true", help="ẩn log chi tiết")
    p.add_argument("--no-color", action="store_true", help="tắt màu")
    p.add_argument("--threads", type=int, default=None, help="số luồng CPU cho PyTorch")
    sub = p.add_subparsers(dest="cmd", metavar="<lệnh>")

    s = sub.add_parser("shell", aliases=["menu"], help="chế độ tương tác dạng menu (mặc định khi gõ `sizerec`)")
    s.set_defaults(fn=None, shell=True)

    s = sub.add_parser("info", help="thông tin phiên bản, môi trường và các mô hình")
    s.set_defaults(fn=cmd_info)

    s = sub.add_parser("data", help="thống kê bộ dữ liệu")
    s.add_argument("data", help="file .json hoặc .json.gz (ModCloth / RentTheRunway)")
    s.add_argument("--json", action="store_true", help="in kết quả dạng JSON")
    s.set_defaults(fn=cmd_data)

    s = sub.add_parser("experiment", help="thực nghiệm so sánh mô hình + kiểm định thống kê",
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    s.add_argument("data", help="file .json hoặc .json.gz")
    s.add_argument("--protocols", nargs="+", choices=["cold", "warm"], default=["cold", "warm"],
                   help="cold: chia theo user; warm: chia ngẫu nhiên + tách theo số lịch sử")
    s.add_argument("--seeds", nargs="+", type=int, default=list(range(1, 11)), help="các seed (mỗi seed một cách chia)")
    s.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                   help=f"mô hình cần so sánh; có thể thêm: {', '.join(k for k in SPECS if k not in DEFAULT_MODELS)}")
    s.add_argument("--boot", type=int, default=200, help="số lần lặp paired bootstrap")
    s.add_argument("--quick", action="store_true", help="chạy thử nhanh: chỉ cold, 1 seed, 50 bootstrap")
    s.add_argument("-o", "--out-dir", default=None, help="thư mục kết quả (mặc định results/<dataset>_<thời gian>)")
    _add_hparams(s)
    s.set_defaults(fn=cmd_experiment)

    s = sub.add_parser("train", help="huấn luyện và lưu mô hình để gợi ý size")
    s.add_argument("data", help="file .json hoặc .json.gz")
    s.add_argument("--model", default=PROPOSED, help=f"cấu hình mô hình (mặc định: {PROPOSED})")
    s.add_argument("-o", "--output", default=None, help="đường dẫn file .pt (mặc định models/<dataset>_<model>.pt)")
    s.add_argument("--val-frac", type=float, default=0.1, help="tỉ lệ dữ liệu dùng cho early stopping (mặc định 0.1)")
    s.add_argument("--seed", type=int, default=1)
    _add_hparams(s)
    s.set_defaults(fn=cmd_train)

    s = sub.add_parser("inspect", help="xem thông tin mô hình đã lưu")
    s.add_argument("model", help="file .pt tạo bởi `sizerec train`")
    s.set_defaults(fn=cmd_inspect)

    s = sub.add_parser("recommend", help="gợi ý size cho một sản phẩm",
                       description="Người dùng cũ: --user. Khách mới (cold-start): nhập số đo cơ thể. "
                                   "Có thể dùng cả hai.")
    s.add_argument("model", help="file .pt tạo bởi `sizerec train`")
    s.add_argument("--item", required=True, help="item_id của sản phẩm")
    s.add_argument("--user", default=None, help="user_id (bỏ trống nếu là khách mới)")
    g = s.add_argument_group("số đo cơ thể (khách tự khai)")
    g.add_argument("--height", help="chiều cao: 5'6\", 5ft 6in, 168cm hoặc số inch")
    g.add_argument("--weight", help="cân nặng: 130lbs hoặc 59kg")
    g.add_argument("--age", type=float)
    g.add_argument("--bust-size", dest="bust_size", help="cỡ áo ngực dạng 34b (RentTheRunway)")
    g.add_argument("--bra-size", dest="bra_size", type=float, help="cỡ vòng dưới ngực, ví dụ 34 (ModCloth)")
    g.add_argument("--cup", help="cup: a, b, c, d, dd/e, ... (ModCloth)")
    g.add_argument("--bust", type=float, help="vòng ngực (inch)")
    g.add_argument("--waist", type=float, help="vòng eo (inch)")
    g.add_argument("--hips", type=float, help="vòng hông (inch)")
    g.add_argument("--shoe", type=float, help="cỡ giày (US)")
    g.add_argument("--body-type", dest="body_type", help="hourglass, athletic, pear, petite, full bust, "
                                                          "straight & narrow, apple")
    s.add_argument("--occasion", help="dịp sử dụng (RTR rented_for): wedding, party, work, ...")
    s.add_argument("--top", type=int, default=10, help="số size hiển thị (mặc định 10)")
    s.add_argument("--explain", action="store_true", help="giải thích các thành phần của dự đoán")
    s.add_argument("--json", action="store_true", help="in kết quả dạng JSON")
    s.set_defaults(fn=cmd_recommend)
    return p


def main(argv=None):
    p = build_parser()
    a = p.parse_args(argv)
    ui = UI(quiet=a.quiet, color=not a.no_color)
    if a.threads:
        import torch
        torch.set_num_threads(a.threads)
    if not getattr(a, "fn", None):
        if getattr(a, "shell", False) or (a.cmd is None and sys.stdin.isatty()):
            from .interactive import run_shell
            try:
                return run_shell(ui)
            except KeyboardInterrupt:
                return 130
        ui.header(f"sizerec {__version__}", "Hybrid Matrix Factorization cho gợi ý size thời trang")
        p.print_help()
        return 0
    try:
        a.fn(a, ui)
    except KeyboardInterrupt:
        ui.error("Đã dừng theo yêu cầu.")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
