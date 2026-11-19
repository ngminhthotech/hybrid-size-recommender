# Workflow ↔ code

Tài liệu đối chiếu 8 bước của sơ đồ workflow với code trong `sizerec/` (bản cập nhật 17/11/2026).

| File | Nội dung |
| --- | --- |
| `hinh1_workflow.png` | Sơ đồ workflow 8 bước |
| `code_khop_workflow.docx`, `.pptx` | Bảng bước → file → hàm → số dòng code |
| `BANG_TRA_SO_DONG.md` | Bảng tra số dòng của từng hàm |
| `HUONG_DAN_TRINH_BAY.md` | Hướng dẫn trình bày code theo từng bước |

Các tài liệu này được viết cho bố cục thư mục của bài nộp (`03-code`, `02-data`, `04-results`). Trong repo này, `03-code` chính là thư mục gốc, dữ liệu nằm ở `data/`, còn kết quả nằm ở `results/`. Ví dụ, chạy script tự kiểm tra từ thư mục gốc của repo:

```bash
python scripts/verify_workflow.py data/modcloth_final_data.json.gz
python scripts/verify_workflow.py data/renttherunway_final_data.json.gz --protocol warm
```
