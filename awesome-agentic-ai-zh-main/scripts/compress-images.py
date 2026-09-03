#!/usr/bin/env python3
"""
compress-images.py — 壓縮 resources/diagrams/ 內的 PNG 到 < 1 MB

策略：
1. PIL `optimize=True` + 降低 8-bit palette（如果可能）
2. 如果還 > 1 MB，等比例縮小到 max width 1600px（README 顯示夠用）
3. 仍 > 1 MB 就用 JPEG 95 quality 儲存

依賴：pip install Pillow
"""

import os
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    print("ERROR: 需要 Pillow。pip install Pillow", file=__import__('sys').stderr)
    raise

REPO_ROOT = Path(__file__).resolve().parent.parent
DIAGRAMS_DIR = REPO_ROOT / "resources" / "diagrams"
TARGET_SIZE_KB = 1000  # 目標：< 1 MB
MAX_WIDTH = 1600       # README banner 不需要超過這個寬度

def compress_png(src: Path) -> bool:
    """嘗試多階段壓縮，回傳是否成功 (< target size)。"""
    original_size_kb = src.stat().st_size // 1024
    print(f"\n=== {src.name} (original {original_size_kb} KB) ===")

    img = Image.open(src)

    # 階段 1: 重存 + optimize
    img.save(src, optimize=True)
    size_kb = src.stat().st_size // 1024
    print(f"  Stage 1 (optimize=True): {size_kb} KB")
    if size_kb < TARGET_SIZE_KB:
        return True

    # 階段 2: 縮 width 到 1600 + optimize
    if img.width > MAX_WIDTH:
        ratio = MAX_WIDTH / img.width
        new_size = (MAX_WIDTH, int(img.height * ratio))
        img_resized = img.resize(new_size, Image.LANCZOS)
        img_resized.save(src, optimize=True)
        size_kb = src.stat().st_size // 1024
        print(f"  Stage 2 (resize to {MAX_WIDTH}w): {size_kb} KB")
        if size_kb < TARGET_SIZE_KB:
            return True
        img = img_resized

    # 階段 3: 換 JPEG (有些 banner 圖 PNG transparency 沒用)
    jpg_path = src.with_suffix(".jpg")
    if img.mode == "RGBA":
        # JPEG 不支援 transparency, 加白色背景
        bg = Image.new("RGB", img.size, "white")
        bg.paste(img, mask=img.split()[3] if len(img.split()) == 4 else None)
        img_to_save = bg
    else:
        img_to_save = img.convert("RGB")
    img_to_save.save(jpg_path, "JPEG", quality=92, optimize=True)
    jpg_size_kb = jpg_path.stat().st_size // 1024
    print(f"  Stage 3 (JPEG q=92): {jpg_size_kb} KB → {jpg_path.name}")

    if jpg_size_kb < TARGET_SIZE_KB:
        # JPG 比 PNG 小，但保留 PNG 為對照（暫時——可以手動決定要不要刪 PNG）
        return True
    print(f"  ⚠️ 仍超過目標。考慮再縮 width 或降 quality。")
    return False


def main():
    if not DIAGRAMS_DIR.exists():
        print(f"ERROR: {DIAGRAMS_DIR} 不存在")
        return

    pngs = list(DIAGRAMS_DIR.glob("*.png"))
    if not pngs:
        print("沒找到 PNG")
        return

    print(f"處理 {len(pngs)} 個 PNG...")
    all_ok = True
    for png in pngs:
        if not compress_png(png):
            all_ok = False

    print()
    print("=" * 50)
    if all_ok:
        print("✓ 全部 < 1 MB")
    else:
        print("⚠️ 有檔案仍 > 1 MB，看 stage 3 的 JPG 版或手動處理")


if __name__ == "__main__":
    main()
