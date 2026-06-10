#!/usr/bin/env python
"""
从 Pixiv 随机图片批量生成蒸馏软目标 targets_pixiv_{time}.json

工作流程（循环执行）：
1. 随机获取 20 张 Pixiv 图片下载到本地
2. 取前 10 张运用 graphcolor 方法生成 result.json
3. 将 result.json 转化为 target_pixiv_{time}.json（存储源 URL 而非文件名）
4. 保存 targets 文件，移除临时图片，回到步骤 1

按 Ctrl+C 可随时安全退出。
"""
import json
import math
import os
import shutil
import signal
import sys
import tempfile
import time
import urllib.request
import urllib.parse
from datetime import datetime
from pathlib import Path
from PIL import Image, ImageDraw


# ─── 配置 ───────────────────────────────────────────────────────────────
PIXIV_API = "https://api.lolicon.app/setu/v2"
BATCH_SIZE = 20          # 每次下载图片数
PROCESS_COUNT = 10       # 实际处理图片数
TEMP_DIR = "pixiv_temp"
BREAKPOINT_FILE = "targets_pixiv_progress.json"

# 复用 graphcolor 的 process_batch
sys.path.insert(0, str(Path(__file__).parent))
from graphcolor import process_batch


# ─── 工具函数 ───────────────────────────────────────────────────────────

def softmax(scores, temperature=2.0):
    """计算 softmax 权重"""
    scaled = [s / temperature for s in scores]
    max_s = max(scaled)
    exps = [math.exp(s - max_s) for s in scaled]
    total = sum(exps)
    return [e / total for e in exps]


def lab_to_rgb(L, a, b):
    """Lab 转 RGB"""
    xn, yn, zn = 95.047, 100.0, 108.883
    fy = (L + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b / 200.0
    delta = 6.0 / 29.0

    def f_inv(t):
        if t > delta:
            return t ** 3
        return 3 * delta * delta * (t - 4.0 / 29.0)

    x = xn * f_inv(fx)
    y = yn * f_inv(fy)
    z = zn * f_inv(fz)
    xr, yr, zr = x / 100.0, y / 100.0, z / 100.0

    rl = 3.2406 * xr - 1.5372 * yr - 0.4986 * zr
    gl = -0.9689 * xr + 1.8758 * yr + 0.0415 * zr
    bl = 0.0557 * xr - 0.2040 * yr + 1.0570 * zr

    def gamma(c):
        if c > 0.0031308:
            return 1.055 * (c ** (1.0 / 2.4)) - 0.055
        return 12.92 * c

    r = max(0, min(255, int(round(gamma(rl) * 255))))
    g = max(0, min(255, int(round(gamma(gl) * 255))))
    b_val = max(0, min(255, int(round(gamma(bl) * 255))))
    return (r, g, b_val)


# ─── 下载 Pixiv 图片 ───────────────────────────────────────────────────

def fetch_pixiv_urls(count=20):
    """通过 lolicon API 获取随机 Pixiv 图片 URL"""
    url = f"{PIXIV_API}?r18=0&num={count}&size=original"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode())

    urls = []
    for item in data.get("data", []):
        # 优先 original，回退 regular
        original = item.get("urls", {}).get("original")
        regular = item.get("urls", {}).get("regular")
        if original or regular:
            src_url = original or regular
            urls.append({
                "url": src_url,
                "pid": item.get("pid"),
                "title": item.get("title", ""),
                "author": item.get("author", ""),
                "tags": item.get("tags", [])[:5],  # 前5个标签
                "page": item.get("page", 0),
            })
    return urls


def download_images(url_infos, temp_dir):
    """下载图片到临时目录，返回 (本地路径列表, 对应URL信息列表)"""
    os.makedirs(temp_dir, exist_ok=True)
    local_paths = []
    valid_infos = []

    for i, info in enumerate(url_infos):
        src_url = info["url"]
        ext = ".jpg"  # Pixiv original 通常是 jpg
        if src_url.endswith(".png"):
            ext = ".png"

        local_name = f"pixiv_{info['pid']}_{info['page']}{ext}"
        local_path = os.path.join(temp_dir, local_name)

        try:
            req = urllib.request.Request(src_url, headers={
                "User-Agent": "Mozilla/5.0",
                "Referer": "https://www.pixiv.net/",
            })
            with urllib.request.urlopen(req, timeout=30) as resp:
                img_data = resp.read()
            with open(local_path, "wb") as f:
                f.write(img_data)

            # 验证图片可读
            img = Image.open(local_path)
            img.verify()
            img.close()

            local_paths.append(local_path)
            valid_infos.append({**info, "local_path": local_path, "local_name": local_name})
            print(f"  [{i+1}/{len(url_infos)}] 下载成功: {local_name}")
        except Exception as e:
            print(f"  [{i+1}/{len(url_infos)}] 下载失败: {local_name} - {e}")

    return local_paths, valid_infos


# ─── 预览图 ─────────────────────────────────────────────────────────────

def create_preview(img_path, colors):
    """创建预览图：原始图片 + N个颜色色块"""
    img = Image.open(img_path).convert("RGB")
    max_size = 400
    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    color_bar_height = 80
    canvas_width = img.width
    canvas_height = img.height + color_bar_height * len(colors) + 20
    canvas = Image.new("RGB", (canvas_width, canvas_height), (255, 255, 255))
    canvas.paste(img, (0, 0))

    draw = ImageDraw.Draw(canvas)
    for i, (L, a, b, score) in enumerate(colors):
        r, g, b_val = lab_to_rgb(L, a, b)
        y_pos = img.height + 10 + i * (color_bar_height + 10)
        bar_width = int(img.width * 0.8)
        draw.rectangle([10, y_pos, 10 + bar_width, y_pos + color_bar_height], fill=(r, g, b_val))
        draw.rectangle([10, y_pos, 10 + bar_width, y_pos + color_bar_height], outline=(0, 0, 0), width=2)
        draw.text((20, y_pos + 5), f"#{i+1}", fill=(0,0,0) if (r+g+b_val) > 382 else (255,255,255))
        text = f" L={L:.1f} a={a:.1f} b={b:.1f}  score={score:.4f}"
        draw.text((10 + bar_width + 10, y_pos + 10), text, fill=(0, 0, 0))
    return canvas


def show_preview_and_wait(canvas, img_info, region_type):
    """弹窗显示预览图"""
    import tempfile as tmp
    region_label = "前景" if region_type == "fg" else "背景"
    tmp_path = os.path.join(tmp.gettempdir(), f"pixiv_{img_info['pid']}_{region_type}_preview.png")
    canvas.save(tmp_path)
    canvas.show()
    print(f"  [{img_info['local_name']} - {region_label}] 预览图已弹窗显示，请查看...")


def get_user_permutation(n_colors):
    """读取用户输入的排列，空白回车默认为 2 1 3；输入 0 跳过此图"""
    while True:
        try:
            if n_colors == 3:
                prompt = "请输入色彩排序（空格分隔三个数字，如 2 1 3；直接回车默认 2 1 3；输入 0 跳过此图）："
            else:
                expected = " ".join(str(i) for i in range(1, n_colors + 1))
                prompt = f"请输入色彩排序（空格分隔{n_colors}个数字，如 {expected}；直接回车保持当前顺序；输入 0 跳过此图）："

            user_input = input(prompt).strip()
            if user_input == "":
                return [2, 1, 3] if n_colors == 3 else list(range(1, n_colors + 1)), False
            if user_input == "0":
                return None, True
            parts = list(map(int, user_input.split()))
            if len(parts) != n_colors:
                print(f"错误：请输入恰好 {n_colors} 个数字")
                continue
            if sorted(parts) != list(range(1, n_colors + 1)):
                expected = " ".join(str(i) for i in range(1, n_colors + 1))
                print(f"错误：输入必须是 {expected} 的排列")
                continue
            return parts, False
        except ValueError:
            print("错误：请输入有效的整数")


# ─── 颜色处理（与 generate_targets.py 逻辑一致） ──────────────────────────

def process_colors(colors, img_info, region_type):
    """处理前景或背景颜色，返回软目标 Lab 值或 None（跳过）"""
    n_colors = min(len(colors), 3)
    if n_colors == 0:
        return 50.0, 0.0, 0.0

    sorted_colors = sorted(colors, key=lambda c: c["score"], reverse=True)[:n_colors]
    local_name = img_info["local_name"]
    img_path = img_info["local_path"]

    print(f"\n  处理 {local_name} - {region_type}")

    if n_colors >= 2:
        score1 = sorted_colors[0]["score"]
        score2 = sorted_colors[1]["score"]
        gap = (score1 - score2) / score1 if score1 > 0 else 0

        for i in range(n_colors):
            c = sorted_colors[i]
            label = f"  当前排序: {i+1}" if i == 0 else f"             {i+1}"
            print(f"{label}: Lab={c['lab']} (score={c['score']:.4f})")
        print(f"  gap = {gap:.4f}")

        if gap < 0.05:
            preview_colors = [(c["lab"][0], c["lab"][1], c["lab"][2], c["score"]) for c in sorted_colors]
            canvas = create_preview(img_path, preview_colors)

            print(f"  gap < 0.05，需要人工确认排序")
            label_str = "  ".join([f"{i+1}: Lab={sorted_colors[i]['lab']}" for i in range(n_colors)])
            print(f"  当前排序：{label_str}")

            show_preview_and_wait(canvas, img_info, region_type)

            perm, skip = get_user_permutation(n_colors)
            if skip:
                print(f"  用户选择跳过此图")
                return None, None, None
            print(f"  用户输入排序: {perm}")

            new_order = [sorted_colors[p - 1] for p in perm]
            original_scores = [c["score"] for c in sorted_colors]
            sorted_colors = new_order
            for i in range(n_colors):
                sorted_colors[i]["score"] = original_scores[i]
    else:
        print(f"  仅 {n_colors} 个颜色，跳过 gap 检测")

    sorted_colors[0]["score"] *= 2.0
    scores = [c["score"] for c in sorted_colors]
    weights = softmax(scores, temperature=2.0)

    L_sum = sum(w * c["lab"][0] for w, c in zip(weights, sorted_colors))
    a_sum = sum(w * c["lab"][1] for w, c in zip(weights, sorted_colors))
    b_sum = sum(w * c["lab"][2] for w, c in zip(weights, sorted_colors))

    print(f"  最终权重: {[f'{w:.4f}' for w in weights]}")
    print(f"  软目标 Lab: L={L_sum:.1f} a={a_sum:.1f} b={b_sum:.1f}")

    return L_sum, a_sum, b_sum


# ─── 断点管理 ────────────────────────────────────────────────────────────

def save_batch_checkpoint(batch_targets, temp_dir):
    checkpoint = {"targets": batch_targets, "temp_dir": temp_dir}
    with open(BREAKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=2, ensure_ascii=False)

def load_batch_checkpoint():
    if os.path.exists(BREAKPOINT_FILE):
        with open(BREAKPOINT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


# ─── 单批次处理 ──────────────────────────────────────────────────────────

def process_one_batch(batch_targets):
    """执行一次完整批次：下载 → graphcolor → 生成 targets"""
    print(f"\n{'='*60}")
    print(f"  开始新批次：下载 {BATCH_SIZE} 张 Pixiv 随机图片...")
    print(f"{'='*60}")

    # 1. 获取 URL 并下载
    url_infos = fetch_pixiv_urls(BATCH_SIZE)
    if not url_infos:
        print("  未获取到有效图片 URL，跳过此批次")
        return batch_targets

    local_paths, valid_infos = download_images(url_infos, TEMP_DIR)
    if len(local_paths) < 1:
        print("  没有成功下载任何图片，跳过此批次")
        return batch_targets

    # 只处理前 PROCESS_COUNT 张
    to_process = valid_infos[:PROCESS_COUNT]
    to_process_paths = [info["local_path"] for info in to_process]
    print(f"\n  共下载 {len(local_paths)} 张，将处理前 {len(to_process_paths)} 张...")

    # 2. 运行 graphcolor
    results_path = os.path.join(TEMP_DIR, "result.json")
    try:
        graph_results = process_batch(
            to_process_paths,
            config={"n_clusters_foreground": 10, "n_clusters_background": 6},
            output_json=results_path,
            verbose=True,
            workers=1,
        )
    except Exception as e:
        print(f"  graphcolor 处理失败: {e}")
        return batch_targets

    # 3. 将 results 转换为 targets（存储源 URL）
    for i, result in enumerate(graph_results):
        info = to_process[i]
        img_filename = info["local_name"]
        src_url = info["url"]

        fg_colors = result["foreground"]["main_colors"]
        L_fg, a_fg, b_fg = process_colors(fg_colors, info, "fg")
        if L_fg is None:
            print(f"  跳过 {img_filename}")
            continue

        bg_colors = result["background"]["main_colors"]
        L_bg, a_bg, b_bg = process_colors(bg_colors, info, "bg")
        if L_bg is None:
            print(f"  跳过 {img_filename}（背景阶段）")
            continue

        batch_targets[src_url] = {
            "L_fg": round(L_fg, 1), "a_fg": round(a_fg, 1), "b_fg": round(b_fg, 1),
            "L_bg": round(L_bg, 1), "a_bg": round(a_bg, 1), "b_bg": round(b_bg, 1),
            "pixiv_id": info["pid"],
            "title": info["title"],
            "author": info["author"],
            "tags": info["tags"],
        }

    # 保存断点
    save_batch_checkpoint(batch_targets, TEMP_DIR)
    return batch_targets


# ─── 主循环 ──────────────────────────────────────────────────────────────

def main():
    print("Pixiv 蒸馏目标生成器")
    print(f"  每次下载 {BATCH_SIZE} 张，处理前 {PROCESS_COUNT} 张")
    print(f"  按 Ctrl+C 可安全退出\n")

    batch_targets = {}

    # 检查断点
    cp = load_batch_checkpoint()
    if cp:
        print(f">>> 发现断点，加载已有 {len(cp['targets'])} 条记录...")
        batch_targets = cp["targets"]
        # 清理上次残留的临时文件
        if os.path.exists(TEMP_DIR):
            shutil.rmtree(TEMP_DIR)

    # 捕获 Ctrl+C
    def handle_interrupt(signum, frame):
        print("\n\n>>> 收到中断信号，保存进度后退出...")
        save_batch_checkpoint(batch_targets, TEMP_DIR)
        if os.path.exists(TEMP_DIR):
            shutil.rmtree(TEMP_DIR, ignore_errors=True)
        # 合并到最终文件
        merge_to_final(batch_targets)
        print(">>> 进度已保存。重新运行脚本可继续。")
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_interrupt)

    batch_num = 0
    while True:
        batch_num += 1
        print(f"\n{'#'*60}")
        print(f"  第 {batch_num} 批次 (当前总计 {len(batch_targets)} 条)")
        print(f"{'#'*60}")

        batch_targets = process_one_batch(batch_targets)

        # 4. 保存一版 target_pixiv_{time}.json
        merge_to_final(batch_targets)

        # 5. 移除临时图片
        if os.path.exists(TEMP_DIR):
            shutil.rmtree(TEMP_DIR, ignore_errors=True)
            print(f"  临时文件已清理")

        print(f"\n  第 {batch_num} 批次完成，当前总计 {len(batch_targets)} 条")
        print(f"  按 Ctrl+C 退出，或等待自动开始下一批次...")
        time.sleep(2)  # 短暂停顿


def merge_to_final(batch_targets):
    """合并所有批次结果到最终 JSON 文件"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"targets_pixiv_{timestamp}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(batch_targets, f, indent=2, ensure_ascii=False)
    print(f"\n  已保存: {output_path} ({len(batch_targets)} 条)")


if __name__ == "__main__":
    main()
