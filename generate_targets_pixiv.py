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
BATCH_SIZE = 20          # 每次下载图片数（全部处理）
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


def select_color_index(n_colors):
    """读取用户选择：0=跳过，1~N=选对应名次"""
    options = "/".join(str(i) for i in range(n_colors + 1))
    while True:
        try:
            user_input = input(f"请选择主色编号（{options}，0=跳过）：").strip()
            if user_input not in [str(i) for i in range(n_colors + 1)]:
                print(f"错误：请输入 0~{n_colors}")
                continue
            return int(user_input)
        except ValueError:
            print("错误：请输入有效的整数")


def process_colors(colors, img_info, region_type):
    """处理前景或背景颜色，返回 (L, a, b, confidence) 或 (None, None, None, None)"""
    n_colors = min(len(colors), 5)
    if n_colors == 0:
        return 50.0, 0.0, 0.0, 1.0

    sorted_colors = sorted(colors, key=lambda c: c["score"], reverse=True)[:n_colors]

    # 计算 teacher 置信度
    if n_colors >= 2:
        s1, s2 = sorted_colors[0]["score"], sorted_colors[1]["score"]
        confidence = s1 / (s1 + s2) if (s1 + s2) > 0 else 1.0
    else:
        confidence = 1.0

    local_name = img_info["local_name"]
    img_path = img_info["local_path"]

    print(f"\n  处理 {local_name} - {region_type} (conf={confidence:.3f})")

    # 打印候选颜色
    for i in range(n_colors):
        c = sorted_colors[i]
        label = f"  第{i+1}名: Lab={c['lab']} (score={c['score']:.4f})"
        print(label)

    if n_colors >= 2:
        score1 = sorted_colors[0]["score"]
        score2 = sorted_colors[1]["score"]
        gap12 = (score1 - score2) / score1 if score1 > 0 else 0

        # 检查 score1 与 score3 的差距
        gap13 = 0
        if n_colors >= 3:
            score3 = sorted_colors[2]["score"]
            gap13 = (score1 - score3) / score1 if score1 > 0 else 0

        print(f"  gap12 = {gap12:.4f}  gap13 = {gap13:.4f}")

        need_human = gap12 < 0.05 or (n_colors >= 3 and gap13 < 0.20)
        if need_human:
            reason = "gap12<0.05" if gap12 < 0.05 else "gap13<0.20"
            preview_colors = [(c["lab"][0], c["lab"][1], c["lab"][2], c["score"]) for c in sorted_colors]
            canvas = create_preview(img_path, preview_colors)

            print(f"  {reason}，需要人工确认主色")
            label_str = "  ".join([f"{i+1}: Lab={sorted_colors[i]['lab']}" for i in range(n_colors)])
            print(f"  候选颜色：{label_str}")

            show_preview_and_wait(canvas, img_info, region_type)

            choice = select_color_index(n_colors)
            if choice == 0:
                print(f"  用户选择跳过此图")
                return None, None, None, None

            selected_lab = sorted_colors[choice - 1]["lab"]
            print(f"  用户选择第{choice}名: Lab={selected_lab}")
            return selected_lab[0], selected_lab[1], selected_lab[2], confidence
    else:
        print(f"  仅 {n_colors} 个颜色，自动选择第1名")

    # gap12 >= 0.05 且 gap13 >= 0.20 或仅1个颜色，自动选择第1名
    selected_lab = sorted_colors[0]["lab"]
    print(f"  自动选择第1名: Lab={selected_lab}")
    return selected_lab[0], selected_lab[1], selected_lab[2], confidence


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

def process_one_batch(batch_targets, quick=False):
    """执行一次完整批次：下载 → graphcolor → 生成 targets"""
    print(f"\n{'='*60}")
    print(f"  开始新批次：下载 {BATCH_SIZE} 张 Pixiv 随机图片...")
    if quick:
        print(f"  [Quick模式] 本批次图片将保留至 extracted_imgs/imgs/pixiv_imgs/")
    print(f"{'='*60}")

    # 1. 获取 URL 并下载
    url_infos = fetch_pixiv_urls(BATCH_SIZE)
    if not url_infos:
        print("  未获取到有效图片 URL，跳过此批次")
        return batch_targets, (None if quick else None)

    local_paths, valid_infos = download_images(url_infos, TEMP_DIR)
    if len(local_paths) < 1:
        print("  没有成功下载任何图片，跳过此批次")
        return batch_targets, (None if quick else None)

    to_process = valid_infos
    to_process_paths = [info["local_path"] for info in to_process]
    print(f"\n  共下载 {len(local_paths)} 张，将全部处理...")

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
        return batch_targets, (None if quick else None)

    # 3. 将 results 转换为 targets（存储源 URL）
    kept_paths = set()  # quick模式：记录成功保留的图片路径
    for i, result in enumerate(graph_results):
        info = to_process[i]
        img_filename = info["local_name"]
        src_url = info["url"]

        fg_colors = result["foreground"]["main_colors"]
        L_fg, a_fg, b_fg, fg_conf = process_colors(fg_colors, info, "fg")
        if L_fg is None:
            print(f"  跳过 {img_filename}")
            if quick:
                os.remove(info["local_path"])  # quick模式下删除被跳过的图片
            continue

        bg_colors = result["background"]["main_colors"]
        L_bg, a_bg, b_bg, bg_conf = process_colors(bg_colors, info, "bg")
        if L_bg is None:
            print(f"  跳过 {img_filename}（背景阶段）")
            if quick:
                kept_paths.discard(info["local_path"])  # 从保留列表中移除
                os.remove(info["local_path"])  # quick模式下删除被跳过的图片
            continue

        batch_targets[src_url] = {
            "L_fg": round(L_fg, 1), "a_fg": round(a_fg, 1), "b_fg": round(b_fg, 1),
            "fg_conf": round(fg_conf, 4),
            "L_bg": round(L_bg, 1), "a_bg": round(a_bg, 1), "b_bg": round(b_bg, 1),
            "bg_conf": round(bg_conf, 4),
            "pixiv_id": info["pid"],
            "title": info["title"],
            "author": info["author"],
            "tags": info["tags"],
        }
        if quick:
            kept_paths.add(info["local_path"])

    # 保存断点
    save_batch_checkpoint(batch_targets, TEMP_DIR)
    
    # quick模式返回 kept_paths 供后续转移
    if quick:
        return batch_targets, kept_paths
    return batch_targets, None


# ─── 主循环 ──────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Pixiv 蒸馏目标生成器")
    parser.add_argument('--quick', action='store_true', help='Quick模式：每批次图片保留至 extracted_imgs/imgs/pixiv_imgs/，不删除')
    args = parser.parse_args()
    quick = args.quick

    print("Pixiv 蒸馏目标生成器")
    print(f"  每次下载并处理 {BATCH_SIZE} 张 Pixiv 随机图片")
    if quick:
        print(f"  [Quick模式] 本批次图片将保留至 extracted_imgs/imgs/pixiv_imgs/")
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

        batch_targets, batch_valid_infos = process_one_batch(batch_targets, quick)

        # 4. 保存一版 target_pixiv_{time}.json
        merge_to_final(batch_targets)

        # 5. 处理临时图片
        if quick:
            # Quick模式：仅转移被保留的图片（跳过/失败的图片已删除）
            pixiv_dir = os.path.join('extracted_imgs', 'imgs', 'pixiv_imgs')
            os.makedirs(pixiv_dir, exist_ok=True)
            moved_count = 0
            if batch_valid_infos:
                for src_path in batch_valid_infos:
                    fname = os.path.basename(src_path)
                    dst = os.path.join(pixiv_dir, fname)
                    if not os.path.exists(dst):
                        shutil.move(src_path, dst)
                        moved_count += 1
                    else:
                        os.remove(src_path)
            # 清理临时目录
            if os.path.exists(TEMP_DIR):
                shutil.rmtree(TEMP_DIR, ignore_errors=True)
            print(f"  [Quick模式] 已转移 {moved_count} 张图片至 {pixiv_dir}/")
        else:
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
