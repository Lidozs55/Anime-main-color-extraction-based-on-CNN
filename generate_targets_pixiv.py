#!/usr/bin/env python
"""
Pixiv 蒸馏目标生成器（流式多线程版）

三阶段流水线，各阶段瓶颈不同，自动错峰并行：
  1. 下载（4 线程，I/O 密集）── 网络等待时计算线程在工作
  2. 计算主色（1 线程，CPU 密集）── 计算时下载线程在拉取下一批
  3. 人工标注（主线程，交互）── 用户看图时下载和计算同时进行

缓冲池自动调节：
  - 下载池维持 ~20 张待处理图片，不足时自动补充
  - 标注队列最多缓冲 5 张，满时自动暂停下载和计算

按 Ctrl+C 可随时安全退出。
"""

import json
import math
import os
import queue
import shutil
import signal
import sys
import tempfile
import threading
import time
import urllib.request
import urllib.parse
from datetime import datetime
from pathlib import Path
from PIL import Image, ImageDraw


# ─── 配置 ───────────────────────────────────────────────────────────────
PIXIV_API = "https://api.lolicon.app/setu/v2"
TEMP_DIR = "pixiv_temp"
BREAKPOINT_FILE = "targets_pixiv_progress.json"

DOWNLOAD_THREADS = 4       # 下载线程数
POOL_TARGET = 10           # 下载池目标：低于此值时开始补充
POOL_MAX = 15              # 下载池上限
ANNOTATION_BUFFER = 5      # 标注队列最大缓冲
URL_BATCH = 10             # 每次从 API 获取的 URL 数

sys.path.insert(0, str(Path(__file__).parent))
from graphcolor import GraphColorPipeline


# ─── 工具函数 ───────────────────────────────────────────────────────────

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


# ─── Pixiv API ──────────────────────────────────────────────────────────

def fetch_pixiv_urls(count=10):
    """通过 lolicon API 获取随机 Pixiv 图片 URL"""
    url = f"{PIXIV_API}?r18=0&num={count}&size=original"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode())

    urls = []
    for item in data.get("data", []):
        original = item.get("urls", {}).get("original")
        regular = item.get("urls", {}).get("regular")
        if original or regular:
            urls.append({
                "url": original or regular,
                "pid": item.get("pid"),
                "title": item.get("title", ""),
                "author": item.get("author", ""),
                "tags": item.get("tags", [])[:5],
                "page": item.get("page", 0),
            })
    return urls


# ─── 预览与交互 ─────────────────────────────────────────────────────────

def create_preview(img_path, colors):
    """创建预览图：原始图片 + N 个颜色色块"""
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
        draw.text((20, y_pos + 5), f"#{i+1}", fill=(0, 0, 0) if (r + g + b_val) > 382 else (255, 255, 255))
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


def _fmt_lab(lab):
    """格式化 Lab 值为简洁字符串，去掉 np.float32 包装"""
    return f"({float(lab[0]):.1f},{float(lab[1]):.1f},{float(lab[2]):.1f})"


def process_colors(colors, img_info, region_type):
    """处理前景或背景颜色，返回 (L, a, b, confidence) 或 (None, None, None, None)"""
    n_colors = min(len(colors), 5)
    if n_colors == 0:
        return 50.0, 0.0, 0.0, 1.0

    sorted_colors = sorted(colors, key=lambda c: c["score"], reverse=True)[:n_colors]

    # 计算 teacher 置信度：score1/(2*score2)，clip 到 1.0
    if n_colors >= 2:
        s1, s2 = sorted_colors[0]["score"], sorted_colors[1]["score"]
        confidence = min(s1 / (2 * s2), 1.0) if s2 > 0 else 1.0
    else:
        confidence = 1.0

    local_name = img_info["local_name"]
    img_path = img_info["local_path"]
    region_label = "前景" if region_type == "fg" else "背景"

    print(f"\n  {local_name} {region_label} conf={confidence:.3f}")

    # 一行显示所有候选
    parts = [f"{i+1}:{_fmt_lab(c['lab'])}({c['score']:.3f})" for i, c in enumerate(sorted_colors)]
    print(f"  {' | '.join(parts)}")

    if n_colors >= 2:
        score1 = sorted_colors[0]["score"]
        score2 = sorted_colors[1]["score"]
        gap12 = (score1 - score2) / score1 if score1 > 0 else 0

        gap13 = 0
        if n_colors >= 3:
            score3 = sorted_colors[2]["score"]
            gap13 = (score1 - score3) / score1 if score1 > 0 else 0

        need_human = gap12 < 0.05 or (n_colors >= 3 and gap13 < 0.20)
        if need_human:
            reason = "gap12<0.05" if gap12 < 0.05 else "gap13<0.20"
            preview_colors = [(float(c["lab"][0]), float(c["lab"][1]), float(c["lab"][2]), c["score"]) for c in sorted_colors]
            canvas = create_preview(img_path, preview_colors)

            print(f"  {reason} → 需要人工选择")
            show_preview_and_wait(canvas, img_info, region_type)

            choice = select_color_index(n_colors)
            if choice == 0:
                return None, None, None, None

            selected_lab = sorted_colors[choice - 1]["lab"]
            print(f"  → 选择第{choice}名 {_fmt_lab(selected_lab)}")
            return float(selected_lab[0]), float(selected_lab[1]), float(selected_lab[2]), 1.0
    else:
        print(f"  仅 {n_colors} 个颜色，自动选择")

    selected_lab = sorted_colors[0]["lab"]
    print(f"  → 自动选择第1名 {_fmt_lab(selected_lab)}")
    return float(selected_lab[0]), float(selected_lab[1]), float(selected_lab[2]), confidence


# ─── 断点管理 ────────────────────────────────────────────────────────────

def save_checkpoint(targets, temp_dir):
    checkpoint = {"targets": targets, "temp_dir": temp_dir}
    with open(BREAKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=2, ensure_ascii=False)


def load_checkpoint():
    if os.path.exists(BREAKPOINT_FILE):
        try:
            with open(BREAKPOINT_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, KeyError):
            print(f">>> 断点文件损坏，将重新开始")
            os.remove(BREAKPOINT_FILE)
    return None


def merge_to_final(targets, output_path):
    """保存到最终 JSON 文件"""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(targets, f, indent=2, ensure_ascii=False)
    print(f"\n  已保存: {output_path} ({len(targets)} 条)")


# ─── 流水线 ─────────────────────────────────────────────────────────────

class StreamingPipeline:
    """
    三阶段流水线：下载 → 计算 → 标注

    各阶段通过有界队列连接，背压自动传播：
      标注队列满 → 计算线程阻塞 → 下载池满 → 下载线程暂停
      标注队列空 → 计算线程工作 → 下载池空 → 下载线程补充
    """

    def __init__(self, quick=False):
        self.quick = quick
        self.targets = {}
        self.lock = threading.Lock()
        self.shutdown_event = threading.Event()
        self.graphcolor = GraphColorPipeline()

        # 有界队列
        self.url_queue = queue.Queue(maxsize=URL_BATCH * 2)
        self.pool = queue.Queue(maxsize=POOL_MAX)
        self.annotation_queue = queue.Queue(maxsize=ANNOTATION_BUFFER)

        # 统计
        self.stats = {"downloaded": 0, "computed": 0, "annotated": 0, "skipped": 0}
        self.stats_lock = threading.Lock()

        # 输出路径（启动时确定，全程复用）
        self.output_path = f"targets_pixiv_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    # ── 启动 ──────────────────────────────────────────────────────────

    def start(self):
        """加载断点并启动后台线程"""
        cp = load_checkpoint()
        if cp:
            self.targets = cp["targets"]
            print(f">>> 恢复 {len(self.targets)} 条记录")

        os.makedirs(TEMP_DIR, exist_ok=True)

        threading.Thread(target=self._url_fetcher, daemon=True, name="url-fetcher").start()
        for i in range(DOWNLOAD_THREADS):
            threading.Thread(target=self._download_worker, daemon=True, name=f"dl-{i}").start()
        threading.Thread(target=self._compute_worker, daemon=True, name="compute").start()

    # ── 阶段 1：URL 获取 ──────────────────────────────────────────────

    def _url_fetcher(self):
        """持续从 lolicon API 获取 URL，保持 url_queue 有货"""
        buffer = []
        while not self.shutdown_event.is_set():
            if not buffer:
                try:
                    buffer = fetch_pixiv_urls(URL_BATCH)
                except Exception as e:
                    print(f"  [URL] 获取失败: {e}")
                    self.shutdown_event.wait(3)
                    continue

            try:
                self.url_queue.put(buffer.pop(0), timeout=1)
            except queue.Full:
                self.shutdown_event.wait(0.5)

    # ── 阶段 2：下载 ─────────────────────────────────────────────────

    def _download_worker(self):
        """下载线程：池不足时自动补充"""
        while not self.shutdown_event.is_set():
            # 池已够用，等待
            if self.pool.qsize() >= POOL_TARGET:
                self.shutdown_event.wait(1)
                continue

            try:
                info = self.url_queue.get(timeout=2)
            except queue.Empty:
                continue

            try:
                path = self._download_one(info)
                if path:
                    info["local_path"] = path
                    info["local_name"] = os.path.basename(path)
                    self.pool.put(info, timeout=5)
                    with self.stats_lock:
                        self.stats["downloaded"] += 1
            except Exception as e:
                print(f"  [下载] 失败: {e}")
            finally:
                self.url_queue.task_done()

    def _download_one(self, info):
        """下载单张图片并验证"""
        src_url = info["url"]
        ext = ".png" if src_url.endswith(".png") else ".jpg"
        local_name = f"pixiv_{info['pid']}_{info['page']}{ext}"
        local_path = os.path.join(TEMP_DIR, local_name)

        req = urllib.request.Request(src_url, headers={
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://www.pixiv.net/",
        })
        with urllib.request.urlopen(req, timeout=30) as resp:
            with open(local_path, "wb") as f:
                f.write(resp.read())

        img = Image.open(local_path)
        img.verify()
        img.close()
        return local_path

    # ── 阶段 3：计算 ─────────────────────────────────────────────────

    def _compute_worker(self):
        """计算线程：从池中取图，运行 graphcolor，结果入标注队列"""
        while not self.shutdown_event.is_set():
            try:
                info = self.pool.get(timeout=2)
            except queue.Empty:
                continue

            try:
                result = self.graphcolor.process(info["local_path"])
                self.annotation_queue.put((result, info), timeout=600)
                with self.stats_lock:
                    self.stats["computed"] += 1
            except queue.Full:
                # 10 分钟标注队列未消费，丢弃
                self._cleanup(info)
            except Exception as e:
                print(f"  [计算] {info.get('local_name', '?')}: {e}")
                self._cleanup(info)
            finally:
                self.pool.task_done()

    # ── 阶段 4：标注（主线程）────────────────────────────────────────

    def run(self):
        """主循环：从标注队列取结果，交互标注"""
        print(f"\n{'='*60}")
        print(f"  流水线已启动")
        print(f"  下载线程: {DOWNLOAD_THREADS} | 池目标: {POOL_TARGET} | 标注缓冲: {ANNOTATION_BUFFER}")
        if self.quick:
            print(f"  [Quick模式] 标注完成的图片保留至 extracted_imgs/imgs/pixiv_imgs/")
        print(f"  按 Ctrl+C 安全退出")
        print(f"{'='*60}\n")

        while not self.shutdown_event.is_set():
            try:
                result, info = self.annotation_queue.get(timeout=2)
            except queue.Empty:
                continue

            try:
                self._annotate(result, info)
            except Exception as e:
                print(f"  [标注] {info.get('local_name', '?')}: {e}")
                self._cleanup(info)
            finally:
                self.annotation_queue.task_done()

    def _annotate(self, result, info):
        """标注单张图片的前景和背景"""
        img_name = info["local_name"]

        # 转换颜色格式（MainColor → dict）
        fg_colors = [{"lab": list(c.lab), "score": c.score} for c in result.foreground.main_colors]
        bg_colors = [{"lab": list(c.lab), "score": c.score} for c in result.background.main_colors]

        # 前景标注
        L_fg, a_fg, b_fg, fg_conf = process_colors(fg_colors, info, "fg")
        if L_fg is None:
            self._handle_skip(info)
            return

        # 背景标注
        L_bg, a_bg, b_bg, bg_conf = process_colors(bg_colors, info, "bg")
        if L_bg is None:
            self._handle_skip(info)
            return

        # 保存结果
        with self.lock:
            self.targets[info["url"]] = {
                "L_fg": round(L_fg, 1), "a_fg": round(a_fg, 1), "b_fg": round(b_fg, 1),
                "fg_conf": round(fg_conf, 4),
                "L_bg": round(L_bg, 1), "a_bg": round(a_bg, 1), "b_bg": round(b_bg, 1),
                "bg_conf": round(bg_conf, 4),
                "pixiv_id": info["pid"],
                "title": info["title"],
                "author": info["author"],
                "tags": info["tags"],
            }
            with self.stats_lock:
                self.stats["annotated"] += 1
                n = self.stats["annotated"]

        # 处理图片文件
        if self.quick:
            pixiv_dir = os.path.join("extracted_imgs", "imgs", "pixiv_imgs")
            os.makedirs(pixiv_dir, exist_ok=True)
            dst = os.path.join(pixiv_dir, info["local_name"])
            if not os.path.exists(dst):
                shutil.move(info["local_path"], dst)
            else:
                os.remove(info["local_path"])
        else:
            os.remove(info["local_path"])

        # 持久化
        save_checkpoint(self.targets, TEMP_DIR)
        merge_to_final(self.targets, self.output_path)

        s = self.stats
        print(f"\n  [{n}] {img_name} OK "
              f"(池={self.pool.qsize()} 待标注={self.annotation_queue.qsize()} "
              f"总={s['annotated']} 跳过={s['skipped']})")

    def _handle_skip(self, info):
        self._cleanup(info)
        with self.stats_lock:
            self.stats["skipped"] += 1
        print(f"  跳过 {info['local_name']}")

    def _cleanup(self, info):
        if "local_path" in info and os.path.exists(info["local_path"]):
            os.remove(info["local_path"])

    # ── 停止 ─────────────────────────────────────────────────────────

    def stop(self):
        """优雅停止：保存进度、清理临时文件"""
        self.shutdown_event.set()
        save_checkpoint(self.targets, TEMP_DIR)
        if self.targets:
            merge_to_final(self.targets, self.output_path)
        if os.path.exists(TEMP_DIR):
            shutil.rmtree(TEMP_DIR, ignore_errors=True)


# ─── 置信度迁移 ──────────────────────────────────────────────────────────

def migrate_confidence(targets_path):
    """
    迁移旧版 targets 文件中的置信度：
    - conf < 0.52 → 人类标注，设为 1.0
    - 其他：旧公式 score1/(score1+score2) 转换为新公式 min(score1/(2*score2), 1.0)
      推导：confidence_new = confidence_old / (2*(1-confidence_old))
    """
    if not os.path.exists(targets_path):
        print(f"未找到 {targets_path}")
        return

    with open(targets_path, 'r', encoding='utf-8') as f:
        targets = json.load(f)

    fixed_human = 0
    fixed_convert = 0
    for url, entry in targets.items():
        for key in ['fg_conf', 'bg_conf']:
            if key not in entry:
                continue
            old_conf = entry[key]
            if old_conf < 0.52:
                entry[key] = 1.0
                fixed_human += 1
            elif old_conf < 0.999:
                new_conf = old_conf / (2 * (1 - old_conf))
                entry[key] = round(min(new_conf, 1.0), 4)
                fixed_convert += 1

    with open(targets_path, 'w', encoding='utf-8') as f:
        json.dump(targets, f, indent=2, ensure_ascii=False)

    print(f"已迁移 {len(targets)} 条数据: {fixed_human} 条人类标注→1.0, {fixed_convert} 条公式转换: {targets_path}")


# ─── 入口 ───────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Pixiv 蒸馏目标生成器（流式多线程版）")
    parser.add_argument('--quick', action='store_true',
                        help='Quick模式：标注完成的图片保留至 extracted_imgs/imgs/pixiv_imgs/')
    parser.add_argument('--migrate', type=str, metavar='FILE',
                        help='迁移旧版 targets 文件的置信度（conf<0.52 → 1.0）')
    args = parser.parse_args()

    if args.migrate:
        migrate_confidence(args.migrate)
        return

    pipeline = StreamingPipeline(quick=args.quick)

    def handle_interrupt(signum, frame):
        print("\n\n>>> 收到中断信号，保存进度后退出...")
        pipeline.stop()
        s = pipeline.stats
        print(f">>> 本次统计: 标注 {s['annotated']} 张, 跳过 {s['skipped']} 张")
        print(">>> 重新运行脚本可继续。")
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_interrupt)

    pipeline.start()
    pipeline.run()


if __name__ == "__main__":
    main()
