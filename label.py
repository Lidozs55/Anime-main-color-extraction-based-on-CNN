"""
label.py - GraphColor 标注工具(前后端一体化)

把"教师模型推理 + Web 标注 + 流式数据生产"打包成一条单一入口。
后端是 Flask + graphcolor.GraphColorPipeline,前端是 templates/index.html
的深色主题界面;通过两个队列把"下载/计算"和"标注"解耦,使得即便在网络
或模型卡顿时,Web 端依然能稳定地一张一张地走完人工标注流程。

四种运行模式:
  1. 无参数                → Pixiv 流式标注(下载到 pixiv_temp/,结束清空)
  2. --quick               → Pixiv 持久化标注(下载到 pixiv_img/,跳过时删除)
  3. <图片路径>            → 本地图片标注(流式调用 graphcolor)
  4. <results.json 路径>   → 本地 results 标注(直接用 results.json 数据)

输出:
  targets_{TIME}.json      本次会话唯一的标签文件(增量写)
  label_progress.json      断点文件,含 session_id / pool / queue / stats

Web 端:  http://localhost:5000
"""
import argparse
import json
import logging
import os
import queue
import shutil
import signal
import sys
import threading
import time
import urllib.request
import webbrowser
import zipfile
from datetime import datetime
from glob import glob
from pathlib import Path

from PIL import Image
from flask import Flask, jsonify, request, send_from_directory

# 项目根目录
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from graphcolor import GraphColorPipeline  # noqa: E402

# ─── 配置 ───────────────────────────────────────────────────────────────
# 常量集中放这里,方便按需调整(线程数 / 缓冲大小 / 路径等)
PIXIV_API = "https://api.lolicon.app/setu/v2"
PIXIV_TEMP_DIR = os.path.join(ROOT, "pixiv_temp")    # 默认模式临时目录,退出清空
PIXIV_IMG_DIR = os.path.join(ROOT, "pixiv_img")      # --quick 模式持久化目录
LOCAL_IMG_DIR = os.path.join(ROOT, "img")            # 本地图片标注目录
TEMPLATES_DIR = os.path.join(ROOT, "templates")
BREAKPOINT_FILE = os.path.join(ROOT, "label_progress.json")

# 流式管线的并发 / 缓冲参数
DOWNLOAD_THREADS = 4    # Pixiv 模式下的并发下载线程数
POOL_TARGET = 3         # 下载池(已下载未计算)的目标大小
POOL_MAX = 5            # 下载池上限(避免一次性拉太多)
ANNOTATION_BUFFER = 5   # 已计算待标注的缓冲大小
URL_BATCH = 10          # 每次从 lolicon 拉取的 URL 批大小

IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff', '.tif'}


# ─── 工具函数 ───────────────────────────────────────────────────────────

def lab_to_rgb(L, a, b):
    """Lab 转 RGB（用于前端色块显示）。"""
    xn, yn, zn = 95.047, 100.0, 108.883
    fy = (L + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b / 200.0
    delta = 6.0 / 29.0

    def f_inv(t):
        return t ** 3 if t > delta else 3 * delta * delta * (t - 4.0 / 29.0)

    x = xn * f_inv(fx)
    y = yn * f_inv(fy)
    z = zn * f_inv(fz)
    xr, yr, zr = x / 100.0, y / 100.0, z / 100.0
    rl = 3.2406 * xr - 1.5372 * yr - 0.4986 * zr
    gl = -0.9689 * xr + 1.8758 * yr + 0.0415 * zr
    bl = 0.0557 * xr - 0.2040 * yr + 1.0570 * zr

    def gamma(c):
        return 1.055 * (c ** (1.0 / 2.4)) - 0.055 if c > 0.0031308 else 12.92 * c

    r = max(0, min(255, int(round(gamma(rl) * 255))))
    g = max(0, min(255, int(round(gamma(gl) * 255))))
    b_val = max(0, min(255, int(round(gamma(bl) * 255))))
    return (r, g, b_val)


def hex_to_lab(hex_color):
    """Hex 颜色字符串转 Lab。"""
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    rl, gl, bl = r / 255.0, g / 255.0, b / 255.0

    def srgb_to_linear(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    rl, gl, bl = srgb_to_linear(rl), srgb_to_linear(gl), srgb_to_linear(bl)
    x = (0.4124564 * rl + 0.3575761 * gl + 0.1804375 * bl) / 0.95047
    y = (0.2126729 * rl + 0.7151522 * gl + 0.0721750 * bl) / 1.00000
    z = (0.0193339 * rl + 0.1191920 * gl + 0.9503041 * bl) / 1.08883

    def f(t):
        return t ** (1.0 / 3.0) if t > 0.008856 else 7.787 * t + 16.0 / 116.0

    fx, fy, fz = f(x), f(y), f(z)
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b_lab = 200.0 * (fy - fz)
    return (L, a, b_lab)


def expand_paths(patterns, extract_dir=None):
    """展开文件路径，支持通配符、目录、zip。"""
    paths = []
    for p in patterns:
        expanded = glob(p, recursive=True)
        if expanded:
            paths.extend(expanded)
        else:
            paths.append(p)

    seen = set()
    result = []
    for p in paths:
        path = Path(p)
        # zip 处理
        if path.suffix.lower() == ".zip" and path.exists():
            target_root = Path(extract_dir) if extract_dir else path.with_name(f"{path.stem}_extracted")
            target_root = target_root.resolve()
            target_root.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(path, "r") as zf:
                for member in zf.infolist():
                    name = member.filename
                    if member.is_dir() or Path(name).suffix.lower() not in IMAGE_EXTS:
                        continue
                    dest = (target_root / name).resolve()
                    if target_root not in dest.parents and dest != target_root:
                        continue
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(member, "r") as src, open(dest, "wb") as out:
                        shutil.copyfileobj(src, out)
                    abs_p = str(dest.resolve())
                    if abs_p not in seen:
                        seen.add(abs_p)
                        result.append(abs_p)
            continue

        if path.is_dir():
            candidates = [str(child) for child in path.rglob("*")
                          if child.suffix.lower() in IMAGE_EXTS]
        else:
            candidates = [str(path)]

        for candidate in candidates:
            abs_p = str(Path(candidate).resolve())
            if abs_p not in seen and Path(candidate).suffix.lower() in IMAGE_EXTS:
                seen.add(abs_p)
                result.append(abs_p)
    return result


def get_resource_path(relative_path):
    """兼容 PyInstaller 打包后的资源路径。"""
    if getattr(sys, 'frozen', False):
        base = sys._MEIPASS
    else:
        base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, relative_path)


# ─── Pixiv API ──────────────────────────────────────────────────────────

def fetch_pixiv_urls(count=10):
    """通过 lolicon API 获取随机 Pixiv 图片 URL。"""
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


# ─── 断点管理 ────────────────────────────────────────────────────────────

def save_checkpoint(state):
    state = dict(state)
    state["saved_at"] = datetime.now().isoformat()
    with open(BREAKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def load_checkpoint():
    if os.path.exists(BREAKPOINT_FILE):
        try:
            with open(BREAKPOINT_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, KeyError):
            print(">>> 断点文件损坏，将重新开始")
            os.remove(BREAKPOINT_FILE)
    return None


def merge_to_final(targets, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(targets, f, indent=2, ensure_ascii=False)


def migrate_confidence(path):
    """迁移旧版 targets 文件中的置信度。"""
    if not os.path.exists(path):
        print(f"未找到 {path}")
        return
    with open(path, 'r', encoding='utf-8') as f:
        targets = json.load(f)
    fixed_human = 0
    fixed_convert = 0
    for _key, entry in targets.items():
        if not isinstance(entry, dict):
            continue
        for k in ['fg_conf', 'bg_conf']:
            if k not in entry:
                continue
            old = entry[k]
            if old < 0.52:
                entry[k] = 1.0
                fixed_human += 1
            elif old < 0.999:
                new_conf = old / (2 * (1 - old))
                entry[k] = round(min(new_conf, 1.0), 4)
                fixed_convert += 1
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(targets, f, indent=2, ensure_ascii=False)
    print(f"已迁移 {len(targets)} 条数据: {fixed_human} 条人类标注→1.0, {fixed_convert} 条公式转换: {path}")


# ─── Web 标注服务器 ──────────────────────────────────────────────────────

class WebAnnotationServer:
    """Flask Web 标注服务器，管理待标注任务和用户交互。"""

    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.lock = threading.Lock()
        self.current_task = None
        self.current_event = None
        self.current_result = None
        self.port = 5000
        self.app = Flask(__name__)
        self.app.logger.disabled = True
        logging.getLogger('werkzeug').setLevel(logging.ERROR)
        self._setup_routes()

    def _setup_routes(self):
        @self.app.route('/')
        def index():
            html_path = get_resource_path('templates/index.html')
            with open(html_path, 'r', encoding='utf-8') as f:
                html = f.read()
            response = self.app.make_response(html)
            response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
            response.headers['Pragma'] = 'no-cache'
            response.headers['Expires'] = '0'
            return response

        @self.app.route('/api/current')
        def api_current():
            with self.lock:
                task = self.current_task
            if task is None:
                return jsonify({"has_task": False})
            return jsonify({
                "has_task": True,
                "image_name": task["img_info"]["local_name"],
                "task_name": task["img_info"]["local_name"],
                "region_type": task["region_type"],
                "region_label": task["region_label"],
                "trigger_reason": task["trigger_reason"],
                "candidates": task["candidates"],
            })

        @self.app.route('/api/submit', methods=['POST'])
        def api_submit():
            data = request.get_json()
            if not data or 'hex' not in data:
                return jsonify({"ok": False, "error": "缺少 hex 参数"}), 400
            hex_color = data['hex']
            if (not isinstance(hex_color, str) or not hex_color.startswith('#')
                    or len(hex_color) != 7):
                return jsonify({"ok": False, "error": "无效的颜色格式"}), 400
            with self.lock:
                if self.current_task is None:
                    return jsonify({"ok": False, "error": "没有待标注任务"}), 400
                task = self.current_task
            L, a, b = hex_to_lab(hex_color)
            candidates = task["candidates"]
            conf = 1.0
            if candidates:
                # 检查是否与某个算法候选匹配，匹配则使用其置信度
                min_dist = float('inf')
                closest = None
                for c in candidates:
                    d = ((L - c["lab"][0])**2 + (a - c["lab"][1])**2
                         + (b - c["lab"][2])**2) ** 0.5
                    if d < min_dist:
                        min_dist = d
                        closest = c
                if min_dist < 1.0 and closest is not None:
                    conf = closest.get("confidence", 1.0)
            with self.lock:
                self.current_result = (L, a, b, conf)
                if self.current_event:
                    self.current_event.set()
            return jsonify({"ok": True})

        @self.app.route('/api/skip', methods=['POST'])
        def api_skip():
            with self.lock:
                if self.current_task is None:
                    return jsonify({"ok": False, "error": "没有待标注任务"}), 400
                self.current_result = None
                if self.current_event:
                    self.current_event.set()
            return jsonify({"ok": True})

        @self.app.route('/api/stats')
        def api_stats():
            return jsonify(self.pipeline.stats)

        @self.app.route('/api/mode')
        def api_mode():
            return jsonify({
                "mode": self.pipeline.mode_label(),
                "quick": self.pipeline.quick,
            })

        @self.app.route('/api/exit', methods=['POST'])
        def api_exit():
            self.pipeline._exiting = True
            with self.lock:
                self.current_result = None
                if self.current_event:
                    self.current_event.set()
            return jsonify({
                "ok": True,
                "message": "系统正在退出，进度已自动保存。",
            })

        @self.app.route('/temp/<path:filename>')
        def serve_temp(filename):
            return send_from_directory(os.path.abspath(self.pipeline.web_root()), filename)

    def annotate(self, task):
        """推送任务到 Web 前端，阻塞等待用户操作。"""
        event = threading.Event()
        with self.lock:
            self.current_task = task
            self.current_event = event
            self.current_result = None
        while not event.wait(timeout=1.0):
            if self.pipeline.shutdown_event.is_set():
                break
        result = self.current_result
        with self.lock:
            self.current_task = None
            self.current_event = None
            self.current_result = None
        return result


# ─── 流水线 ─────────────────────────────────────────────────────────────

class StreamingPipeline:
    """三阶段 + Web 标注主循环。支持 pixiv / local_image / local_results 三种模式。"""

    def __init__(self, mode, quick=False, port=5000, source=None):
        assert mode in ("pixiv", "local_image", "local_results")
        self.mode = mode
        self.quick = quick and mode == "pixiv"
        self.port = port
        self.source = source

        self.targets = {}
        self.lock = threading.Lock()
        self.shutdown_event = threading.Event()
        self._exiting = False

        self.stats = {"downloaded": 0, "computed": 0, "annotated": 0, "skipped": 0}
        self.stats_lock = threading.Lock()

        self.session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_path = os.path.join(ROOT, f"targets_{self.session_id}.json")

        # pixiv 模式使用多阶段队列；本地模式只用 annotation_queue
        if mode == "pixiv":
            self.url_queue = queue.Queue(maxsize=URL_BATCH * 2)
            self.pool = queue.Queue(maxsize=POOL_MAX)
        else:
            self.url_queue = None
            self.pool = None
        self.annotation_queue = queue.Queue(maxsize=ANNOTATION_BUFFER)

        # 教师模型：pixiv 与 local_image 都需要；local_results 不需要
        self.graphcolor = GraphColorPipeline() if mode != "local_results" else None

        self.web_server = WebAnnotationServer(self)
        self.web_server.port = port

    def web_root(self):
        """根据模式返回 Web 服务器提供图片的根目录。"""
        if self.mode == "pixiv":
            return PIXIV_IMG_DIR if self.quick else PIXIV_TEMP_DIR
        if self.mode == "local_image":
            # 优先使用 img/，否则用源路径所在的目录
            return LOCAL_IMG_DIR
        if self.mode == "local_results":
            # 本地 results 模式假设图片已在 img/
            return LOCAL_IMG_DIR
        return LOCAL_IMG_DIR

    def mode_label(self):
        if self.mode == "pixiv":
            return "Pixiv 流式" + (" [Quick]" if self.quick else "")
        if self.mode == "local_image":
            return f"本地图片 ({self.source})"
        if self.mode == "local_results":
            return f"本地 results ({self.source})"
        return self.mode

    # ── 启动 ────────────────────────────────────────────────────────

    def start(self):
        cp = load_checkpoint()
        if cp:
            same_session = cp.get("session_id") == self.session_id
            if same_session:
                # 同会话：恢复 targets/stats
                self.targets = cp.get("targets", {}) or {}
                self.output_path = cp.get("output_path", self.output_path)
                self.stats.update({k: v for k, v in (cp.get("stats") or {}).items()
                                   if k in self.stats})
                print(f">>> 恢复会话 {self.session_id}: {len(self.targets)} 条记录")
            else:
                # 旧会话：备份为 .bak 并开始新会话
                bak_name = f"label_progress_{cp.get('session_id')}.json.bak"
                print(f">>> 上次会话 {cp.get('session_id')} "
                      f"({len(cp.get('targets', {}) or {})} 条) 已备份为 {bak_name}")
                try:
                    shutil.move(BREAKPOINT_FILE, os.path.join(ROOT, bak_name))
                except OSError:
                    pass
                self.targets = {}
        else:
            print(f">>> 新会话 {self.session_id}")
            print(f">>> 输出文件: {os.path.basename(self.output_path)}")

        # 准备目录
        if self.mode == "pixiv":
            if self.quick:
                os.makedirs(PIXIV_IMG_DIR, exist_ok=True)
            else:
                if os.path.isdir(PIXIV_TEMP_DIR):
                    shutil.rmtree(PIXIV_TEMP_DIR, ignore_errors=True)
                os.makedirs(PIXIV_TEMP_DIR, exist_ok=True)
        else:
            os.makedirs(LOCAL_IMG_DIR, exist_ok=True)

        # pixiv 模式：恢复池 + 启动后台线程
        if self.mode == "pixiv":
            self._restore_pixiv_pools(cp if (cp and same_session) else None)
            threading.Thread(target=self._url_fetcher, daemon=True, name="url-fetcher").start()
            for i in range(DOWNLOAD_THREADS):
                threading.Thread(target=self._download_worker, daemon=True,
                                 name=f"dl-{i}").start()
            threading.Thread(target=self._compute_worker, daemon=True, name="compute").start()
        elif self.mode == "local_image":
            self._preload_local_images()
        elif self.mode == "local_results":
            self._preload_local_results()

    def _restore_pixiv_pools(self, cp):
        """从断点恢复下载池与标注队列（仅 pixiv 模式）。"""
        if not cp or not self.pool:
            return
        pool_items = cp.get("pool_items", []) or []
        restored_pool = 0
        for item in pool_items:
            if isinstance(item, dict) and "local_path" in item and os.path.exists(item["local_path"]):
                try:
                    self.pool.put(item, timeout=1)
                    restored_pool += 1
                except queue.Full:
                    break
        annotation_items = cp.get("annotation_items", []) or []
        restored_annotation = 0
        for item in annotation_items:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                result_dict, info = item
                if (isinstance(info, dict) and "local_path" in info
                        and os.path.exists(info["local_path"])):
                    try:
                        from types import SimpleNamespace
                        fg_colors = [SimpleNamespace(lab=c["lab"], score=c["score"])
                                     for c in result_dict["foreground"]["main_colors"]]
                        bg_colors = [SimpleNamespace(lab=c["lab"], score=c["score"])
                                     for c in result_dict["background"]["main_colors"]]
                        result = SimpleNamespace(
                            foreground=SimpleNamespace(main_colors=fg_colors),
                            background=SimpleNamespace(main_colors=bg_colors),
                        )
                        self.annotation_queue.put((result, info), timeout=1)
                        restored_annotation += 1
                    except queue.Full:
                        break
        if restored_pool or restored_annotation:
            print(f">>> 恢复池: 下载池 {restored_pool} 张, 标注队列 {restored_annotation} 张")

    # ── Pixiv 阶段 1：URL 获取 ──────────────────────────────────────

    def _url_fetcher(self):
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

    # ── Pixiv 阶段 2：下载 ──────────────────────────────────────────

    def _download_worker(self):
        while not self.shutdown_event.is_set():
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
                    try:
                        self.pool.put(info, timeout=5)
                        with self.stats_lock:
                            self.stats["downloaded"] += 1
                    except queue.Full:
                        self._cleanup_file(info)
            except Exception as e:
                print(f"  [下载] 失败: {e}")
            finally:
                self.url_queue.task_done()

    def _download_one(self, info):
        src_url = info["url"]
        ext = ".png" if src_url.endswith(".png") else ".jpg"
        local_name = f"pixiv_{info['pid']}_{info['page']}{ext}"
        target_dir = PIXIV_IMG_DIR if self.quick else PIXIV_TEMP_DIR
        local_path = os.path.join(target_dir, local_name)
        req = urllib.request.Request(src_url, headers={
            "User-Agent": "Mozilla/5.0", "Referer": "https://www.pixiv.net/",
        })
        with urllib.request.urlopen(req, timeout=30) as resp:
            with open(local_path, "wb") as f:
                f.write(resp.read())
        img = Image.open(local_path)
        img.verify()
        img.close()
        return local_path

    # ── Pixiv 阶段 3：计算 ──────────────────────────────────────────

    def _compute_worker(self):
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
                self._cleanup_file(info)
            except Exception as e:
                print(f"  [计算] {info.get('local_name', '?')}: {e}")
                self._cleanup_file(info)
            finally:
                self.pool.task_done()

    # ── 本地图片模式：预加载 ─────────────────────────────────────────

    def _preload_local_images(self):
        """本地图片模式：扩展路径 → graphcolor.process → 入队。"""
        paths = expand_paths([self.source])
        if not paths:
            print(f"错误: 未找到任何图片，请检查路径 {self.source}")
            self.shutdown_event.set()
            return
        pending = [p for p in paths if os.path.basename(p) not in self.targets]
        print(f"找到 {len(paths)} 张图片（{len(pending)} 张待标注）")
        if not pending:
            print("全部已标注，下次启动会直接退出。")
            return
        for path in pending:
            if self.shutdown_event.is_set():
                break
            try:
                result = self.graphcolor.process(path)
                info = {
                    "local_path": path,
                    "local_name": os.path.basename(path),
                    "key": os.path.basename(path),
                    "source": "local",
                }
                self.annotation_queue.put((result, info), timeout=600)
                with self.stats_lock:
                    self.stats["computed"] += 1
            except Exception as e:
                print(f"  [计算] {os.path.basename(path)}: {e}")

    # ── 本地 results 模式：预加载 ────────────────────────────────────

    def _preload_local_results(self):
        """results.json 模式：读 JSON → 直接入队（不再计算）。"""
        if not os.path.exists(self.source):
            print(f"错误: {self.source} 不存在")
            self.shutdown_event.set()
            return
        with open(self.source, 'r', encoding='utf-8') as f:
            results = json.load(f)
        if not isinstance(results, list):
            print(f"错误: {self.source} 不是 list of results")
            self.shutdown_event.set()
            return
        pending = []
        for r in results:
            if not isinstance(r, dict):
                continue
            img_path = r.get("image", "")
            key = os.path.basename(img_path)
            if key not in self.targets:
                pending.append(r)
        print(f"找到 {len(results)} 条结果（{len(pending)} 条待标注）")
        if not pending:
            print("全部已标注，下次启动会直接退出。")
            return
        for r in pending:
            if self.shutdown_event.is_set():
                break
            img_path = r.get("image", "")
            key = os.path.basename(img_path)
            info = {
                "local_path": img_path,
                "local_name": key,
                "key": key,
                "source": "results",
            }
            from types import SimpleNamespace

            def colors_to_objs(cs):
                return [SimpleNamespace(lab=list(c["lab"]), score=c["score"])
                        for c in (cs or [])]

            result = SimpleNamespace(
                foreground=SimpleNamespace(
                    main_colors=colors_to_objs(r.get("foreground", {}).get("main_colors", []))),
                background=SimpleNamespace(
                    main_colors=colors_to_objs(r.get("background", {}).get("main_colors", []))),
                skin_info={},
            )
            self.annotation_queue.put((result, info), timeout=600)
            with self.stats_lock:
                self.stats["computed"] += 1

    # ── 阶段 4：标注（主线程）────────────────────────────────────────

    def run(self):
        port = self.web_server.port
        print(f"\n{'='*60}")
        print(f"  模式: {self.mode_label()}")
        print(f"  会话: {self.session_id}")
        print(f"  输出: {os.path.basename(self.output_path)}")
        print(f"  Web:  http://localhost:{port}")
        if self.mode == "pixiv":
            if self.quick:
                print(f"  [Quick] 图片直接下载到 pixiv_img/，跳过时删除")
            else:
                print(f"  [默认] 图片下载到 pixiv_temp/，结束清空")
        print(f"  Ctrl+C 安全退出")
        print(f"{'='*60}\n")

        flask_thread = threading.Thread(
            target=lambda: self.web_server.app.run(
                host='0.0.0.0', port=port, debug=False, use_reloader=False),
            daemon=True, name="flask"
        )
        flask_thread.start()
        time.sleep(1.5)
        try:
            webbrowser.open(f"http://localhost:{port}")
        except Exception:
            print(f"  请手动打开浏览器访问 http://localhost:{port}")

        try:
            while not self.shutdown_event.is_set() and not self._exiting:
                try:
                    result, info = self.annotation_queue.get(timeout=2)
                except queue.Empty:
                    continue
                try:
                    self._annotate(result, info)
                except Exception as e:
                    print(f"  [标注] {info.get('local_name', '?')}: {e}")
                    self._cleanup_file(info)
                finally:
                    self.annotation_queue.task_done()
        except KeyboardInterrupt:
            print("\n\n>>> 收到中断信号...")
        finally:
            try:
                self.save_full_checkpoint()
            except Exception as e:
                print(f">>> 断点保存失败: {e}")
            if self._exiting:
                time.sleep(1)
            self.stop()
            s = self.stats
            print(f">>> 本次统计: 标注 {s['annotated']} 张, 跳过 {s['skipped']} 张")
            print(">>> 重新运行脚本可继续。")

    def _annotate(self, result, info):
        img_name = info["local_name"]
        fg_colors = [{"lab": list(c.lab), "score": c.score}
                     for c in result.foreground.main_colors]
        bg_colors = [{"lab": list(c.lab), "score": c.score}
                     for c in result.background.main_colors]

        # 前景
        fg_result = self._auto_annotate(fg_colors)
        L_fg, a_fg, b_fg, fg_conf = fg_result
        if L_fg is None:
            reason = self._calc_trigger_reason(fg_colors)
            task = self._make_web_task(fg_colors, info, "fg", reason)
            print(f"  {img_name} 前景 → Web 标注 ({reason})")
            annot = self.web_server.annotate(task)
            if annot is None:
                self._handle_skip(info)
                return
            L_fg, a_fg, b_fg, fg_conf = annot

        # 背景
        bg_result = self._auto_annotate(bg_colors)
        L_bg, a_bg, b_bg, bg_conf = bg_result
        if L_bg is None:
            reason = self._calc_trigger_reason(bg_colors)
            task = self._make_web_task(bg_colors, info, "bg", reason)
            print(f"  {img_name} 背景 → Web 标注 ({reason})")
            annot = self.web_server.annotate(task)
            if annot is None:
                self._handle_skip(info)
                return
            L_bg, a_bg, b_bg, bg_conf = annot

        # 保存结果
        with self.lock:
            entry = {
                "L_fg": round(L_fg, 1), "a_fg": round(a_fg, 1), "b_fg": round(b_fg, 1),
                "fg_conf": round(fg_conf, 4),
                "L_bg": round(L_bg, 1), "a_bg": round(a_bg, 1), "b_bg": round(b_bg, 1),
                "bg_conf": round(bg_conf, 4),
            }
            if self.mode == "pixiv" and "url" in info:
                entry["pixiv_id"] = info.get("pid")
                entry["title"] = info.get("title", "")
                entry["author"] = info.get("author", "")
                entry["tags"] = info.get("tags", [])
                key = info["url"]
            else:
                key = info.get("key") or info["local_name"]
            self.targets[key] = entry
            with self.stats_lock:
                self.stats["annotated"] += 1
                n = self.stats["annotated"]

        # 持久化
        merge_to_final(self.targets, self.output_path)
        save_checkpoint(self._state_dict())
        pool_size = self.pool.qsize() if self.pool else 0
        s = self.stats
        print(f"\n  [{n}] {img_name} OK "
              f"(池={pool_size} 待标注={self.annotation_queue.qsize()} "
              f"总={s['annotated']} 跳过={s['skipped']})")

        # 标注完成后清理文件
        self._post_annotate_cleanup(info)

    def _auto_annotate(self, colors):
        n_colors = min(len(colors), 5)
        if n_colors == 0:
            return 50.0, 0.0, 0.0, 1.0
        sorted_colors = sorted(colors, key=lambda c: c["score"], reverse=True)[:n_colors]
        if n_colors >= 2:
            s1, s2 = sorted_colors[0]["score"], sorted_colors[1]["score"]
            confidence = min(s1 / (2 * s2), 1.0) if s2 > 0 else 1.0
        else:
            confidence = 1.0
        if n_colors >= 2:
            s1 = sorted_colors[0]["score"]
            s2 = sorted_colors[1]["score"]
            gap12 = (s1 - s2) / s1 if s1 > 0 else 0
            gap13 = 0
            if n_colors >= 3:
                s3 = sorted_colors[2]["score"]
                gap13 = (s1 - s3) / s1 if s1 > 0 else 0
            if gap12 < 0.06 or (n_colors >= 3 and gap13 < 0.12):
                return None, None, None, None
        lab = sorted_colors[0]["lab"]
        return float(lab[0]), float(lab[1]), float(lab[2]), confidence

    def _calc_trigger_reason(self, colors):
        sorted_colors = sorted(colors, key=lambda c: c["score"], reverse=True)[:5]
        n = len(sorted_colors)
        if n < 2:
            return ""
        s1, s2 = sorted_colors[0]["score"], sorted_colors[1]["score"]
        gap12 = (s1 - s2) / s1 if s1 > 0 else 0
        if gap12 < 0.08:
            return f"gap12={gap12:.3f}<0.08"
        if n >= 3:
            s3 = sorted_colors[2]["score"]
            gap13 = (s1 - s3) / s1 if s1 > 0 else 0
            if gap13 < 0.15:
                return f"gap13={gap13:.3f}<0.15"
        return ""

    def _make_web_task(self, colors, info, region_type, trigger_reason):
        sorted_colors = sorted(colors, key=lambda c: c["score"], reverse=True)[:5]
        region_label = "前景" if region_type == "fg" else "背景"
        candidates = []
        for c in sorted_colors:
            L, a, b = float(c["lab"][0]), float(c["lab"][1]), float(c["lab"][2])
            r, g, b_val = lab_to_rgb(L, a, b)
            hex_color = f"#{r:02x}{g:02x}{b_val:02x}"
            s1 = sorted_colors[0]["score"]
            s2 = sorted_colors[1]["score"] if len(sorted_colors) > 1 else s1
            conf = min(s1 / (2 * s2), 1.0) if s2 > 0 else 1.0
            candidates.append({
                "lab": [round(L, 2), round(a, 2), round(b, 2)],
                "score": round(c["score"], 4),
                "hex": hex_color,
                "confidence": round(conf, 4),
            })
        return {
            "img_info": info,
            "region_type": region_type,
            "region_label": region_label,
            "candidates": candidates,
            "trigger_reason": trigger_reason,
        }

    def _handle_skip(self, info):
        # 关键：quick 模式下从 pixiv_img/ 删除；非 quick 从 pixiv_temp/ 删除
        self._cleanup_file(info)
        with self.stats_lock:
            self.stats["skipped"] += 1
        print(f"  跳过 {info.get('local_name', '?')}")

    def _cleanup_file(self, info):
        """删除图片文件（用于跳过或非 quick 模式清理）。"""
        lp = info.get("local_path")
        if not lp or not os.path.exists(lp):
            return
        try:
            os.remove(lp)
        except OSError:
            pass

    def _post_annotate_cleanup(self, info):
        """标注完成后处理图片文件。quick 模式保留；非 quick 模式删除临时。"""
        lp = info.get("local_path")
        if not lp or not os.path.exists(lp):
            return
        if self.mode == "pixiv":
            if self.quick:
                # 保留至 pixiv_img/ 供 CNN 训练
                return
            # 非 quick：删除临时文件
            try:
                os.remove(lp)
            except OSError:
                pass
        # 本地模式：不动原始图片

    def _state_dict(self):
        state = {
            "session_id": self.session_id,
            "output_path": self.output_path,
            "mode": self.mode,
            "quick": self.quick,
            "targets": self.targets,
            "stats": self.stats,
            "saved_at": datetime.now().isoformat(),
        }
        if self.mode == "pixiv":
            # 下载池
            pool_items = []
            if self.pool is not None:
                while not self.pool.empty():
                    try:
                        pool_items.append(self.pool.get_nowait())
                    except queue.Empty:
                        break
                for item in pool_items:
                    self.pool.put(item)
            state["pool_items"] = pool_items
            # 标注队列（取走后转 dict 保存，再放回）
            save_items = []
            requeue_items = []
            while not self.annotation_queue.empty():
                try:
                    result, info = self.annotation_queue.get_nowait()
                    result_dict = {
                        "foreground": {
                            "main_colors": [
                                {"lab": list(c.lab), "score": float(c.score)}
                                for c in result.foreground.main_colors
                            ]
                        },
                        "background": {
                            "main_colors": [
                                {"lab": list(c.lab), "score": float(c.score)}
                                for c in result.background.main_colors
                            ]
                        },
                    }
                    save_items.append([result_dict, info])
                    requeue_items.append((result, info))
                except queue.Empty:
                    break
            for item in requeue_items:
                self.annotation_queue.put(item)
            state["annotation_items"] = save_items
        return state

    def save_full_checkpoint(self):
        """完整保存：合并 final + 写断点。"""
        merge_to_final(self.targets, self.output_path)
        save_checkpoint(self._state_dict())

    def stop(self):
        self.shutdown_event.set()
        if self.mode == "pixiv" and not self.quick:
            # 非 quick 模式：清理临时目录
            if os.path.isdir(PIXIV_TEMP_DIR):
                shutil.rmtree(PIXIV_TEMP_DIR, ignore_errors=True)


# ─── 模式检测 ──────────────────────────────────────────────────────────

def detect_mode(source, quick):
    """根据 source 与 quick 决定模式。"""
    if source is None:
        return "pixiv"
    if quick:
        print("错误: --quick 仅在无参数 Pixiv 模式下生效")
        return "error_quick_conflict"
    if not os.path.exists(source):
        return "error_path_not_found"
    if source.lower().endswith(".json"):
        try:
            with open(source, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, list) and data and isinstance(data[0], dict) and "foreground" in data[0]:
                return "local_results"
        except (json.JSONDecodeError, OSError):
            pass
    return "local_image"


# ─── 入口 ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="label.py - GraphColor 标注工具（前后端一体）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
模式:
  无参数                 Pixiv 流式标注（下载到 pixiv_temp/，结束清空）
  无参数 --quick         Pixiv 持久化标注（下载到 pixiv_img/，跳过时删除）
  <图片路径>             本地图片标注
  <results.json 路径>    本地 results 标注

示例:
  python label.py                                   # Pixiv 流式
  python label.py --quick                           # Pixiv 持久化
  python label.py img/*.png                         # 本地图片
  python label.py outputs/results.json              # 本地 results
  python label.py --migrate targets.json            # 迁移旧版置信度
        """)
    parser.add_argument('source', nargs='?', default=None,
                        help='本地图片路径（文件/目录/通配符/zip）或 results.json 路径；省略则走 Pixiv 流水线')
    parser.add_argument('--quick', action='store_true',
                        help='Pixiv 持久化模式（仅与无参数 Pixiv 模式搭配）')
    parser.add_argument('--port', type=int, default=5000, help='Web 服务器端口（默认 5000）')
    parser.add_argument('--migrate', type=str, metavar='FILE',
                        help='迁移旧版 targets 文件置信度')
    args = parser.parse_args()

    if args.migrate:
        migrate_confidence(args.migrate)
        return

    mode = detect_mode(args.source, args.quick)
    if mode == "error_quick_conflict" or mode == "error_path_not_found":
        if mode == "error_path_not_found":
            print(f"错误: 路径 {args.source} 不存在")
        sys.exit(1)

    pipeline = StreamingPipeline(
        mode=mode,
        quick=(args.quick and mode == "pixiv"),
        port=args.port,
        source=args.source,
    )

    def handle_interrupt(signum, frame):
        print("\n\n>>> 收到中断信号，正在退出...")
        pipeline.shutdown_event.set()
    signal.signal(signal.SIGINT, handle_interrupt)

    pipeline.start()
    pipeline.run()


if __name__ == "__main__":
    main()
