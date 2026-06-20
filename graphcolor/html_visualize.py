"""
HTML 可视化模块。

将处理结果生成静态 HTML 页面，支持流式输出（处理一张追加一张）。
每张图片展示：
  - 原图
  - 主体提取图
  - 前景主色（Top 5，按 score 排序）
  - 背景主色（Top 5，按 score 排序）
"""
from pathlib import Path
from typing import List, Optional
import html
import json


class HTMLVisualizer:
    """HTML 可视化生成器，支持流式写入。"""

    # HTML 头部模板
    HTML_HEAD = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>GraphColor 结果可视化</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; background: #f5f5f5; color: #333; padding: 20px; }
  h1 { text-align: center; margin-bottom: 30px; font-size: 24px; color: #222; }
  .stats { text-align: center; margin-bottom: 20px; color: #666; font-size: 14px; }
  .card { background: #fff; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin-bottom: 24px; overflow: hidden; }
  .card-header { padding: 12px 16px; border-bottom: 1px solid #eee; font-weight: 600; font-size: 15px; }
  .card-body { padding: 16px; }
  .images { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 16px; }
  .img-wrapper { flex: 1; min-width: 250px; }
  .img-wrapper img { width: 100%; border-radius: 6px; object-fit: contain; max-height: 400px; background: #f0f0f0; }
  .img-label { text-align: center; font-size: 13px; color: #888; margin-top: 6px; }
  .colors-section { display: flex; gap: 24px; flex-wrap: wrap; }
  .colors-block { flex: 1; min-width: 280px; }
  .colors-block h3 { font-size: 14px; margin-bottom: 10px; color: #444; }
  .color-row { display: flex; align-items: center; gap: 10px; margin-bottom: 8px; }
  .color-swatch { width: 48px; height: 48px; border-radius: 6px; border: 1px solid rgba(0,0,0,0.1); flex-shrink: 0; }
  .color-info { flex: 1; }
  .color-hex { font-family: "SF Mono", "Fira Code", monospace; font-size: 13px; font-weight: 500; }
  .color-meta { font-size: 12px; color: #888; margin-top: 2px; }
  .color-score { font-size: 12px; font-weight: 500; color: #444; }
  .score-bar { height: 4px; border-radius: 2px; background: #e0e0e0; margin-top: 4px; }
  .score-bar-fill { height: 100%; border-radius: 2px; background: linear-gradient(90deg, #4a90d9, #67b26f); }
</style>
</head>
<body>
<h1>GraphColor 结果可视化</h1>
<div class="stats" id="stats">共 <span id="total-count">0</span> 张图片</div>
<div id="cards-container">
"""

    HTML_FOOTER = """</div>
<script>
  function updateStats(n) {
    document.getElementById('total-count').textContent = n;
  }
</script>
</body>
</html>
"""

    def __init__(self, output_path: str):
        self.output_path = Path(output_path)
        self.html_dir = self.output_path.parent
        self.count = 0
        self._write_init()

    def _write_init(self):
        """写入 HTML 头部，打开文件准备流式追加。"""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
            f.write(self.HTML_HEAD)

    def add_result(self, result: dict):
        """
        流式追加一条结果到 HTML。

        Args:
            result: 单条 JSON 结果，包含 image, foreground, background 等字段。
        """
        self.count += 1
        card_html = self._render_card(result)
        with open(self.output_path, "a", encoding="utf-8") as f:
            f.write(card_html)

    def close(self):
        """关闭 HTML，写入尾部。"""
        with open(self.output_path, "a", encoding="utf-8") as f:
            f.write(self.HTML_FOOTER)

    def _render_card(self, result: dict) -> str:
        """渲染单张图片的卡片 HTML。"""
        image_path = result.get("image", "")
        img_name = Path(image_path).name

        # 获取主体提取图路径
        seg_visual = self._find_seg_visual(image_path)

        # 获取前景/背景主色
        fg_colors = result.get("foreground", {}).get("main_colors", [])[:5]
        bg_colors = result.get("background", {}).get("main_colors", [])[:5]

        # 构建原图相对路径
        orig_rel = self._rel_path(image_path, self.html_dir)

        cards = []
        cards.append('<div class="card">')
        cards.append(f'<div class="card-header">{html.escape(img_name)}</div>')
        cards.append('<div class="card-body">')

        # 图片区域
        cards.append('<div class="images">')
        cards.append('<div class="img-wrapper">')
        cards.append(f'<img src="{orig_rel}" alt="原图">')
        cards.append('<div class="img-label">原图</div>')
        cards.append('</div>')

        if seg_visual:
            seg_rel = self._rel_path(seg_visual, self.html_dir)
            cards.append('<div class="img-wrapper">')
            cards.append(f'<img src="{seg_rel}" alt="主体提取">')
            cards.append('<div class="img-label">主体提取</div>')
            cards.append('</div>')

        cards.append('</div>')

        # 颜色区域
        cards.append('<div class="colors-section">')

        # 前景主色
        cards.append('<div class="colors-block">')
        cards.append('<h3>前景主色</h3>')
        for c in fg_colors:
            cards.append(self._render_color_row(c))
        cards.append('</div>')

        # 背景主色
        cards.append('<div class="colors-block">')
        cards.append('<h3>背景主色</h3>')
        for c in bg_colors:
            cards.append(self._render_color_row(c))
        cards.append('</div>')

        cards.append('</div>')  # .colors-section

        cards.append('</div>')  # .card-body
        cards.append('</div>')  # .card

        return "\n".join(cards)

    @staticmethod
    def _render_color_row(color: dict) -> str:
        """渲染单个颜色行。"""
        hex_color = color.get("hex", "#000000")
        score = color.get("score", 0)
        proportion = color.get("proportion", 0)
        lab = color.get("lab", [0, 0, 0])

        score_pct = f"{score:.3f}"
        prop_pct = f"{proportion:.1%}"

        parts = []
        parts.append('<div class="color-row">')
        parts.append(f'<div class="color-swatch" style="background-color: {hex_color}"></div>')
        parts.append('<div class="color-info">')
        parts.append(f'<div class="color-hex">{hex_color}</div>')
        parts.append(f'<div class="color-meta">Lab({lab[0]:.1f}, {lab[1]:.1f}, {lab[2]:.1f}) | 占比 {prop_pct}</div>')
        parts.append(f'<div class="color-score">Score: {score_pct}</div>')
        parts.append(f'<div class="score-bar"><div class="score-bar-fill" style="width: {score * 100}%"></div></div>')
        parts.append('</div>')
        parts.append('</div>')

        return "\n".join(parts)

    @staticmethod
    def _find_seg_visual(image_path: str) -> Optional[str]:
        """根据原图路径推测主体提取图路径。"""
        orig = Path(image_path)
        name_stem = orig.stem  # e.g. 100236449_p0

        # 尝试 seg_visuals 目录
        base_dir = orig.parent.parent  # extracted_imgs
        seg_dir = base_dir.parent / "outputs" / "seg_visuals"
        if seg_dir.exists():
            # 尝试查找包含 stem 的 png 文件
            candidate = seg_dir / f"{name_stem}_graphcolor.png"
            if candidate.exists():
                return str(candidate)

            # 遍历查找包含 stem 的文件
            for f in seg_dir.iterdir():
                if f.stem.startswith(name_stem) and f.suffix.lower() == ".png":
                    return str(f)

        return None

    @staticmethod
    def _rel_path(abs_path: str, html_dir: Optional[Path] = None) -> str:
        """将绝对路径转换为相对于 HTML 文件的相对路径。"""
        if not abs_path:
            return ""
        p = Path(abs_path).resolve()
        if html_dir is not None:
            html_dir = html_dir.resolve()
            try:
                return str(p.relative_to(html_dir)).replace("\\", "/")
            except ValueError:
                # 不在同一路径树下，手动计算相对路径
                try:
                    common_len = 0
                    for pp, hp in zip(p.parts, html_dir.parts):
                        if pp == hp:
                            common_len += 1
                        else:
                            break
                    if common_len > 0:
                        up_count = len(html_dir.parts) - common_len
                        rel_parts = [".."] * up_count + list(p.parts[common_len:])
                        return "/".join(rel_parts)
                except Exception:
                    pass
        return str(p).replace("\\", "/")


def generate_html_from_json(json_path: str, output_path: str):
    """
    从已有 JSON 结果生成完整 HTML。

    Args:
        json_path: results.json 路径
        output_path: 输出 HTML 路径
    """
    with open(json_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    viz = HTMLVisualizer(output_path)
    for r in results:
        viz.add_result(r)
    viz.close()

    print(f"HTML 已生成: {output_path} (共 {len(results)} 张图片)")
