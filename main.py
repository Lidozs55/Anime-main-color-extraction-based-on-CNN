#!/usr/bin/env python
"""
GraphColor - 手绘动漫图片主色提取工具

用法:
    python main.py <图片路径...> [--output result.json] [--max-size 512]

示例:
    python main.py image.png
    python main.py img1.jpg img2.jpg --output results.json
    python main.py "images/*.png" --output batch_result.json
"""
import argparse
import glob
import shutil
import sys
import zipfile
from pathlib import Path
from graphcolor import process_batch


IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff', '.tif'}


def parse_args():
    parser = argparse.ArgumentParser(
        description="GraphColor - 手绘动漫图片主色提取工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py image.png
  python main.py img1.jpg img2.jpg --output results.json
  python main.py "images/*.png" --max-size 256
        """
    )
    parser.add_argument(
        "images", nargs="+",
        help="图片路径，支持通配符（如 *.png）"
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="输出JSON文件路径"
    )
    parser.add_argument(
        "--visual-dir", type=str, default=None,
        help="输出结果图目录；每张图左下角叠加前景/背景两个主色色块"
    )
    parser.add_argument(
        "--seg-visual-dir", type=str, default=None,
        help="输出主体识别可视化图目录；绿色蒙版标注检测到的主体范围"
    )
    parser.add_argument(
        "--html-output", type=str, default=None,
        help="输出 HTML 可视化结果文件路径；包含原图、主体提取图、前景/背景主色"
    )
    parser.add_argument(
        "--extract-dir", type=str, default=None,
        help="当输入为zip时，图片解压目录（默认: <zip名>_extracted）"
    )
    parser.add_argument(
        "--max-size", type=int, default=512,
        help="压缩后图片最长边像素数 (默认: 512)"
    )
    parser.add_argument(
        "--clusters-fg", type=int, default=10,
        help="前景聚类数 (默认: 10)"
    )
    parser.add_argument(
        "--clusters-bg", type=int, default=6,
        help="背景聚类数 (默认: 6)"
    )
    parser.add_argument(
        "--color-weight", type=float, default=3.0,
        help="颜色权重 (相对于亮度), 默认: 3.0"
    )
    parser.add_argument(
        "--lightness-weight", type=float, default=0.2,
        help="亮度权重, 调低使聚类更关注色度而非亮度, 默认: 0.2"
    )
    parser.add_argument(
        "--a-boost", type=float, default=1.1,
        help="a* 通道放大倍数, 强化红绿差异, 默认: 1.1"
    )
    parser.add_argument(
        "--workers", "-j", type=int, default=1,
        help="并行工作进程数 (默认: 1=单线程; 设为CPU核心数可最大化速度)"
    )
    return parser.parse_args()


def extract_zip_images(zip_path: Path, extract_dir: str | None = None) -> list[str]:
    """安全解压 zip 中的图片并返回图片路径。"""
    target_root = Path(extract_dir) if extract_dir else zip_path.with_name(f"{zip_path.stem}_extracted")
    target_root = target_root.resolve()
    target_root.mkdir(parents=True, exist_ok=True)

    extracted: list[str] = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            name = member.filename
            if member.is_dir() or Path(name).suffix.lower() not in IMAGE_EXTS:
                continue

            dest = (target_root / name).resolve()
            if target_root not in dest.parents and dest != target_root:
                raise ValueError(f"zip包含不安全路径: {name}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member, "r") as src, open(dest, "wb") as out:
                shutil.copyfileobj(src, out)
            extracted.append(str(dest))

    return extracted


def expand_paths(patterns: list[str], extract_dir: str | None = None) -> list[str]:
    """展开文件路径（支持通配符和目录遍历）"""
    paths = []
    for p in patterns:
        expanded = glob.glob(p, recursive=True)
        if expanded:
            paths.extend(expanded)
        else:
            # 可能不是通配符，直接加进来
            paths.append(p)

    # 去重并过滤出图片文件
    seen = set()
    result = []
    for p in paths:
        path = Path(p)
        if path.suffix.lower() == ".zip" and path.exists():
            for extracted in extract_zip_images(path, extract_dir):
                abs_p = str(Path(extracted).resolve())
                if abs_p not in seen:
                    seen.add(abs_p)
                    result.append(abs_p)
            continue

        if path.is_dir():
            candidates = [
                str(child) for child in path.rglob("*")
                if child.suffix.lower() in IMAGE_EXTS
            ]
        else:
            candidates = [str(path)]

        for candidate in candidates:
            abs_p = str(Path(candidate).resolve())
            if abs_p not in seen and Path(candidate).suffix.lower() in IMAGE_EXTS:
                seen.add(abs_p)
                result.append(abs_p)
    return result


def main():
    args = parse_args()

    image_paths = expand_paths(args.images, args.extract_dir)
    if not image_paths:
        print("错误: 未找到任何图片文件", file=sys.stderr)
        sys.exit(1)

    print(f"找到 {len(image_paths)} 张图片")
    print(f"配置: max_size={args.max_size}, "
          f"fg_clusters={args.clusters_fg}, bg_clusters={args.clusters_bg}, "
          f"color_weight={args.color_weight}, lightness_weight={args.lightness_weight}")
    print("-" * 60)

    config = {
        "max_size": args.max_size,
        "n_clusters_foreground": args.clusters_fg,
        "n_clusters_background": args.clusters_bg,
        "color_weight": args.color_weight,
        "lightness_weight": args.lightness_weight,
        "a_boost": args.a_boost,
    }

    results = process_batch(
        image_paths,
        config=config,
        output_json=args.output,
        output_preview_dir=args.visual_dir,
        output_seg_visual_dir=args.seg_visual_dir,
        output_html=args.html_output,
        verbose=True,
        workers=args.workers
    )

    print("-" * 60)
    print(f"成功处理 {len(results)} 张图片")


if __name__ == "__main__":
    main()
