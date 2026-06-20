"""
将训练好的模型预测结果输出为与 graphcolor 完全一致的预览图：
主图 + 左下角前景/背景两个主色色块。

用法:
    python preview.py [--model best_model.pth] [--output-dir ../outputs/model_previews] [--img-dir ../extracted_imgs/imgs]
"""
import sys, os, glob, argparse, time
import torch
import numpy as np
import cv2
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import ColorNetMasked


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}


def lab_to_bgr(L, a, b):
    """Lab (标量) -> BGR (uint8)"""
    lab_cv2 = np.array([[[L * 255 / 100.0, a + 128, b + 128]]], dtype=np.uint8)
    bgr = cv2.cvtColor(lab_cv2, cv2.COLOR_Lab2BGR)
    return tuple(int(v) for v in bgr[0, 0])


def draw_swatch(bgr_img, fg_lab, bg_lab, swatch_size=None):
    """在图片左下角叠加两个方形色块（前景、背景），与 graphcolor 格式一致。"""
    h, w = bgr_img.shape[:2]
    canvas = bgr_img.copy()

    size = swatch_size or max(28, min(96, int(round(min(h, w) * 0.105))))
    gap = max(4, int(round(size * 0.16)))
    margin = max(8, int(round(size * 0.22)))
    stroke = max(2, int(round(size * 0.045)))

    fg_bgr = lab_to_bgr(*fg_lab)
    bg_bgr = lab_to_bgr(*bg_lab)
    colors = [fg_bgr, bg_bgr]

    y1 = max(0, h - margin - size)
    y2 = min(h, y1 + size)
    for idx, bgr_color in enumerate(colors):
        x1 = margin + idx * (size + gap)
        x2 = min(w, x1 + size)
        if x1 >= w:
            break

        cv2.rectangle(canvas, (x1, y1), (x2 - 1, y2 - 1), bgr_color, cv2.FILLED)
        cv2.rectangle(canvas, (x1, y1), (x2 - 1, y2 - 1), (255, 255, 255), stroke)
        cv2.rectangle(canvas, (x1, y1), (x2 - 1, y2 - 1), (0, 0, 0), max(1, stroke // 2))

    return canvas


def collect_images(*dirs):
    """收集多个目录下所有 jpg/png/webp 图片"""
    paths = []
    for d in dirs:
        if not d or not os.path.isdir(d):
            continue
        for ext in ('.jpg', '.png', '.jpeg', '.webp', '.bmp'):
            paths.extend(glob.glob(os.path.join(d, f'*{ext}')))
            paths.extend(glob.glob(os.path.join(d, f'*{ext.upper()}')))
    return sorted(set(paths))


def main():
    parser = argparse.ArgumentParser(description="模型预测结果可视化")
    parser.add_argument('--model', default=None, help='模型权重路径（默认: student/best_model.pth）')
    parser.add_argument('--output-dir', default=None, help='输出目录（默认: ../outputs/model_previews）')
    parser.add_argument('--img-dir', default=None, help='本地图片目录（默认: ../img）')
    parser.add_argument('--pixiv-dir', default=None, help='Pixiv 图片目录（默认: ../pixiv_img）')
    args = parser.parse_args()

    project_root = os.path.normpath(os.path.join(SCRIPT_DIR, '..'))
    model_path = args.model or os.path.join(SCRIPT_DIR, 'best_model.pth')
    output_dir = args.output_dir or os.path.join(project_root, 'outputs', 'model_previews')
    img_dir = args.img_dir or os.path.join(project_root, 'img')
    pixiv_dir = args.pixiv_dir or os.path.join(project_root, 'pixiv_img')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载模型
    model = ColorNetMasked().to(device)
    ckpt = torch.load(model_path, map_location=device)
    if 'model' in ckpt:
        model.load_state_dict(ckpt['model'])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    # 收集图片：本地 + pixiv
    all_paths = collect_images(img_dir, pixiv_dir)
    if not all_paths:
        print(f"错误: 在 {img_dir} 与 {pixiv_dir} 中未找到任何图片")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    print(f"设备: {device}")
    print(f"模型: {model_path}")
    print(f"待处理: {len(all_paths)} 张图片")
    print(f"输出目录: {output_dir}")

    total_start = time.time()
    success_count = 0
    skip_count = 0

    with torch.no_grad():
        for i, path in enumerate(all_paths):
            name = os.path.basename(path)
            img = cv2.imread(path)
            if img is None:
                print(f"  [{i+1}/{len(all_paths)}] 跳过: {name} (无法读取)")
                skip_count += 1
                continue

            t0 = time.time()

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_resized = img_pil.resize((128, 128), Image.LANCZOS)
            img_tensor = torch.tensor(np.array(img_resized).transpose(2, 0, 1), dtype=torch.float32) / 255.0
            img_tensor = img_tensor.unsqueeze(0).to(device)

            pred_fg, pred_bg, _ = model(img_tensor)

            fg_lab = pred_fg[0].cpu().numpy()
            bg_lab = pred_bg[0].cpu().numpy()

            out_img = draw_swatch(img, fg_lab, bg_lab)

            out_path = os.path.join(output_dir, name)
            cv2.imwrite(out_path, out_img)

            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(all_paths)}] OK  {name}  {elapsed*1000:.0f}ms  "
                  f"FG=({fg_lab[0]:.0f},{fg_lab[1]:.0f},{fg_lab[2]:.0f})  "
                  f"BG=({bg_lab[0]:.0f},{bg_lab[1]:.0f},{bg_lab[2]:.0f})")
            success_count += 1

    total_elapsed = time.time() - total_start
    avg_per_image = total_elapsed / success_count if success_count > 0 else 0

    print(f"\n完成！")
    print(f"  成功: {success_count} 张  跳过: {skip_count} 张")
    print(f"  总耗时: {total_elapsed:.2f}s")
    print(f"  平均每张: {avg_per_image*1000:.0f}ms")
    print(f"  预览图已保存至: {output_dir}")


if __name__ == '__main__':
    main()
