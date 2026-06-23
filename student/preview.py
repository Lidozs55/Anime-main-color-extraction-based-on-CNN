"""
将训练好的模型预测结果输出为与 graphcolor 完全一致的预览图:
主图 + 左下角前景/背景两个主色色块。

数据流:
  读图 → [阴影去除] → 128x128 resize → student 模型 → Lab → BGR 色块叠加 → 保存

阴影去除(默认开启):
  student/dataset.py 在训练时同步应用,所以推理时也必须应用,
  否则会出现 train/inference 之间的 domain shift
  (教师给的是去阴影后的 Lab,student 训练时看到原图 → 推理时却是去阴影后图)。

用法:
    python preview.py [--model best_model.pth] [--output-dir ../outputs/model_previews] [--img-dir ../extracted_imgs/imgs] [--no-shadow-removal]
"""
import sys, os, glob, argparse, time
import torch
import numpy as np
import cv2
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# 引入 graphcolor.shadow(在父目录)
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))

from model import ColorNetMasked
from graphcolor.shadow import ShadowRemover


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}


def lab_to_bgr(L, a, b):
    """Lab (标量) -> BGR (uint8 三元组)"""
    lab_cv2 = np.array([[[L * 255 / 100.0, a + 128, b + 128]]], dtype=np.uint8)
    bgr = cv2.cvtColor(lab_cv2, cv2.COLOR_Lab2BGR)
    return tuple(int(v) for v in bgr[0, 0])


def draw_swatch_on_original(original_bgr, fg_lab, bg_lab, swatch_size=None):
    """在 **原图** 左下角叠加两个方形色块(前景、背景)。

    色块大小基于 **原图** 短边: size = min(orig_H, orig_W) * 0.06。
    叠加前不再次去阴影、不再 resize,保证叠加结果与 graphcolor 格式一致。
    """
    h, w = original_bgr.shape[:2]
    canvas = original_bgr.copy()

    size = swatch_size or max(20, int(round(min(h, w) * 0.06)))
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

        # 主色块 + 双层描边(白外 / 黑内),深浅底图都可见
        cv2.rectangle(canvas, (x1, y1), (x2 - 1, y2 - 1), bgr_color, cv2.FILLED)
        cv2.rectangle(canvas, (x1, y1), (x2 - 1, y2 - 1), (255, 255, 255), stroke)
        cv2.rectangle(canvas, (x1, y1), (x2 - 1, y2 - 1), (0, 0, 0), max(1, stroke // 2))

    return canvas


def collect_images(*dirs):
    """收集多个目录下所有 jpg/png/webp/bmp 图片(去重 + 排序)。"""
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
    parser.add_argument('--no-shadow-removal', action='store_true',
                        help='关闭阴影去除(默认开启,与 dataset.py 训练时一致)')
    parser.add_argument('--shadow-sigma-ratio', type=float, default=None,
                        help='覆盖 ShadowRemover 的 sigma_ratio(默认 0.125)')
    parser.add_argument('--full-mode', action='store_true',
                        help='使用完整阴影去除模式(默认快速,跳过连通块+ab检查,全局目标L);'
                             '加此参数切换到标准模式 (色差约束连通块 + 逐块目标L + ab 一致性检查)')
    parser.add_argument('--json', action='store_true',
                        help='输出 JSON 而非预览图(用于程序化消费) -- 同时可指定 --json-name 自定义文件名')
    parser.add_argument('--json-name', default='preview_predictions.json',
                        help='--json 模式下的输出文件名(默认: preview_predictions.json,存于 outputs/)')
    args = parser.parse_args()

    project_root = os.path.normpath(os.path.join(SCRIPT_DIR, '..'))
    model_path = args.model or os.path.join(SCRIPT_DIR, 'best_model.pth')
    img_dir = args.img_dir or os.path.join(project_root, 'img')

    # JSON 模式: 始终写入 ../outputs/<filename>.json(忽略 --output-dir)
    # 预览图模式: 默认 ../outputs/model_previews, 可被 --output-dir 覆盖
    if args.json:
        outputs_root = os.path.join(project_root, 'outputs')
        output_dir = outputs_root
        json_path = os.path.join(outputs_root, args.json_name)
    else:
        output_dir = args.output_dir or os.path.join(project_root, 'outputs', 'model_previews')
        json_path = None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 阴影去除器(必须与 student/dataset.py 训练时一致)
    # 默认 fast_mode=True;--full-mode 切回标准模式
    shadow_kwargs = {"enabled": not args.no_shadow_removal, "fast_mode": not args.full_mode}
    if args.shadow_sigma_ratio is not None:
        shadow_kwargs["sigma_ratio"] = args.shadow_sigma_ratio
    shadow_remover = ShadowRemover(**shadow_kwargs)

    # 加载模型
    model = ColorNetMasked().to(device)
    ckpt = torch.load(model_path, map_location=device)
    if 'model' in ckpt:
        model.load_state_dict(ckpt['model'])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    # 收集图片:仅处理 imgs 目录
    all_paths = collect_images(img_dir)
    if not all_paths:
        print(f"错误: 在 {img_dir} 中未找到任何图片")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    print(f"设备: {device}")
    print(f"模型: {model_path}")
    print(f"阴影去除: {'关闭' if args.no_shadow_removal else '开启'}")
    if not args.no_shadow_removal:
        print(f"阴影去除模式: {'完整' if args.full_mode else '快速'}")
    print(f"输出模式: {'JSON' if args.json else '预览图'}")
    if args.json:
        print(f"JSON 输出: {json_path}")
    else:
        print(f"预览图输出: {output_dir}")
    print(f"待处理: {len(all_paths)} 张图片")

    total_start = time.perf_counter()
    success_count = 0
    skip_count = 0
    t_read_list = []
    t_infer_list = []
    t_write_list = []
    json_records = []  # JSON 模式累积

    # 每 N 张打印一次进度,减少控制台 I/O
    PRINT_EVERY = 20

    def _should_print(idx, total):
        # 0-based idx:首张(i=0)、最后一张、每 PRINT_EVERY 张都打
        return (idx + 1) % PRINT_EVERY == 0 or idx == total - 1

    with torch.no_grad():
        for i, path in enumerate(all_paths):
            name = os.path.basename(path)
            t0 = time.perf_counter()
            original_img = cv2.imread(path)
            if original_img is None:
                print(f"  [{i+1}/{len(all_paths)}] 跳过: {name} (无法读取)")
                skip_count += 1
                continue
            t_io_read = time.perf_counter() - t0

            t1 = time.perf_counter()

            # 推理管线(必须与 train pipeline 完全一致):
            #   原图 → resize_to_max(512) → 去阴影 → resize(128x128) → CNN
            # 注意: 训练时 dataset.py 在去阴影后还会 resize 回原图尺寸给
            #       RandomResizedCrop 采样,但 RandomResizedCrop 的输出已经是
            #       128x128,与本步直接 resize(128) 等价。
            infer_img = original_img
            if max(infer_img.shape[:2]) > 512:
                infer_img = cv2.resize(infer_img, (512, 512),
                                       interpolation=cv2.INTER_AREA)
            infer_img = shadow_remover.remove(infer_img)
            img_rgb = cv2.cvtColor(infer_img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_resized = img_pil.resize((128, 128), Image.LANCZOS)
            img_tensor = torch.tensor(np.array(img_resized).transpose(2, 0, 1), dtype=torch.float32) / 255.0
            img_tensor = img_tensor.unsqueeze(0).to(device)

            pred_fg, pred_bg, _ = model(img_tensor)

            fg_lab = pred_fg[0].cpu().numpy()
            bg_lab = pred_bg[0].cpu().numpy()
            t_infer = time.perf_counter() - t1

            t_io_write = 0.0
            if args.json:
                # JSON 模式: 累积记录,循环结束后一次性写入
                json_records.append({
                    "image": name,
                    "path": os.path.normpath(path),
                    "image_size": [int(original_img.shape[1]), int(original_img.shape[0])],
                    "fg_lab": [round(float(fg_lab[0]), 2),
                               round(float(fg_lab[1]), 2),
                               round(float(fg_lab[2]), 2)],
                    "bg_lab": [round(float(bg_lab[0]), 2),
                               round(float(bg_lab[1]), 2),
                               round(float(bg_lab[2]), 2)],
                    "elapsed_ms": round((t_io_read + t_infer) * 1000, 2),
                    "read_ms": round(t_io_read * 1000, 2),
                    "infer_ms": round(t_infer * 1000, 2),
                })
                if _should_print(i, len(all_paths)):
                    print(f"  [{i+1}/{len(all_paths)}] OK  {name}")
            else:
                t2 = time.perf_counter()
                # 预览图模式: 在 **原图** 上叠加色块(色块大小按原图短边 0.06)
                out_img = draw_swatch_on_original(original_img, fg_lab, bg_lab)
                out_path = os.path.join(output_dir, name)
                cv2.imwrite(out_path, out_img)
                t_io_write = time.perf_counter() - t2
                if _should_print(i, len(all_paths)):
                    print(f"  [{i+1}/{len(all_paths)}] OK  {name}")
            success_count += 1

            # 各分项耗时累加,最后做平均
            t_read_list.append(t_io_read)
            t_infer_list.append(t_infer)
            t_write_list.append(t_io_write)

    # JSON 模式: 一次性写入文件
    if args.json and json_records:
        import json as _json
        payload = {
            "model": os.path.normpath(model_path),
            "shadow_removal": not args.no_shadow_removal,
            "image_count": len(json_records),
            "skipped_count": skip_count,
            "predictions": json_records,
        }
        with open(json_path, 'w', encoding='utf-8') as f:
            _json.dump(payload, f, ensure_ascii=False, indent=2)

    total_elapsed = time.perf_counter() - total_start

    # 分项统计:读盘 / 推理 / 写盘
    def _stat(lst):
        if not lst:
            return 0.0, 0.0
        return (float(np.mean(lst)) * 1000, float(np.median(lst)) * 1000)

    avg_read, med_read = _stat(t_read_list)
    avg_infer, med_infer = _stat(t_infer_list)
    avg_write, med_write = _stat(t_write_list)

    print(f"\n完成！")
    print(f"  成功: {success_count} 张  跳过: {skip_count} 张")
    print(f"  总耗时: {total_elapsed:.2f}s")
    print(f"  读盘   平均/中位: {avg_read:.0f}ms / {med_read:.0f}ms")
    print(f"  推理   平均/中位: {avg_infer:.0f}ms / {med_infer:.0f}ms")
    print(f"  写盘   平均/中位: {avg_write:.0f}ms / {med_write:.0f}ms")
    print(f"  端到端 平均/中位: {(avg_read+avg_infer+avg_write):.0f}ms / "
          f"{(med_read+med_infer+med_write):.0f}ms")
    if args.json:
        print(f"  JSON 已保存至: {json_path}")
    else:
        print(f"  预览图已保存至: {output_dir}")

if __name__ == '__main__':
    main()
