#!/usr/bin/env python
"""
生成蒸馏软目标 targets.json

读取 outputs/results.json，对每张图片的前景/背景主色进行交互校正，
直接选取人类认可的颜色作为 target（不做加权平均）。

支持断点续传：Ctrl+C 中断后，重新运行会从上次中断处继续。
"""
import json
import math
import os
import sys
import tempfile

from PIL import Image, ImageDraw


BREAKPOINT_FILE = 'targets_progress.json'


def lab_to_rgb(L, a, b):
    """Lab 转 RGB (简化版，用于预览)"""
    xn, yn, zn = 95.047, 100.0, 108.883
    
    fy = (L + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b / 200.0
    
    delta = 6.0 / 29.0
    
    def f_inv(t):
        if t > delta:
            return t ** 3
        else:
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
        else:
            return 12.92 * c
    
    r = max(0, min(255, int(round(gamma(rl) * 255))))
    g = max(0, min(255, int(round(gamma(gl) * 255))))
    b_val = max(0, min(255, int(round(gamma(bl) * 255))))
    
    return (r, g, b_val)


def create_preview(img_path, colors):
    """
    创建预览图：原始图片 + N个颜色色块（按分数排序）
    返回 PIL Image 对象
    """
    img = Image.open(img_path).convert('RGB')
    
    max_size = 400
    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
    
    color_bar_height = 80
    canvas_width = img.width
    canvas_height = img.height + color_bar_height * len(colors) + 20
    canvas = Image.new('RGB', (canvas_width, canvas_height), (255, 255, 255))
    canvas.paste(img, (0, 0))
    
    draw = ImageDraw.Draw(canvas)
    
    for i, (L, a, b, score) in enumerate(colors):
        r, g, b_val = lab_to_rgb(L, a, b)
        y_pos = img.height + 10 + i * (color_bar_height + 10)
        
        bar_width = int(img.width * 0.8)
        draw.rectangle([10, y_pos, 10 + bar_width, y_pos + color_bar_height], fill=(r, g, b_val))
        draw.rectangle([10, y_pos, 10 + bar_width, y_pos + color_bar_height], outline=(0, 0, 0), width=2)
        
        draw.text((20, y_pos + 5), f"#{i+1}", fill=(0, 0, 0) if (r+g+b_val) > 382 else (255, 255, 255))
        
        text = f" L={L:.1f} a={a:.1f} b={b:.1f}  score={score:.4f}"
        draw.text((10 + bar_width + 10, y_pos + 10), text, fill=(0, 0, 0))
    
    return canvas


def show_preview_and_wait(canvas, img_filename, region_type):
    """弹窗显示预览图，等待用户关闭后继续"""
    tmp_dir = tempfile.gettempdir()
    region_label = "前景" if region_type == "fg" else "背景"
    tmp_path = os.path.join(tmp_dir, f"{img_filename}_{region_type}_preview.png")
    canvas.save(tmp_path)
    
    canvas.show()
    print(f"  [正在处理{region_label}] 预览图已弹窗显示，请查看...")


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


def save_checkpoint(targets, last_img, last_region):
    """保存断点进度"""
    checkpoint = {
        'targets': targets,
        'last_img': last_img,
        'last_region': last_region,
    }
    with open(BREAKPOINT_FILE, 'w', encoding='utf-8') as f:
        json.dump(checkpoint, f, indent=2, ensure_ascii=False)


def load_checkpoint():
    """加载断点进度"""
    if os.path.exists(BREAKPOINT_FILE):
        with open(BREAKPOINT_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def process_colors(colors, img_filename, region_type, img_dir):
    """
    处理前景或背景颜色
    colors: list of dict with 'lab' and 'score'
    返回 (L, a, b, confidence) 或 (None, None, None, None)（用户选择跳过此图）
    confidence = score1 / (score1 + score2)，衡量 teacher 对 top 候选的区分度
    """
    n_colors = min(len(colors), 5)
    if n_colors == 0:
        return 50.0, 0.0, 0.0, 1.0  # 默认中性值，置信度 1.0

    sorted_colors = sorted(colors, key=lambda c: c['score'], reverse=True)[:n_colors]

    # 计算 teacher 置信度：score1/(2*score2)，clip 到 1.0
    if n_colors >= 2:
        s1, s2 = sorted_colors[0]['score'], sorted_colors[1]['score']
        confidence = min(s1 / (2 * s2), 1.0) if s2 > 0 else 1.0
    else:
        confidence = 1.0

    print(f"\n  处理 {img_filename} - {region_type} (conf={confidence:.3f})")

    # 打印候选颜色
    for i in range(n_colors):
        c = sorted_colors[i]
        label = f"  第{i+1}名: Lab={c['lab']} (score={c['score']:.4f})"
        print(label)

    if n_colors >= 2:
        score1 = sorted_colors[0]['score']
        score2 = sorted_colors[1]['score']
        gap12 = (score1 - score2) / score1 if score1 > 0 else 0

        # 检查 score1 与 score3 的差距
        gap13 = 0
        if n_colors >= 3:
            score3 = sorted_colors[2]['score']
            gap13 = (score1 - score3) / score1 if score1 > 0 else 0

        print(f"  gap12 = {gap12:.4f}  gap13 = {gap13:.4f}")

        need_human = gap12 < 0.05 or (n_colors >= 3 and gap13 < 0.20)
        if need_human:
            reason = "gap12<0.05" if gap12 < 0.05 else "gap13<0.20"
            # 需要人工选择
            preview_colors = []
            for c in sorted_colors:
                lab = c['lab']
                preview_colors.append((lab[0], lab[1], lab[2], c['score']))

            img_path = os.path.join(img_dir, img_filename)
            if not os.path.exists(img_path):
                print(f"  警告：找不到图片 {img_path}，默认选择第1名")
                selected_lab = sorted_colors[0]['lab']
                return selected_lab[0], selected_lab[1], selected_lab[2], confidence
            else:
                canvas = create_preview(img_path, preview_colors)

                print(f"  {reason}，需要人工确认主色")
                label_str = '  '.join([f"{i+1}: Lab={sorted_colors[i]['lab']}" for i in range(n_colors)])
                print(f"  候选颜色：{label_str}")

                show_preview_and_wait(canvas, img_filename, region_type)

                choice = select_color_index(n_colors)
                if choice == 0:
                    print(f"  用户选择跳过此图")
                    return None, None, None, None

                selected_lab = sorted_colors[choice - 1]['lab']
                print(f"  用户选择第{choice}名: Lab={selected_lab}")
                return selected_lab[0], selected_lab[1], selected_lab[2], 1.0
    else:
        print(f"  仅 {n_colors} 个颜色，自动选择第1名")

    # gap12 >= 0.05 且 gap13 >= 0.20 或仅1个颜色，自动选择第1名
    selected_lab = sorted_colors[0]['lab']
    print(f"  自动选择第1名: Lab={selected_lab}")
    return selected_lab[0], selected_lab[1], selected_lab[2], confidence


def migrate_confidence(targets_path='targets.json'):
    """
    迁移旧版 targets.json 中的置信度：
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
                # 旧公式 → 新公式转换
                new_conf = old_conf / (2 * (1 - old_conf))
                entry[key] = round(min(new_conf, 1.0), 4)
                fixed_convert += 1

    with open(targets_path, 'w', encoding='utf-8') as f:
        json.dump(targets, f, indent=2, ensure_ascii=False)

    print(f"已迁移 {len(targets)} 条数据: {fixed_human} 条人类标注→1.0, {fixed_convert} 条公式转换")


def main():
    results_path = 'outputs/results.json'
    targets_path = 'targets.json'
    img_dir = 'extracted_imgs/imgs'
    
    if not os.path.exists(results_path):
        print(f"错误: {results_path} 不存在，请先运行 python main.py 生成结果")
        sys.exit(1)
    
    with open(results_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    # 加载断点
    checkpoint = load_checkpoint()
    if checkpoint:
        targets = checkpoint['targets']
        last_img = checkpoint['last_img']
        last_region = checkpoint['last_region']
        
        # 跳过已处理的图片
        skip = True
        start_idx = 0
        for i, result in enumerate(results):
            img_filename = os.path.basename(result['image'])
            if skip and img_filename == last_img:
                start_idx = i
                if last_region == 'fg':
                    print(f"\n>>> 从断点恢复：跳过 {img_filename} 的前景，从其背景开始")
                    # 重新处理该图片的背景
                    bg_colors = result['background']['main_colors']
                    L_bg, a_bg, b_bg, bg_conf = process_colors(bg_colors, img_filename, 'bg', img_dir)
                    if L_bg is not None:
                        targets[img_filename] = {
                            'L_fg': round(targets[img_filename].get('L_fg', 50.0), 1),
                            'a_fg': round(targets[img_filename].get('a_fg', 0.0), 1),
                            'b_fg': round(targets[img_filename].get('b_fg', 0.0), 1),
                            'fg_conf': targets[img_filename].get('fg_conf', 1.0),
                            'L_bg': round(L_bg, 1),
                            'a_bg': round(a_bg, 1),
                            'b_bg': round(b_bg, 1),
                            'bg_conf': round(bg_conf, 4),
                        }
                        save_checkpoint(targets, img_filename, 'bg')
                    else:
                        if img_filename in targets:
                            del targets[img_filename]
                        save_checkpoint(targets, '', 'skip')
                skip = False
                continue
        results = results[start_idx:] if not skip else []
        if not results and skip:
            # 全部处理完了
            with open(targets_path, 'w', encoding='utf-8') as f:
                json.dump(targets, f, indent=2, ensure_ascii=False)
            print(f"\ntargets.json 已保存（从断点恢复），共 {len(targets)} 张图片")
            if os.path.exists(BREAKPOINT_FILE):
                os.remove(BREAKPOINT_FILE)
            return
    else:
        targets = {}
    
    for result in results:
        img_path = result['image']
        img_filename = os.path.basename(img_path)
        
        fg_colors = result['foreground']['main_colors']
        L_fg, a_fg, b_fg, fg_conf = process_colors(fg_colors, img_filename, 'fg', img_dir)
        if L_fg is None:
            print(f"  跳过 {img_filename}")
            continue

        targets.setdefault(img_filename, {})
        targets[img_filename]['L_fg'] = round(L_fg, 1)
        targets[img_filename]['a_fg'] = round(a_fg, 1)
        targets[img_filename]['b_fg'] = round(b_fg, 1)
        targets[img_filename]['fg_conf'] = round(fg_conf, 4)
        save_checkpoint(targets, img_filename, 'fg')

        bg_colors = result['background']['main_colors']
        L_bg, a_bg, b_bg, bg_conf = process_colors(bg_colors, img_filename, 'bg', img_dir)
        if L_bg is None:
            print(f"  跳过 {img_filename}（背景阶段）")
            # 移除之前添加的前景数据
            if img_filename in targets:
                del targets[img_filename]
            save_checkpoint(targets, img_filename, 'bg')
            continue

        targets[img_filename]['L_bg'] = round(L_bg, 1)
        targets[img_filename]['a_bg'] = round(a_bg, 1)
        targets[img_filename]['b_bg'] = round(b_bg, 1)
        targets[img_filename]['bg_conf'] = round(bg_conf, 4)
        save_checkpoint(targets, img_filename, 'bg')
    
    with open(targets_path, 'w', encoding='utf-8') as f:
        json.dump(targets, f, indent=2, ensure_ascii=False)
    
    if os.path.exists(BREAKPOINT_FILE):
        os.remove(BREAKPOINT_FILE)
    print(f"\ntargets.json 已保存，共 {len(targets)} 张图片")


if __name__ == '__main__':
    import sys
    if '--migrate' in sys.argv:
        idx = sys.argv.index('--migrate')
        path = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else 'targets.json'
        migrate_confidence(path)
    else:
        main()
