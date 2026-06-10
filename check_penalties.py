"""检查亮度/饱和度惩罚效果"""
import json
from pathlib import Path

with open('outputs/results.json', 'r') as f:
    results = json.load(f)

count = 0
for r in results:
    for region in ['foreground', 'background']:
        dom = r[region]['dominant_color']
        rgb = dom['rgb']
        is_near_white = all(v > 225 for v in rgb)
        is_near_black = all(v < 30 for v in rgb)
        if is_near_white or is_near_black:
            fname = Path(r['image']).name
            print(f"{fname:30s} | {region:10s} | {dom['hex']:8s} | score={dom['score']:.3f}")
            count += 1

print(f'共 {count} 个纯黑/白 dominant_color')

# 统计所有输出的主色中接近黑白的数量
total_bw = 0
for r in results:
    for region in ['foreground', 'background']:
        for c in r[region]['main_colors']:
            rgb = c['rgb']
            if all(v > 225 for v in rgb) or all(v < 30 for v in rgb):
                total_bw += 1

print(f'所有主色中共 {total_bw} 个接近黑白')
