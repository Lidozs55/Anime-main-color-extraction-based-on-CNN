# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for generate_targets_pixiv.py
打包 Pixiv 蒸馏目标生成器为单个 exe 文件

构建命令: pyinstaller generate_targets_pixiv.spec
"""

import os
import sys

block_cipher = None

# 项目根目录
project_root = os.path.dirname(os.path.abspath(SPEC))

# 额外的 site-packages 路径（确保找到所有已安装包）
# 优先使用 conda 环境的 site-packages
extra_paths = [
    r'C:\ProgramData\miniconda3\Lib\site-packages',
]
# 也从当前 Python 环境收集（但排除 Roaming 避免版本冲突）
for p in sys.path:
    if 'site-packages' in p and os.path.isdir(p) and 'Roaming' not in p and p not in extra_paths:
        extra_paths.append(p)

a = Analysis(
    ['generate_targets_pixiv.py'],
    pathex=[
        project_root,  # 项目根目录（graphcolor 包所在）
        *extra_paths,  # 所有 site-packages
    ],
    binaries=[],
    datas=[],
    hiddenimports=[
        # graphcolor 子模块
        'graphcolor',
        'graphcolor.pipeline',
        'graphcolor.preprocess',
        'graphcolor.segment',
        'graphcolor.cluster',
        'graphcolor.scoring',
        'graphcolor.visualize',
        'graphcolor.html_visualize',
        # 第三方依赖
        'cv2',
        'numpy',
        'PIL',
        'PIL.Image',
        'PIL.ImageDraw',
        'PIL.ImageShow',
        'PIL.ImageFilter',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'matplotlib',
        'pandas',
        'jupyter',
        'IPython',
        'notebook',
        'torch',
        'torchvision',
    ],
    noarchive=False,
    optimize=0,
)

from PyInstaller.utils.hooks import collect_submodules, collect_data_files, collect_all

# 收集 graphcolor 整个包
graphcolor_hidden = collect_submodules('graphcolor')
a.hiddenimports.extend(graphcolor_hidden)

# 用 collect_all 收集完整的第三方包（含数据文件、二进制、子模块）
for pkg in ['sklearn', 'scipy', 'rembg', 'pymatting', 'onnxruntime', 'pooch']:
    try:
        result = collect_all(pkg)
        if len(result) == 3:
            datas, binaries, hiddenimports = result
        else:
            datas, binaries = result
            hiddenimports = []
        # 确保所有条目都是 (name, path, typecode) 三元组
        for item in datas:
            if len(item) == 3:
                a.datas.append(item)
        for item in binaries:
            if len(item) == 3:
                a.binaries.append(item)
        a.hiddenimports.extend(hiddenimports)
        print(f"[spec] Collected {pkg}: {len(datas)} data, {len(binaries)} binaries, {len(hiddenimports)} hidden")
    except Exception as e:
        print(f"[spec] Warning: failed to collect {pkg}: {e}")

# 显式收集 numpy.libs DLL（numpy 2.x 的 openblas 等依赖）
import glob as _glob
numpy_libs_dir = None
for sp in extra_paths:
    candidate = os.path.join(sp, 'numpy.libs')
    if os.path.isdir(candidate):
        numpy_libs_dir = candidate
        break
if numpy_libs_dir:
    for dll in _glob.glob(os.path.join(numpy_libs_dir, '*.dll')):
        a.binaries.append((os.path.basename(dll), dll, 'BINARY'))
    print(f"[spec] Collected numpy.libs DLLs from {numpy_libs_dir}")

# 显式收集 scipy.libs DLL
scipy_libs_dir = None
for sp in extra_paths:
    candidate = os.path.join(sp, 'scipy.libs')
    if os.path.isdir(candidate):
        scipy_libs_dir = candidate
        break
if scipy_libs_dir:
    for dll in _glob.glob(os.path.join(scipy_libs_dir, '*.dll')):
        a.binaries.append((os.path.basename(dll), dll, 'BINARY'))
    print(f"[spec] Collected scipy.libs DLLs from {scipy_libs_dir}")

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='Pixiv标注工具',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # 需要控制台交互（用户输入选择颜色）
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)
