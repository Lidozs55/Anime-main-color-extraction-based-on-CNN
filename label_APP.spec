# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for label.py (前后端一体化标注工具)
# 输出: dist/label_APP/label_APP.exe （单 exe）
# 用法: pyinstaller label_APP.spec --noconfirm

import os
import site

block_cipher = None

# 解析项目根目录（PyInstaller 提供的全局变量 SPECPATH）
try:
    PROJECT_ROOT = SPECPATH
except NameError:
    PROJECT_ROOT = os.path.dirname(os.path.abspath(SPEC))


def find_site_packages():
    """尝试定位 site-packages；优先使用 build_env 虚拟环境，否则使用当前 Python。"""
    build_env = os.path.join(PROJECT_ROOT, "build_env", "Lib", "site-packages")
    if os.path.isdir(build_env):
        return [build_env]
    # 否则使用当前 Python 的 site-packages
    try:
        sps = site.getsitepackages()
    except Exception:
        sps = []
    return [sp for sp in sps if os.path.isdir(sp)]


# 收集 hiddenimports（label.py 动态导入的子模块）
hiddenimports = [
    "graphcolor.pipeline",
    "graphcolor.preprocess",
    "graphcolor.segment",
    "graphcolor.cluster",
    "graphcolor.scoring",
    "graphcolor.visualize",
    "graphcolor.html_visualize",
    "flask",
    "PIL",
    "PIL.Image",
    "urllib.request",
    "webbrowser",
    "zipfile",
]

# 收集 datas：templates 必须包含
datas = [
    (os.path.join(PROJECT_ROOT, "templates"), "templates"),
]

# 额外 python 路径
extra_paths = find_site_packages()

a = Analysis(
    ["label.py"],
    pathex=[PROJECT_ROOT] + extra_paths,
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "torch",
        "torchvision",
        "torchaudio",
        "matplotlib",
        "scipy",
        "pandas",
        "IPython",
        "notebook",
        "jupyter",
        "pytest",
        "tensorboard",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="label_APP",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,           # 打包后双击启动，浏览器自动打开
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="label_APP",
)
