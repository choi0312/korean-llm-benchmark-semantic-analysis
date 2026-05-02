import random
import shutil
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


def ensure_dirs(output_dir):
    output_dir = Path(output_dir)
    paths = {
        "root": output_dir,
        "figures": output_dir / "figures",
        "tables": output_dir / "tables",
        "models": output_dir / "models",
        "cache": output_dir / "cache",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


def set_korean_font():
    candidates = [
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf",
    ]
    for fp in candidates:
        if Path(fp).exists():
            fm.fontManager.addfont(fp)
            plt.rcParams["font.family"] = Path(fp).stem
            break
    plt.rcParams["axes.unicode_minus"] = False


def zip_outputs(output_dir):
    output_dir = Path(output_dir)
    return shutil.make_archive(str(output_dir), "zip", output_dir)
