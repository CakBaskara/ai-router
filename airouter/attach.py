import os
import re
import shutil
import time
from pathlib import Path

from . import config

MEDIA = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif", ".webp": "image/webp"}
CONVERT = {".bmp", ".tif", ".tiff"}
MAX_BYTES = 20 * 1024 * 1024


def folder() -> Path:
    path = Path(os.environ.get("AI_ROUTER_ATTACH", config.REPO_ROOT / "logs" / "attachments"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def is_image(path: Path) -> bool:
    return path.suffix.lower() in MEDIA


def media_type(path: Path) -> str:
    return MEDIA[path.suffix.lower()]


def _target(name: str) -> Path:
    safe = re.sub(r"[^\w.-]+", "_", name).strip("_") or "file"
    return folder() / f"{time.strftime('%Y%m%d-%H%M%S')}-{safe}"


def store(src: Path) -> Path:
    src = Path(src)
    if src.stat().st_size > MAX_BYTES:
        raise ValueError(f"{src.name} lebih dari 20 MB")
    if src.suffix.lower() in CONVERT:
        from PIL import Image
        dest = _target(src.stem + ".png")
        Image.open(src).save(dest)
        return dest
    dest = _target(src.name)
    shutil.copy2(src, dest)
    return dest


def from_clipboard() -> list[Path]:
    from PIL import ImageGrab
    data = ImageGrab.grabclipboard()
    if data is None:
        return []
    if isinstance(data, list):
        return [store(Path(p)) for p in data if Path(p).is_file()]
    dest = _target("clipboard.png")
    data.save(dest)
    return [dest]


def pick_files() -> list[Path]:
    import tkinter
    from tkinter import filedialog
    root = tkinter.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        names = filedialog.askopenfilenames(parent=root, title="Lampirkan file", initialdir=os.getcwd())
    finally:
        root.destroy()
    return [Path(n) for n in names]


def describe(files: list[Path]) -> str:
    return "\n".join(f"📎 {f.name.split('-', 2)[-1]}" for f in files)
