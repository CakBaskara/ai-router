import shutil
from pathlib import Path


def shim_dir(name: str) -> Path | None:
    found = shutil.which(name)
    return Path(found).parent if found else None


def node_script(name: str, *relative: str) -> list[str]:
    directory = shim_dir(name)
    if directory:
        script = directory.joinpath("node_modules", *relative)
        node = directory / "node.exe"
        if script.exists():
            return [str(node) if node.exists() else (shutil.which("node") or "node"), str(script)]
    return [shutil.which(name) or name]
