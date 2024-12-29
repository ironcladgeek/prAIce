from pathlib import Path

PATHS = {
    "root": Path(__file__).absolute().parent.parent,
    "app": Path(__file__).absolute().parent,
    "logs": Path(__file__).absolute().parent.parent / "logs",
}
