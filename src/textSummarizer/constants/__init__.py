from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent

CONFIG_FILE_PATH = ROOT_DIR / "config" / "config.yaml"

PARAMS_FILE_PATH = ROOT_DIR / "params.yaml"