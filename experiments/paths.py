from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "results" / "data"
FIGURES = ROOT / "results" / "figures"
PRESENTATION = ROOT / "presentation_images"

for folder in (DATA, FIGURES, PRESENTATION):
    folder.mkdir(parents=True, exist_ok=True)
