"""Modal deployment of the FilmPalate API.

Usage (from the repo root, after build_model.py has written api/artifacts/):
    uv run modal serve  "FilmPalate/api/modal_serve.py"   # temporary dev URL
    uv run modal deploy "FilmPalate/api/modal_serve.py"   # public URL
"""

from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
APP_DIR = HERE.parent
ROOT = APP_DIR.parent

app = modal.App("filmpalate-api")

# Pinned to uv.lock: the joblib artifact must be unpickled by the exact
# scikit-learn / numpy / pandas versions that wrote it.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "scikit-learn==1.9.1",
        "numpy==2.5.4",
        "pandas==3.1.0",
        "scipy==1.18.1",
        "joblib==1.6.0",
        "pyarrow==26.0.0",
        "fastapi==0.162.5",
        "pydantic==2.14.3",
    )
    # Classes the pickled pipeline refers to.
    .add_local_dir(APP_DIR / "watchlist", "/root/watchlist")
    .add_local_dir(ROOT / "List taste model" / "listmodel", "/root/listmodel")
    .add_local_file(ROOT / "Gradient boost model" / "gbmodel" / "__init__.py", "/root/gbmodel/__init__.py")
    .add_local_file(ROOT / "Gradient boost model" / "gbmodel" / "recommenders.py",
                    "/root/gbmodel/recommenders.py")
    .add_local_file(HERE / "serve.py", "/root/api/serve.py")
    .add_local_dir(HERE / "artifacts", "/root/api/artifacts")
)


@app.function(image=image, min_containers=0, memory=3072, scaledown_window=600)
@modal.concurrent(max_inputs=20)
@modal.asgi_app()
def fastapi_app():
    import sys

    sys.path.insert(0, "/root")
    from api.serve import app as web_app

    return web_app
