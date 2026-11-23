"""Modal deployment of the combined recommender API.

Usage (from the repo root, after build_site.py):
    uv run modal serve "Combined website/api/modal_serve.py"    # dev server
    uv run modal deploy "Combined website/api/modal_serve.py"   # public URL
"""

from pathlib import Path

import modal

HERE = Path(__file__).parent
SITE = HERE.parent
ROOT = SITE.parent

app = modal.App("reelpicks-combined-api")

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
    # The pickled models refer to classes in these packages.
    .add_local_dir(SITE / "combined", "/root/combined")
    .add_local_dir(ROOT / "List taste model" / "listmodel", "/root/listmodel")
    .add_local_dir(ROOT / "Gradient boost model" / "gbmodel", "/root/gbmodel")
    .add_local_file(HERE / "serve.py", "/root/api/serve.py")
    .add_local_dir(HERE / "artifacts", "/root/api/artifacts")
)


@app.function(image=image, min_containers=0, memory=4096, scaledown_window=600)
@modal.concurrent(max_inputs=20)
@modal.asgi_app()
def fastapi_app():
    import sys

    sys.path.insert(0, "/root")
    from api.serve import app as web_app

    return web_app
