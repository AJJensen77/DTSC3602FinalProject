"""Deploy the list-model demo (Streamlit) on Modal.

Usage (from the repo root, after evaluate_list_model.py has saved the artifact):
    uv run modal deploy "List taste model/demo_app/modal_app.py"

No secrets or database: the fitted model is baked into the image and loaded once
per container.
"""

import shlex
import subprocess
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE.parent
SHARED = MODEL_DIR.parent / "Gradient boost model" / "gbmodel"

app = modal.App("ishika-list-model-demo")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        # Pinned to uv.lock so the unpickled model matches the one fitted locally.
        "streamlit==1.66.1",
        "pandas==3.1.0",
        "numpy==2.5.4",
        "scipy==1.18.1",
        "scikit-learn==1.9.1",
        "joblib==1.6.0",
    )
    .env({"LIST_MODEL_ARTIFACT": "/root/list_model.joblib"})
    .add_local_file(HERE / "app.py", "/root/app.py")
    .add_local_dir(MODEL_DIR / "listmodel", "/root/listmodel")
    # The model class extends the group's shared recommender contract.
    .add_local_file(SHARED / "__init__.py", "/root/gbmodel/__init__.py")
    .add_local_file(SHARED / "recommenders.py", "/root/gbmodel/recommenders.py")
    .add_local_file(MODEL_DIR / "artifacts" / "list_model.joblib", "/root/list_model.joblib")
)


@app.function(image=image, memory=1024, scaledown_window=300)
@modal.concurrent(max_inputs=50)
@modal.web_server(8000, startup_timeout=180)
def run():
    cmd = (
        f"streamlit run {shlex.quote('/root/app.py')} --server.port 8000 "
        "--server.address 0.0.0.0 --server.headless true "
        "--server.enableCORS=false --server.enableXsrfProtection=false"
    )
    subprocess.Popen(cmd, shell=True)
