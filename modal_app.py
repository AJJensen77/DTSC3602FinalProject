"""Deploy the D2 Streamlit dashboard on Modal.

Usage:
    uv run modal deploy modal_app.py

Reads DATABASE_URL from the local .env file at deploy time (see .env.example),
so no named Modal secret is required. .env is gitignored, so the credential is
never committed - but it is uploaded to Modal when you deploy.
"""

import shlex
import subprocess
from pathlib import Path

import modal

LOCAL_DIR = Path(__file__).parent

app = modal.App("dtsc-3602-final-project-d2")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        # Pinned to the versions in uv.lock so the deployed app matches local.
        "streamlit==1.66.1",
        "pandas==3.1.0",
        "numpy==2.5.4",
        "matplotlib==3.11.2",
        "pyarrow==26.0.0",
        "sqlalchemy==2.1.6",
        "psycopg[binary]==3.3.6",
        "python-dotenv==1.2.4",
    )
    .add_local_file(LOCAL_DIR / "app.py", "/root/app.py")
    .add_local_file(LOCAL_DIR / "common.py", "/root/common.py")
    # The dashboard imports the shared palette helpers from the package.
    .add_local_dir(
        LOCAL_DIR / "src" / "dtsc_3602_final_project",
        "/root/dtsc_3602_final_project",
    )
)


# Credentials are read from the local .env at deploy time rather than from a
# named secret, so this project does not share (or clobber) the secret the
# earlier homework apps use. .env is gitignored and never committed.
@app.function(image=image, secrets=[modal.Secret.from_dotenv(LOCAL_DIR)])
@modal.concurrent(max_inputs=100)
@modal.web_server(8000, startup_timeout=120)
def run():
    target = shlex.quote("/root/app.py")
    cmd = (
        f"streamlit run {target} --server.port 8000 --server.address 0.0.0.0 "
        "--server.headless true --server.enableCORS=false "
        "--server.enableXsrfProtection=false"
    )
    subprocess.Popen(cmd, shell=True)
