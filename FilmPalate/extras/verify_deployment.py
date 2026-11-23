"""Check that the deployed API returns the same recommendations as the local pipeline.

``build_model.py`` saves reference outputs of the fitted pipeline for three requests
in ``sample_predictions.json`` (next to this script). This sends the same requests to the
deployed API (``POST /recommend``) and compares film ids and order, then checks the
health, pipeline-info and error responses.

Usage (repo root):
    uv run python "FilmPalate/extras/verify_deployment.py" [API_URL]
"""

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_URL = "https://yagerr05--filmpalate-api-fastapi-app.us-east.modal.run"


def call(url, body=None):
    req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"null")


def main():
    base = (sys.argv[1] if len(sys.argv) > 1 else DEFAULT_URL).rstrip("/")
    samples = json.loads((HERE / "sample_predictions.json").read_text())
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= cond
        print(f"[{'PASS' if cond else 'FAIL'}] {name}{'  ' + detail if detail else ''}")

    status, health = call(f"{base}/health")
    check("GET /health", status == 200 and health.get("status") == "ok", str(health))
    status, info = call(f"{base}/pipeline")
    check("GET /pipeline", status == 200 and info.get("steps") == ["encode", "rank"])

    for name, s in samples.items():
        req = s["request"]
        body = {"ratings": [{"movie_id": int(m), "stars": v} for m, v in req["ratings"].items()],
                "want_to_watch": req["want_to_watch"], "k": len(s["top"])}
        status, r = call(f"{base}/recommend", body)
        top = [m["movie_id"] for m in r["recommendations"]] if status == 200 else None
        new = [m["movie_id"] for m in r["new_releases"]] if status == 200 else None
        check(f"{name}: top picks match local", top == s["top"])
        check(f"{name}: new releases match local", new == s["new"])

    status, _ = call(f"{base}/recommend", {"ratings": [{"movie_id": 1, "stars": 4.3}]})
    check("invalid stars -> 422", status == 422)
    status, _ = call(f"{base}/recommend", {"ratings": [{"movie_id": 123456789, "stars": 4}]})
    check("unknown movie -> 422", status == 422)
    status, _ = call(f"{base}/movies/999999999")
    check("unknown movie page -> 404", status == 404)

    print("\nAll checks passed." if ok else "\nSome checks FAILED.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
