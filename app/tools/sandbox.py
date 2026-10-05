"""Run LLM-generated Python in an isolated container (no network, read-only FS, limits)."""
import os, subprocess, sys, uuid
from dataclasses import dataclass
from pathlib import Path
from app.config import settings

MAX_OUT = 8000


@dataclass
class SandboxResult:
    ok: bool
    stdout: str
    stderr: str


def run_dir(run_id: str) -> Path:
    p = Path(settings.runs_dir).resolve() / run_id
    for sub in ("data", "work", "out"):
        (p / sub).mkdir(parents=True, exist_ok=True)
    os.chmod(p / "out", 0o777)               # sandbox user (65534) must write charts
    return p


def has_dataset(run_id: str) -> bool:
    return (run_dir(run_id) / "data" / "input.csv").exists()


def list_outputs(run_id: str) -> list[str]:
    return sorted(f.name for f in (run_dir(run_id) / "out").iterdir() if f.is_file())


def run_python(code: str, run_id: str, name: str = "code") -> SandboxResult:
    d = run_dir(run_id)
    (d / "work" / f"{name}.py").write_text(code)
    timeout = settings.sandbox_timeout
    if settings.sandbox_mode == "local":     # UNSAFE: dev/tests only
        env = {**os.environ, "DATA_DIR": str(d / "data"), "OUT_DIR": str(d / "out")}
        try:
            p = subprocess.run([sys.executable, str(d / "work" / f"{name}.py")], capture_output=True,
                               text=True, timeout=timeout, env=env, cwd=d / "work")
        except subprocess.TimeoutExpired:
            return SandboxResult(False, "", f"Timeout after {timeout}s")
        return SandboxResult(p.returncode == 0, p.stdout[-MAX_OUT:], p.stderr[-MAX_OUT:])

    host = Path(settings.host_runs_dir or settings.runs_dir).resolve() / run_id
    cname = f"agentos-sbx-{uuid.uuid4().hex[:10]}"
    cmd = ["docker", "run", "--rm", "--name", cname, "--network", "none", "--read-only",
           "--memory", "512m", "--cpus", "1", "--pids-limit", "128", "--user", "65534:65534",
           "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
           "--tmpfs", "/tmp:size=64m",
           "-v", f"{host}/data:/data:ro", "-v", f"{host}/work:/work:ro", "-v", f"{host}/out:/out:rw",
           settings.sandbox_image, "python", f"/work/{name}.py"]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 10)
    except subprocess.TimeoutExpired:
        subprocess.run(["docker", "kill", cname], capture_output=True)
        return SandboxResult(False, "", f"Timeout after {timeout}s")
    return SandboxResult(p.returncode == 0, p.stdout[-MAX_OUT:], p.stderr[-MAX_OUT:])
