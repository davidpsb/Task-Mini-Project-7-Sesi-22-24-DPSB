"""set_env.py — environment check & auto-install helper.

Dipakai dari notebook:
    from src.set_env import check_and_setup_environment
    result = check_and_setup_environment(project_root=PROJECT_ROOT)

Atau dari CLI:
    python src/set_env.py
    python src/set_env.py --no-install
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import InvalidVersion, Version


DEFAULT_NOTEBOOK_ONLY = {"ipykernel": "6.0"}
DEFAULT_TIMEOUT = 300


# ─────────────────────────────────────────────────────────
# Internal
# ─────────────────────────────────────────────────────────
def _find_project_root(start: Path | None = None) -> Path:
    """Naik dari file ini (atau `start`) sampai ketemu requirements.txt."""
    p = (start or Path(__file__)).resolve()
    if p.is_file():
        p = p.parent
    while p != p.parent:
        if (p / "requirements.txt").is_file():
            return p
        p = p.parent
    raise FileNotFoundError(
        f"requirements.txt tidak ditemukan dari {start or __file__}"
    )


def _parse_requirements(path: Path) -> dict[str, str]:
    """requirements.txt → {package: min_version}.

    Handle: `pkg>=1.2`, `pkg==1.2.3`, `pkg~=1.2`, `pkg` (tanpa versi),
    `pkg>=1.2,<2` (ambil batas bawah). Baris `-r`/`-c`/`--opt` dilewati.
    """
    out: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        try:
            req = Requirement(line)
        except Exception:
            # fallback regex untuk format tidak standar
            m = re.match(
                r"^([A-Za-z0-9_.\-]+)\s*(?:>=|==|~=)?\s*([0-9][A-Za-z0-9.\-]*)?",
                line,
            )
            if m:
                out[m.group(1).lower()] = m.group(2) or "0"
            continue
        min_ver = "0"
        for spec in req.specifier:
            if spec.operator in (">=", "==", "~="):
                min_ver = spec.version
                break
        out[req.name.lower()] = min_ver
    return out


def _check_package(name: str, min_ver: str) -> tuple[str, str]:
    """Cek package → return (installed_version, status).

    status:
      "ok"       — versi terpasang ≥ min_ver
      "outdated" — terpasang tapi di bawah min_ver
      "missing"  — belum terinstall
    """
    try:
        installed = version(name)
    except PackageNotFoundError:
        return "-", "missing"
    try:
        status = "ok" if Version(installed) >= Version(min_ver) else "outdated"
    except InvalidVersion:
        status = "outdated"
    return installed, status


def _print_table(rows: list[tuple], title: str) -> None:
    c1 = max(len(r[0]) for r in rows) + 2
    c2, c3 = 10, 14
    print(f"\n── {title} ──")
    print(f"{'Package':<{c1}} {'Min':<{c2}} {'Terpasang':<{c3}} Status")
    print("-" * (c1 + c2 + c3 + 10))
    icon = {"ok": "✅", "outdated": "⚠️ ", "missing": "❌"}
    for pkg, mv, iv, st in rows:
        print(f"{pkg:<{c1}} {mv:<{c2}} {iv:<{c3}} {icon[st]} {st}")


def _install(packages: list[str], timeout: int) -> bool:
    if not packages:
        return True
    print(f"\n📦 Install {len(packages)} package: {', '.join(packages)}")
    env = {**os.environ, "UV_HTTP_TIMEOUT": str(timeout)}
    try:
        subprocess.run(["uv", "--version"], check=True, capture_output=True)
        cmd = ["uv", "pip", "install", *packages]
    except (FileNotFoundError, subprocess.CalledProcessError):
        cmd = [sys.executable, "-m", "pip", "install", "--quiet", *packages]
    print(f"   $ {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if r.returncode != 0:
        print("❌ Install gagal.")
        if r.stderr:
            print("STDERR:", r.stderr[-500:])
        return False
    print("✅ Install selesai.")
    return True


# ─────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────
def check_and_setup_environment(
    project_root: Path | None = None,
    auto_install: bool = True,
    notebook_only: dict[str, str] | None = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> dict:
    """Cek semua package dari requirements.txt (+ notebook_only),
    install otomatis yang kurang/outdated bila auto_install=True.

    Return dict:
      ok            : bool        — semua memenuhi syarat di akhir
      need_restart  : bool        — True bila ada yang baru diinstall → restart kernel
      reqfile       : Path        — lokasi requirements.txt yang dipakai
      missing       : list[str]   — package yang masih missing (kosong bila ok)
      outdated      : list[str]   — package yang masih outdated (kosong bila ok)
    """
    if project_root is None:
        project_root = _find_project_root()
    project_root = Path(project_root)

    print(f"Python  : {sys.version.split()[0]}")
    print(f"Interp. : {sys.executable}")
    print(f"Root    : {project_root}")

    reqfile = project_root / "requirements.txt"
    if not reqfile.is_file():
        raise FileNotFoundError(f"requirements.txt tidak ditemukan di {project_root}")

    parsed = _parse_requirements(reqfile)
    notebook_only = notebook_only if notebook_only is not None else DEFAULT_NOTEBOOK_ONLY
    for k, v in notebook_only.items():
        parsed.setdefault(k, v)

    print(f"reqfile : {reqfile}")
    print(f"Total   : {len(parsed)} package ({len(notebook_only)} notebook-only)")

    # rows: (package, min_ver, installed_ver, status)
    rows = [(p, mv, *_check_package(p, mv)) for p, mv in parsed.items()]
    _print_table(rows, "Cek Awal")

    kurang = [(p, v, s) for p, v, _, s in rows if s in ("missing", "outdated")]
    need_restart = False

    if not kurang:
        print("\n🎉 Semua package lengkap. Tidak ada yang perlu diinstall.")
    elif not auto_install:
        print(f"\n⚠️  {len(kurang)} package kurang/outdated (auto_install=False).")
        print(
            "Install manual:  uv pip install "
            + " ".join(f"'{p}>={v}'" for p, v, _ in kurang)
        )
    else:
        print(f"\n⚠️  {len(kurang)} package perlu diinstall.")
        if _install([f"{p}>={v}" for p, v, _ in kurang], timeout=timeout):
            print("\n🔁 Verifikasi ulang...")
            rows2 = [(p, mv, *_check_package(p, mv)) for p, mv in parsed.items()]
            _print_table(rows2, "Setelah Install")
            still = [p for p, _, _, s in rows2 if s in ("missing", "outdated")]
            if still:
                print(f"\n⚠️  Masih bermasalah: {still}")
                miss = [p for p in still if _check_package(p, parsed[p])[1] == "missing"]
                old = [p for p in still if p not in miss]
                return {
                    "ok": False,
                    "need_restart": False,
                    "reqfile": reqfile,
                    "missing": miss,
                    "outdated": old,
                }
            need_restart = True
            print("\n✅ Semua package OK sekarang.")

    if need_restart:
        print("\n" + "═" * 60)
        print("🔄 RESTART KERNEL SEKARANG (Kernel → Restart Kernel)")
        print("═" * 60)

    miss = [p for p, _, _, s in rows if s == "missing"] if not need_restart else []
    old = [p for p, _, _, s in rows if s == "outdated"] if not need_restart else []
    return {
        "ok": not miss and not old,
        "need_restart": need_restart,
        "reqfile": reqfile,
        "missing": miss,
        "outdated": old,
    }


# ─────────────────────────────────────────────────────────
# CLI entrypoint
# ─────────────────────────────────────────────────────────
def main() -> int:
    auto = "--no-install" not in sys.argv
    try:
        result = check_and_setup_environment(auto_install=auto)
    except FileNotFoundError as e:
        print(f"❌ {e}", file=sys.stderr)
        return 1
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())