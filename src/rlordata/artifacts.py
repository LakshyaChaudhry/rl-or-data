"""Artifact store: copy run directories to durable storage. AGENT-OWNED; tasks/02a.

Lambda Cloud local disks vanish with the instance, so every eval / tier / training run syncs its
run directory at exit and the idle-terminate script (``setup/idle_shutdown.sh``) syncs everything
before terminating. The destination is ``RLORDATA_ARTIFACTS``:

- ``s3://bucket/prefix``  → boto3 ``upload_file`` per file (imported lazily; GPU extra)
- ``/lambda/nfs/<name>/rlordata-artifacts`` (or any local path) → ``shutil.copy2`` mirror

Layout under the destination mirrors the repo-relative path of the source
(``runs/eval/<model>/<split>/<decoding>/`` stays ``runs/eval/...``), so a sync from any machine
lands in the same place.

    python -m rlordata.artifacts sync runs/eval/Qwen__Qwen3-4B-Base
    python -m rlordata.artifacts sync-all        # runs/, data/splits/, data/samples/, configs/locked/
    python -m rlordata.artifacts restore runs data/samples   # new instance: copy the store back into the repo
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

from rlordata.envfile import load_env

ENV_VAR = "RLORDATA_ARTIFACTS"
DEFAULT_SYNC_ALL_ROOTS = ("runs", "data/splits", "data/samples", "configs/locked")


def artifacts_root() -> str | None:
    """``RLORDATA_ARTIFACTS`` or None (then syncing is a warned no-op)."""
    value = os.environ.get(ENV_VAR, "").strip()
    return value or None


def _relative_key(src: Path, repo_root: Path) -> Path:
    """Repo-relative path for ``src``; falls back to ``runs/<name>`` for paths outside the repo."""
    try:
        return src.resolve().relative_to(repo_root.resolve())
    except ValueError:
        return Path("runs") / src.name


def _iter_files(src: Path) -> list[Path]:
    if src.is_file():
        return [src]
    return sorted(p for p in src.rglob("*") if p.is_file())


def _parse_s3(url: str) -> tuple[str, str]:
    rest = url[len("s3://") :]
    bucket, _, prefix = rest.partition("/")
    if not bucket:
        raise ValueError(f"bad s3 url: {url!r}")
    return bucket, prefix.strip("/")


def sync_run_dir(
    run_dir: str | Path,
    dest_root: str | None = None,
    *,
    repo_root: str | Path | None = None,
    quiet: bool = False,
) -> str | None:
    """Mirror ``run_dir`` (file or directory) under ``dest_root``. Returns the destination or None.

    ``dest_root`` defaults to ``RLORDATA_ARTIFACTS``. When neither is set, prints a warning and
    returns None so callers never crash at exit because storage is unconfigured.
    """
    src = Path(run_dir)
    dest_root = dest_root if dest_root is not None else artifacts_root()
    if dest_root is None:
        if not quiet:
            print(
                f"[artifacts] {ENV_VAR} unset — NOT syncing {src}. On Lambda this data is lost at terminate.",
                file=sys.stderr,
            )
        return None
    if not src.exists():
        if not quiet:
            print(f"[artifacts] nothing to sync: {src} does not exist", file=sys.stderr)
        return None
    repo = Path(repo_root) if repo_root is not None else Path.cwd()
    key = _relative_key(src, repo)
    files = _iter_files(src)
    if dest_root.startswith("s3://"):
        dest = _sync_s3(src, files, dest_root, key)
    else:
        dest = _sync_local(src, files, Path(dest_root), key)
    if not quiet:
        print(f"[artifacts] synced {len(files)} file(s): {src} -> {dest}")
    return dest


def _sync_local(src: Path, files: list[Path], dest_root: Path, key: Path) -> str:
    dest = dest_root / key
    if src.is_file():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        return str(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for f in files:
        target = dest / f.relative_to(src)
        if _unchanged(f, target):
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, target)
    return str(dest)


def _unchanged(src: Path, target: Path) -> bool:
    """True when ``target`` already holds ``src`` (same size and mtime; ``copy2`` preserves mtime).

    A run dir is synced many times (every stage, the idle guard). Rewriting a finished 500 MB
    adapter with identical bytes on each of them is slow and, if the box dies mid-copy, truncates
    the only durable copy. Unchanged files are left alone.
    """
    if not target.is_file():
        return False
    a, b = src.stat(), target.stat()
    return a.st_size == b.st_size and int(a.st_mtime) == int(b.st_mtime)


def _sync_s3(src: Path, files: list[Path], dest_root: str, key: Path) -> str:
    try:
        import boto3  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - exercised on the GPU box only
        raise RuntimeError(
            "RLORDATA_ARTIFACTS is s3:// but boto3 is not installed (gpu extra)"
        ) from exc
    bucket, prefix = _parse_s3(dest_root)
    client = boto3.client("s3")
    base = "/".join(p for p in (prefix, key.as_posix()) if p)
    for f in files:
        rel = f.name if src.is_file() else f.relative_to(src).as_posix()
        s3_key = f"{base}/{rel}" if not src.is_file() else base
        client.upload_file(str(f), bucket, s3_key)
    return f"s3://{bucket}/{base}"


def restore(
    rel_paths: list[str],
    dest_root: str | None = None,
    *,
    repo_root: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, int]:
    """Copy repo-relative paths from the artifact store back into the repo (fresh instance).

    Existing local files are kept unless ``overwrite``. Returns ``{"copied": n, "skipped": m}``.
    """
    dest_root = dest_root if dest_root is not None else artifacts_root()
    if dest_root is None:
        raise RuntimeError(f"{ENV_VAR} unset; nothing to restore from")
    repo = Path(repo_root) if repo_root is not None else Path.cwd()
    copied = skipped = 0
    for rel in rel_paths:
        rel_path = Path(rel)
        if dest_root.startswith("s3://"):
            pairs = _s3_list(dest_root, rel_path)
        else:
            src_root = Path(dest_root) / rel_path
            if not src_root.exists():
                print(f"[artifacts] nothing stored at {src_root}", file=sys.stderr)
                continue
            files = _iter_files(src_root)
            pairs = [
                (f, rel_path / f.relative_to(src_root) if src_root.is_dir() else rel_path)
                for f in files
            ]
        for src, target_rel in pairs:
            target = repo / target_rel
            if target.exists() and not overwrite:
                skipped += 1
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(src, Path):
                shutil.copy2(src, target)
            else:  # (bucket, key) from S3
                import boto3  # type: ignore[import-not-found]

                boto3.client("s3").download_file(src[0], src[1], str(target))
            copied += 1
    print(f"[artifacts] restored {copied} file(s), kept {skipped} existing, from {dest_root}")
    return {"copied": copied, "skipped": skipped}


def _s3_list(dest_root: str, rel_path: Path) -> list[tuple[tuple[str, str], Path]]:
    import boto3  # type: ignore[import-not-found]

    bucket, prefix = _parse_s3(dest_root)
    base = "/".join(p for p in (prefix, rel_path.as_posix()) if p)
    client = boto3.client("s3")
    out: list[tuple[tuple[str, str], Path]] = []
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=base):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            rel = key[len(prefix) :].lstrip("/") if prefix else key
            out.append(((bucket, key), Path(rel)))
    return out


def sync_all(
    dest_root: str | None = None,
    roots: tuple[str, ...] = DEFAULT_SYNC_ALL_ROOTS,
    *,
    repo_root: str | Path | None = None,
) -> list[str]:
    """Sync every existing root in ``roots``; used by the idle-terminate script."""
    repo = Path(repo_root) if repo_root is not None else Path.cwd()
    done: list[str] = []
    for r in roots:
        p = repo / r
        if p.exists():
            dest = sync_run_dir(p, dest_root, repo_root=repo)
            if dest:
                done.append(dest)
    return done


def main(argv: list[str] | None = None) -> int:
    load_env()
    parser = argparse.ArgumentParser(prog="python -m rlordata.artifacts")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_sync = sub.add_parser("sync", help="sync one or more run directories")
    p_sync.add_argument("paths", nargs="+")
    p_sync.add_argument("--dest", default=None, help=f"override {ENV_VAR}")
    p_all = sub.add_parser("sync-all", help=f"sync {', '.join(DEFAULT_SYNC_ALL_ROOTS)}")
    p_all.add_argument("--dest", default=None)
    p_restore = sub.add_parser(
        "restore", help="copy repo-relative paths from the store back into the repo"
    )
    p_restore.add_argument("paths", nargs="+", help="e.g. runs data/samples data/splits")
    p_restore.add_argument(
        "--dest", default=None, help=f"override {ENV_VAR} (the store to read from)"
    )
    p_restore.add_argument("--overwrite", action="store_true", help="replace existing local files")
    args = parser.parse_args(argv)
    if args.cmd == "restore":
        counts = restore(args.paths, args.dest, overwrite=args.overwrite)
        return 0 if counts["copied"] or counts["skipped"] else 1
    if args.cmd == "sync":
        results = [sync_run_dir(p, args.dest) for p in args.paths]
    else:
        results = sync_all(args.dest)
    if artifacts_root() is None and args.dest is None:
        return 3  # loud non-zero so the idle script logs it
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
