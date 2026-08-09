"""CLI: python -m sa normalize|serve ..."""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(
            "usage:\n"
            "  python -m sa normalize <session_dir> [-o package_dir]\n"
            "  python -m sa serve <package_dir> [--port 8765]\n"
            "  python -m sa viz <session_dir>   # normalize + serve\n"
        )
        return 0

    cmd, rest = argv[0], argv[1:]
    if cmd == "normalize":
        from sa.normalize import main as norm_main

        return norm_main(rest)
    if cmd == "serve":
        from sa.serve import main as serve_main

        return serve_main(rest)
    if cmd in ("viz", "analyze", "open"):
        from pathlib import Path

        from sa.normalize import normalize_session, _read_json
        from sa.serve import serve

        if not rest:
            print("error: viz needs a session_dir", file=sys.stderr)
            return 1
        session_dir = Path(rest[0])
        out = None
        port = 8765
        i = 1
        while i < len(rest):
            if rest[i] in ("-o", "--out") and i + 1 < len(rest):
                out = Path(rest[i + 1])
                i += 2
            elif rest[i] == "--port" and i + 1 < len(rest):
                port = int(rest[i + 1])
                i += 2
            else:
                i += 1
        summary = _read_json(session_dir / "summary.json") or {}
        sid = (summary.get("info") or {}).get("id") or session_dir.name
        out = out or (Path(__file__).resolve().parent.parent / "packages" / str(sid))
        print(f"normalizing {session_dir} → {out}")
        normalize_session(session_dir, out)
        serve(out, port=port)
        return 0

    print(f"unknown command: {cmd}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
