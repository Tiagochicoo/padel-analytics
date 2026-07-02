"""Patch app.py to add /api/video-file endpoint."""
from pathlib import Path

app_path = Path("/home/tpereira/rep/padel-analytics/src/server/app.py")
content = app_path.read_text()

old = '    return JSONResponse({"videos": videos})\n\n\nif __name__ == "__main__":'
new = '''    return JSONResponse({"videos": videos})\n\n\n@app.get("/api/video-file")\ndef video_file(path: str):\n    """Serve a video file for canvas-overlay annotated playback."""\n    from fastapi.responses import FileResponse\n    p = Path(path)\n    if not p.exists():\n        return JSONResponse({"ok": False, "error": "file not found"}, status_code=404)\n    # Security: only allow files from recordings dir\n    try:\n        p.resolve().relative_to(RECORDINGS_DIR.resolve())\n    except ValueError:\n        return JSONResponse({"ok": False, "error": "access denied"}, status_code=403)\n    return FileResponse(str(p))\n\n\nif __name__ == "__main__":'''

if old in content:
    content = content.replace(old, new)
    app_path.write_text(content)
    print("OK - video-file endpoint added")
else:
    print("ERROR: could not find insertion point")
    lines = content.split("\n")
    for line in lines[-15:]:
        print(f"  |{line}")
