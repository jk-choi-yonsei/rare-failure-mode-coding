"""Fetch the FDA MAUDE files for the second failure-mode corpus.

accessdata.fda.gov rejects HEAD, so sizes come from a ranged GET, and a
partial download resumes from where it stopped. FDA rewrites some of these
files monthly (foidevproblem.zip is cumulative), so the run writes
data/maude/MANIFEST.json with size, SHA-256 and Last-Modified for each file:
the pre-registration cites those hashes to pin the corpus a result used.

Usage: python maude_fetch.py          the 2024 corpus -> MANIFEST.json
       python maude_fetch.py 2025     2025 narratives for the out-of-time test
                                      -> MANIFEST_2025.json (codes come from the
                                      cumulative foidevproblem.zip already pinned)
"""
import hashlib
import json
import sys
import urllib.request

from common import DATA  # also injects truststore for corporate TLS

BASE = "https://www.accessdata.fda.gov/MAUDE/ftparea/"
FILES = [
    "deviceproblemcodes.zip",  # device problem code -> description
    "foidevproblem.zip",       # MDR report key -> device problem code, all years
    "foitext2024.zip",         # narratives of reports received in 2024
    "device2024.zip",          # device records (brand, manufacturer, product code)
]
FILES_BY_YEAR = {"2025": ["foitext2025.zip"]}
OUT = DATA / "maude"
CHUNK = 1 << 20


def remote_info(name):
    req = urllib.request.Request(BASE + name, headers={"Range": "bytes=0-63"})
    with urllib.request.urlopen(req, timeout=60) as r:
        total = int(r.headers["Content-Range"].rsplit("/", 1)[1])
        return total, r.headers.get("Last-Modified")


def fetch(name):
    total, modified = remote_info(name)
    dest = OUT / name
    if dest.exists() and dest.stat().st_size == total:
        print(f"{name}: already complete ({total:,} bytes)", flush=True)
        return total, modified
    part = OUT / (name + ".part")
    have = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(BASE + name, headers={"Range": f"bytes={have}-"})
    with urllib.request.urlopen(req, timeout=120) as r, part.open("ab") as fh:
        next_report = have + 50 * CHUNK
        while chunk := r.read(CHUNK):
            fh.write(chunk)
            have += len(chunk)
            if have >= next_report:
                print(f"  {name}: {have / total:.0%}", flush=True)
                next_report += 50 * CHUNK
    if have != total:
        raise SystemExit(f"{name}: got {have:,} of {total:,} bytes; rerun to resume")
    part.replace(dest)
    print(f"{name}: done ({total:,} bytes)", flush=True)
    return total, modified


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def main():
    year = sys.argv[1] if len(sys.argv) > 1 else None
    files = FILES_BY_YEAR[year] if year else FILES
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name in files:
        size, modified = fetch(name)
        manifest[name] = {"url": BASE + name, "bytes": size,
                          "last_modified": modified, "sha256": sha256(OUT / name)}
    (OUT / (f"MANIFEST_{year}.json" if year else "MANIFEST.json")).write_text(
        json.dumps(manifest, indent=1))
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
