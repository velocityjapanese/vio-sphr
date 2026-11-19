"""Push ViolinSpheric credentials into GitHub Actions secrets without printing them.

Reads the OAuth client JSON that Google gives you, extracts the three values,
and pipes each one straight into `gh secret set`. Secret values never touch
this script's stdout, the chat, or the repository.

Usage:
    python scripts/set_secrets.py --repo velocityjapanese/vio-sphr \
        --token-json "C:\\Users\\you\\Downloads\\client_secret_xxx.json"

Optional environment variables are forwarded when present:
    POLLINATIONS_API_KEY, CHANNEL_NAME,
    FB_PAGE_ID / FB_PAGE_ACCESS_TOKEN (or FACEBOOK_PAGE_ID / FACEBOOK_PAGE_ACCESS_TOKEN)
"""
import argparse
import json
import os
import shutil
import subprocess
import sys


def load_oauth(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for scope in ("installed", "web"):
        if isinstance(data.get(scope), dict):
            data = data[scope]
            break
    return {
        "YT_CLIENT_ID": data.get("client_id"),
        "YT_CLIENT_SECRET": data.get("client_secret"),
        "YT_REFRESH_TOKEN": data.get("refresh_token"),
    }


def set_secret(repo, name, value, dry_run=False):
    if not value:
        print(f"  - {name}: not present, skipped")
        return True
    if dry_run:
        print(f"  - {name}: would be set (dry run)")
        return True
    proc = subprocess.run(
        ["gh", "secret", "set", name, "--repo", repo],
        input=value, text=True, capture_output=True,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip().splitlines()
        message = detail[-1] if detail else "unknown error"
        print(f"  ! {name}: failed ({message})")
        return False
    print(f"  - {name}: set")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="owner/name, e.g. velocityjapanese/vio-sphr")
    ap.add_argument("--token-json", required=True, help="path to the downloaded OAuth client JSON")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not shutil.which("gh"):
        sys.exit("GitHub CLI (gh) is not installed or not on PATH.")
    if not os.path.exists(args.token_json):
        sys.exit(f"OAuth JSON not found: {args.token_json}")

    try:
        secrets = load_oauth(args.token_json)
    except Exception as exc:
        sys.exit(f"Could not read OAuth JSON: {exc}")

    missing = [k for k, v in secrets.items() if not v]
    if missing:
        print("Warning: these fields were missing from the file: " + ", ".join(missing))
        print("A refresh token is required. If you only have the client JSON, generate one")
        print("with the OAuth flow first (the token file usually includes refresh_token).")

    optional = {
        "POLLINATIONS_API_KEY": os.getenv("POLLINATIONS_API_KEY"),
        "CHANNEL_NAME": os.getenv("CHANNEL_NAME"),
        "FB_PAGE_ID": os.getenv("FB_PAGE_ID") or os.getenv("FACEBOOK_PAGE_ID"),
        "FB_PAGE_ACCESS_TOKEN": (os.getenv("FB_PAGE_ACCESS_TOKEN")
                                 or os.getenv("FACEBOOK_PAGE_ACCESS_TOKEN")),
    }

    print(f"Uploading secrets to {args.repo} (values are never printed):")
    ok = True
    for name, value in secrets.items():
        ok = set_secret(args.repo, name, value, args.dry_run) and ok
    for name, value in optional.items():
        if value:
            ok = set_secret(args.repo, name, value, args.dry_run) and ok
    if not any(optional.values()):
        print("  (no POLLINATIONS/CHANNEL/Facebook variables in this shell; add them in the GitHub UI if needed)")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
