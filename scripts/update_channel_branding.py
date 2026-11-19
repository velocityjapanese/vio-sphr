"""Update the ViolinSpheric channel's description, keywords and banner.

YouTube's Data API can set the channel banner (channelBanners.insert +
brandingSettings.image.bannerExternalUrl) and the description/keywords
(channels.update). It has no endpoint for the channel avatar, so that one
image still has to be uploaded in YouTube Studio.

Usage:
    python scripts/update_channel_branding.py \
        --token-json "path/to/token_ViolinSpheric.json" \
        --confirm-channel ViolinSpheric --apply
"""
import argparse
import json
import os
import sys

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from googleapiclient.http import MediaFileUpload  # noqa: E402
from publish_youtube import get_authenticated_service  # noqa: E402


def _load_env(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for scope in ("installed", "web"):
        if isinstance(data.get(scope), dict):
            data = data[scope]
            break
    os.environ["YOUTUBE_CLIENT_ID"] = data["client_id"]
    os.environ["YOUTUBE_CLIENT_SECRET"] = data["client_secret"]
    os.environ["YOUTUBE_REFRESH_TOKEN"] = data["refresh_token"]
    os.environ["YOUTUBE_TOKEN_URI"] = data.get("token_uri", "https://oauth2.googleapis.com/token")


def _read(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def _keywords(path):
    terms = [t.strip() for t in _read(path).split(",") if t.strip()]
    return " ".join(f'"{t}"' if " " in t else t for t in terms)


def _writable_branding(existing, description, keywords, banner_url=None):
    settings = existing.get("brandingSettings", {}) or {}
    channel = dict(settings.get("channel", {}) or {})
    channel["description"] = description
    channel["keywords"] = keywords
    body = {"channel": channel}
    if settings.get("watch"):
        body["watch"] = settings["watch"]
    if banner_url:
        body["image"] = {"bannerExternalUrl": banner_url}
    return body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--token-json", required=True)
    ap.add_argument("--confirm-channel", required=True,
                    help="safety gate: must match the authenticated channel title")
    ap.add_argument("--description-file",
                    default=os.path.join(PROJECT_DIR, "branding", "channel_description.txt"))
    ap.add_argument("--keywords-file",
                    default=os.path.join(PROJECT_DIR, "branding", "channel_keywords.txt"))
    ap.add_argument("--banner",
                    default=os.path.join(PROJECT_DIR, "branding", "channel_banner_2048x1152.jpg"))
    ap.add_argument("--skip-banner", action="store_true")
    ap.add_argument("--apply", action="store_true",
                    help="write changes; without this flag only reports the target channel")
    args = ap.parse_args()

    _load_env(args.token_json)
    yt = get_authenticated_service()
    resp = yt.channels().list(part="snippet,brandingSettings", mine=True).execute()
    items = resp.get("items", [])
    if len(items) != 1:
        sys.exit(f"Expected exactly one authenticated channel, found {len(items)}.")
    ch = items[0]
    title = (ch.get("snippet", {}) or {}).get("title", "")
    channel_id = ch.get("id", "")
    print(f"Authenticated channel: {title} ({channel_id})")
    if title.strip().lower() != args.confirm_channel.strip().lower():
        sys.exit("Channel title does not match --confirm-channel; aborting without changes.")

    description = _read(args.description_file)
    keywords = _keywords(args.keywords_file)
    print(f"Description characters: {len(description)} | keyword string: {keywords}")
    if not args.apply:
        print("Dry check only. Re-run with --apply to write the branding.")
        return

    banner_url = None
    if not args.skip_banner and os.path.exists(args.banner):
        print(f"Uploading banner: {os.path.basename(args.banner)}")
        media = MediaFileUpload(args.banner, mimetype="image/jpeg", resumable=False)
        result = yt.channelBanners().insert(
            body={"kind": "youtube#channelBanner"}, media_body=media).execute()
        banner_url = result.get("url")
        print("Banner uploaded." if banner_url else "Banner upload returned no URL.")

    body = {
        "id": channel_id,
        "brandingSettings": _writable_branding(ch, description, keywords, banner_url),
    }
    yt.channels().update(part="brandingSettings", body=body).execute()
    print("Channel description and keywords updated.")

    check = yt.channels().list(part="brandingSettings", mine=True).execute()["items"][0]
    bs = check.get("brandingSettings", {}) or {}
    bch = bs.get("channel", {}) or {}
    img = bs.get("image", {}) or {}
    print(f"Verified description chars: {len(bch.get('description') or '')}")
    print(f"Verified banner set: {bool(img.get('bannerExternalUrl') or img.get('bannerImageUrl'))}")
    print("Avatar note: YouTube has no API for the channel avatar. Upload "
          "branding/channel_avatar_800.png in YouTube Studio → Customisation → Branding.")


if __name__ == "__main__":
    main()