# ViolinSpheric (`vio-sphr`)

**Repository:** https://github.com/kreggscode/vio-sphr
**Channel:** https://www.youtube.com/channel/UCWODeubE_VQ4Wyrk1zwLEVA
**Schedule:** daily at 08:00 UTC (one original suite per run)

A fully automated, daily long-form music channel: every run composes a brand-new
original violin + piano suite, renders it in 1080p with a photoreal violin whose
strings and bow follow the performance, builds a 1280x720 thumbnail from a real
frame, writes SEO metadata with AI (plus deterministic, policy-safe fallbacks),
and publishes to YouTube and Facebook.

Everything is generated locally or in GitHub Actions. No third-party music,
recordings, or stock footage is used.

## What one run does

1. **Compose** a seeded, original violin + piano piece (new piece every run, no
   reuse of previous titles).
2. **Render audio** with FluidSynth (MuseScore_General violin + Salamander
   piano), a convolution hall, and loudness normalisation to -16 LUFS.
3. **Render video** at 1920x1080 / 30 fps: lit stage, animated note ribbons,
   synced bow and finger motion on a CC0 photoreal violin, and a sheet-music
   panel.
4. **Thumbnail**: 1280x720 hero frame + brand row, kicker, title, episode chip,
   gold frame.
5. **SEO**: Pollinations (`gemini-fast`) title/description/tags, then a strict
   guard rejects false duration claims, composer/maker names, health claims and
   invented links; a deterministic template is used when the AI output fails.
6. **Upload**: YouTube Data API v3 resumable upload + custom thumbnail;
   optional Facebook Page mirror.
7. **Record**: `published_videos.json` is committed back by CI so episode
   numbers and piece titles stay unique.

The GitHub Actions workflow runs daily at `08:00 UTC` and can be triggered
manually with `upload_youtube`, `duration_minutes`, `dry_run`, and `smoke`
inputs.

## Layout

```
bot.py                     compose -> audio -> render -> mux (CLI)
composition.py             seeded composer (titles, sections, notes)
audio.py                   two-stem FluidSynth render + mastering
auto_pipeline.py           end-to-end orchestrator, SEO, history
thumbnail_generator.py     1280x720 thumbnail
publish_youtube.py         OAuth refresh-token upload + thumbnail
publish_facebook.py        optional Facebook Page upload
render/                    scene, violin layer, sheet-music panel, master
assets/                    OFL fonts, brand logo, CC0 violin photo
branding/                  channel avatar, banner, description, keywords
scripts/set_secrets.py     pushes secrets to GitHub without printing them
.github/workflows/         daily automation
```

## Local quick start

```bash
pip install -r requirements.txt
# ffmpeg + fluidsynth must be on PATH (or keep Windows binaries in bin/)
python bot.py --smoke --seed 42                 # 3-second motion check
python bot.py --minutes 2 --seed 123 --workers 2 --output preview.mp4
python auto_pipeline.py --video output/preview.mp4 --dry-run
```

Soundfonts are not committed (they are large). Put them in `soundfonts/`:

- `MuseScore_General.sf3` (violin) — MIT, S. Christian Collins
- `SalamanderGrandPiano.sf2` (piano) — CC BY 3.0, Alexander Holm
- `TimGM6mb.sf2` or `FluidR3_GM.sf2` as fallback

CI downloads and caches them automatically.

## Connect YouTube — keep the JSON out of chat and the repo

**Do not paste the OAuth JSON into chat, an issue, a commit, or this README.**
Nothing in the repo needs the raw file: the code reads the same values from
environment variables (locally) or GitHub Actions secrets (in CI).

### 1. Save the JSON outside the repository

Keep the downloaded client/token JSON somewhere private, for example:

```
C:\Users\<you>\.secrets\vio_sphr_oauth.json
```

It must contain `client_id`, `client_secret`, and `refresh_token` (flat, or
nested under `installed` / `web`).

### 2. Push the values as repository secrets

Once `gh` is logged in to the account that owns the repo:

```bash
python scripts/set_secrets.py \
  --repo kreggscode/vio-sphr \
  --token-json "C:\Users\<you>\.secrets\vio_sphr_oauth.json"
```

The helper pipes each field into `gh secret set` and prints only the secret
names. It also forwards `POLLINATIONS_API_KEY`, `CHANNEL_NAME`, and Facebook
credentials if they are present in the environment.

Manual alternative: repository **Settings → Secrets and variables → Actions**
and add `YT_CLIENT_ID`, `YT_CLIENT_SECRET`, `YT_REFRESH_TOKEN` one field at a
time.

### 3. Optional secrets

| Secret | Purpose |
| --- | --- |
| `POLLINATIONS_API_KEY` | AI titles/descriptions/tags (falls back safely without it) |
| `CHANNEL_NAME` | Overrides `ViolinSpheric` in branding |
| `FB_PAGE_ID`, `FB_PAGE_ACCESS_TOKEN` | Facebook Page mirror |

### Local runs

Copy `.env.example` to `.env` (gitignored) and fill the same names, or export
them in your shell. `publish_youtube.py` also accepts a local
`token_ViolinSpheric.json` in the project root; it is gitignored, but the
environment variables are preferred.

### Validate without uploading

```bash
python publish_youtube.py --validate
```

## Create / push the repository

```bash
gh auth login                       # pick the account that owns the repo
gh repo create kreggscode/vio-sphr --public --source . --remote origin --push
```

If the repository already exists:

```bash
git remote add origin https://github.com/kreggscode/vio-sphr.git
git push -u origin main
```

To publish under a different account, replace `kreggscode` with that account's
name and authenticate it with `gh auth login` first. Choose `--private` instead
of `--public` if you want the code hidden; GitHub Actions minutes for private
repositories are metered.

## Channel branding

The banner and the channel description/keywords can be applied with the API:

```bash
python scripts/update_channel_branding.py \
  --token-json "C:\Users\<you>\.secrets\vio_sphr_oauth.json" \
  --confirm-channel ViolinSpheric --apply
```

This uploads `branding/channel_banner_2048x1152.jpg`, sets
`branding/channel_description.txt`, and sets `branding/channel_keywords.txt`.
The script aborts unless the authenticated channel's title matches
`--confirm-channel`, so it cannot overwrite another channel.

YouTube has no API for the profile picture, so upload
`branding/channel_avatar_800.png` once in YouTube Studio → Customisation →
Branding. The same emblem is used in every video's corner watermark and in
thumbnails.

The same emblem is used in every video's corner watermark and in thumbnails.

## Publishing behavior

- New YouTube channels can only upload videos up to 15 minutes until phone
  verification is complete. Verify the channel, or run with `--minutes 10`,
  until then. The pipeline uses the real rendered duration in the title.
- If the Google Cloud OAuth consent screen is still in **Testing**, refresh
  tokens expire after 7 days. Publish the app (or keep re-authorising) for
  unattended daily runs; the preflight fails before rendering when a token is
  dead.
- The pipeline fails fast if the YouTube refresh token is missing or revoked,
  instead of rendering first and losing the run.
- Uploads are bounded (`PIA_YT_UPLOAD_TIMEOUT_SEC`, `PIA_YT_MAX_CHUNKS`) so a
  stalled connection cannot hang until the job timeout.
- A failed upload does not write history and the job exits non-zero, so nobody
  assumes a video was published.
- `published_videos.json` is the only file CI commits back.

## Copyright and attribution

- All music is original, algorithmically composed for the run; no copyrighted
  recordings or transcriptions are used.
- Violin body: "The Ex Kurtz Violin", The Metropolitan Museum of Art, CC0 1.0
  (see `assets/violin_body_credit.txt`).
- Violin samples: MuseScore_General SoundFont, MIT, S. Christian Collins.
- Piano samples: Salamander Grand Piano V3, CC BY 3.0, Alexander Holm.
- Fonts: Cormorant Garamond and Inter, SIL Open Font License.
- The render metadata embeds the full attribution, and the pipeline appends it
  to every video description.
