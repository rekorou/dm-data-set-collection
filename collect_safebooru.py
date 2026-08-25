#!/usr/bin/env python3
"""
IllustraMeta collector -- Safebooru adapter.

  python collect_safebooru.py --probe      # field/health report, run FIRST
  python collect_safebooru.py --pilot      # 50 per class
  python collect_safebooru.py --target 300 # full run

Requires: pip install requests
"""

import argparse
import csv
import hashlib
import os
import random
import re
import time
from datetime import datetime, timezone
from urllib.parse import urlparse

import requests

BASE = "https://safebooru.org/index.php"
POST_URL = BASE + "?page=post&s=view&id={}"
HEADERS = {"User-Agent": "MMCM-CS-DataMining-StudentProject/1.0 (academic coursework)"}

DELAY_MIN, DELAY_MAX = 1.0, 2.0
PAGE_SIZE = 100          # Gelbooru-family APIs cap at 100
AI_TAG = "ai-generated"  # confirmed present on safebooru.org

# --- leakage control -------------------------------------------------------
# Anything naming the generation method must never reach the feature matrix.
LEAK_PATTERNS = [
    r"ai[-_ ]?generated", r"ai[-_ ]?assisted", r"stable[-_ ]?diffusion",
    r"novel[-_ ]?ai", r"nai[-_ ]?helper", r"nai[-_ ]?diffusion", r"midjourney",
    r"dall[-_ ]?e", r"waifu[-_ ]?diffusion", r"comfy[-_ ]?ui", r"automatic1111",
    r"^ai$", r"ai[-_ ]?art", r"machine[-_ ]?generated", r"tagme",
]
LEAK_RE = re.compile("|".join(LEAK_PATTERNS), re.IGNORECASE)

STANDARD_DIMS = {512, 640, 768, 832, 896, 1024, 1152, 1216, 1344, 1536}


def polite_sleep():
    time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))


def api_get(tags, pid=0, limit=PAGE_SIZE):
    params = {"page": "dapi", "s": "post", "q": "index", "json": 1,
              "limit": limit, "pid": pid, "tags": tags}
    for attempt in range(3):
        try:
            r = requests.get(BASE, params=params, headers=HEADERS, timeout=30)
            if r.status_code != 200:
                print(f"  HTTP {r.status_code}")
                time.sleep(5)
                continue
            if not r.text.strip():
                return []          # empty body = no more results
            data = r.json()
            # Some forks wrap results in {"post": [...]}
            if isinstance(data, dict):
                data = data.get("post", [])
            return data if isinstance(data, list) else []
        except Exception as e:
            print(f"  request failed ({e}), retry {attempt+1}/3")
            time.sleep(5)
    return []


# --- feature engineering ---------------------------------------------------

def strip_leak_tags(tag_string):
    tags = [t for t in (tag_string or "").split() if t]
    clean = [t for t in tags if not LEAK_RE.search(t)]
    return clean, len(tags) - len(clean)


def source_type(source):
    """
    Coarse category only. The raw string sometimes names the generator
    (e.g. a NAIHelper filename), which would leak the label outright.
    A human uploading from disk and a prompter uploading from disk both
    land in 'local_file', which is the point.
    """
    s = (source or "").strip()
    if not s:
        return "none"
    if s.startswith("file:"):
        return "local_file"
    host = (urlparse(s).netloc or "").lower().replace("www.", "")
    if not host:
        return "other"
    if "pximg" in host or "pixiv" in host:
        return "pixiv"
    if host in ("x.com", "twitter.com") or "twimg" in host:
        return "twitter"
    if "fanbox" in host or "patreon" in host:
        return "paywall"
    if "tumblr" in host or "deviantart" in host or "artstation" in host:
        return "art_platform"
    return "other"


def safe_div(a, b):
    return round(a / b, 6) if b else None


def flatten(post, label):
    w = int(post.get("width") or 0)
    h = int(post.get("height") or 0)

    clean_tags, n_removed = strip_leak_tags(post.get("tags", ""))
    tag_lens = [len(t) for t in clean_tags]

    # `change` is last-modified, not upload time. Weak temporal proxy --
    # note this as a limitation in the writeup.
    hour = dow = month = None
    try:
        ts = int(post.get("change") or 0)
        if ts:
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            hour, dow, month = dt.hour, dt.strftime("%A"), dt.strftime("%B")
    except Exception:
        pass

    owner = post.get("owner")
    score = post.get("score")

    return {
        "post_id": post.get("id"),
        "post_url": POST_URL.format(post["id"]) if post.get("id") else None,
        "owner_hashed": hashlib.sha256(str(owner).encode()).hexdigest()[:12] if owner else None,

        "is_ai_generated": label,

        # dimensions
        "image_width": w or None,
        "image_height": h or None,
        "aspect_ratio": safe_div(w, h),
        "megapixels": round((w * h) / 1_000_000, 4) if w and h else None,
        "is_standard_resolution": int(w in STANDARD_DIMS or h in STANDARD_DIMS),
        "is_portrait": int(h > w) if w and h else None,
        "long_side": max(w, h) if w and h else None,

        # tag structure (counts only -- raw strings deliberately excluded)
        "tag_count": len(clean_tags),
        "avg_tag_length": round(sum(tag_lens) / len(tag_lens), 3) if tag_lens else None,
        "max_tag_length": max(tag_lens) if tag_lens else None,
        "multiword_tag_ratio": safe_div(sum("_" in t for t in clean_tags), len(clean_tags)),

        # provenance (coarse category only -- see source_type docstring)
        "source_type": source_type(post.get("source")),
        "has_source": int(bool((post.get("source") or "").strip())),

        # engagement -- may be null site-wide; --probe will tell you
        "score": score,
        "comment_count": post.get("comment_count"),
        "has_notes": int(bool(post.get("has_notes"))),

        # post structure
        "has_parent": int(bool(post.get("parent_id"))),
        "has_sample": int(bool(post.get("sample"))),

        # temporal (proxy)
        "change_hour": hour,
        "change_day_of_week": dow,
        "change_month": month,

        # audit only -- DROP before modeling, it tracks the label by construction
        "_leak_tags_removed": n_removed,
    }


# --- collection ------------------------------------------------------------

def collect(tags, label, target, filter_ai=False, id_window=None,
            spread_pages=0):
    """
    id_window: (min_id, max_id). Keep only posts inside it. Used to draw the
      human class from the same period as the AI class -- without this the
      human sample is just the newest posts on the site, and any temporal
      feature becomes a recency detector rather than an AI detector.
    spread_pages: sample page offsets randomly across this range instead of
      walking 0,1,2... Prevents the sample collapsing onto one day.
    """
    rows, seen, stall = [], set(), 0
    rejected_window = 0

    if spread_pages:
        page_queue = list(range(spread_pages))
        random.shuffle(page_queue)
    else:
        page_queue = None
    pid = 0

    print(f"\nCollecting label={label}  tags={tags!r}  target={target}")
    if id_window:
        print(f"  restricted to post IDs {id_window[0]}-{id_window[1]}")

    while len(rows) < target and stall < 6:
        if page_queue is not None:
            if not page_queue:
                break
            pid = page_queue.pop()
        batch = api_get(tags, pid=pid)
        if not batch:
            stall += 1
            if page_queue is None:
                pid += 1
            continue
        added = 0
        for post in batch:
            pid_ = post.get("id")
            if pid_ is None or pid_ in seen:
                continue
            if post.get("status") not in (None, "active"):
                continue
            if not post.get("width"):
                continue
            if filter_ai and LEAK_RE.search(post.get("tags", "")):
                continue
            if id_window and not (id_window[0] <= pid_ <= id_window[1]):
                rejected_window += 1
                continue
            seen.add(pid_)
            rows.append(flatten(post, label))
            added += 1
            if len(rows) >= target:
                break
        stall = stall + 1 if added == 0 else 0
        if page_queue is None:
            pid += 1
        print(f"  {len(rows)}/{target}")
        polite_sleep()

    if rejected_window:
        print(f"  rejected {rejected_window} posts outside the ID window")
    return rows


# --- diagnostics -----------------------------------------------------------

def probe():
    """Field inventory + health check. Run this before collecting anything."""
    print("Probing Safebooru...\n")
    ai = api_get(f"{AI_TAG} rating:general", limit=20)
    hu = api_get(f"-{AI_TAG} rating:general", limit=20)

    print(f"AI-tagged records returned:    {len(ai)}")
    print(f"Human (negated) records:       {len(hu)}")
    if not hu:
        print("  !! Negated search returned nothing -- the run will fall back")
        print("     to client-side filtering automatically.")
    if not ai:
        print("  !! No AI records. Check the tag name and stop here.")
        return

    print("\nField inventory (from AI sample):")
    keys = sorted({k for p in ai for k in p})
    for k in keys:
        vals = [p.get(k) for p in ai]
        nulls = sum(v is None or v == "" for v in vals)
        uniq = len({str(v) for v in vals})
        flag = ""
        if nulls == len(vals):
            flag = "  <-- ALWAYS NULL, feature is dead"
        elif uniq == 1:
            flag = "  <-- CONSTANT, feature is dead"
        print(f"  {k:<18} nulls {nulls}/{len(vals)}  unique {uniq}{flag}")


def leak_scan(rows):
    """Any feature that alone separates the classes perfectly is a leak."""
    print("\nLEAK SCAN")
    y = [r["is_ai_generated"] for r in rows]
    found = False
    for col in rows[0]:
        if col in ("is_ai_generated", "post_id", "post_url", "owner_hashed"):
            continue
        a = {r[col] for r, lab in zip(rows, y) if lab == 1 and r[col] is not None}
        b = {r[col] for r, lab in zip(rows, y) if lab == 0 and r[col] is not None}
        if a and b and not (a & b):
            print(f"  LEAK: {col} -- perfectly separates the classes. Drop it.")
            found = True
    if not found:
        print("  clean -- no single feature separates the classes")


def write_csv(rows, path):
    if not rows:
        print("Nothing to write.")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print(f"\nWrote {len(rows)} rows -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true", help="field health check -- RUN FIRST")
    ap.add_argument("--pilot", action="store_true", help="50 per class")
    ap.add_argument("--target", type=int, default=300, help="records PER CLASS")
    ap.add_argument("--out", default="illustrameta_safebooru.csv")
    args = ap.parse_args()

    if args.probe:
        probe()
        return

    target = 50 if args.pilot else args.target
    out = "illustrameta_pilot.csv" if args.pilot else args.out

    # AI class first -- it is the scarce one, so it defines the time window.
    ai_rows = collect(f"{AI_TAG} rating:general", 1, target)
    if not ai_rows:
        print("No AI records collected. Run --probe and check the tag name.")
        return

    ai_ids = [r["post_id"] for r in ai_rows if r["post_id"]]
    window = (min(ai_ids), max(ai_ids))
    print(f"\nAI class spans post IDs {window[0]}-{window[1]}")
    print("Drawing the human class from the same window so that temporal")
    print("features cannot stand in for the label.")

    # Widen slightly so the human class isn't starved at the edges.
    pad = int((window[1] - window[0]) * 0.05)
    window = (window[0] - pad, window[1] + pad)

    human_rows = collect(f"-{AI_TAG} rating:general", 0, target,
                         id_window=window, spread_pages=300)

    if len(human_rows) < target * 0.5:
        print("\nWindow-matched sampling came up short. Retrying unrestricted --")
        print("if this path runs, DROP the temporal features before modeling.")
        human_rows = collect(f"-{AI_TAG} rating:general", 0, target,
                             spread_pages=300)

    rows = ai_rows + human_rows
    random.shuffle(rows)
    write_csv(rows, out)

    n_ai = sum(r["is_ai_generated"] for r in rows)
    print(f"\nTotal: {len(rows)}  (minimum required: 500)")
    if len(rows) < 500 and not args.pilot:
        print("  *** BELOW MINIMUM -- raise --target")
    print(f"Balance: {n_ai} AI / {len(rows) - n_ai} human")

    # Temporal overlap check -- if the ID ranges barely overlap, any
    # time-derived feature is really a recency detector.
    ai_ids = sorted(r["post_id"] for r in rows if r["is_ai_generated"] == 1 and r["post_id"])
    hu_ids = sorted(r["post_id"] for r in rows if r["is_ai_generated"] == 0 and r["post_id"])
    if ai_ids and hu_ids:
        lo = max(min(ai_ids), min(hu_ids))
        hi = min(max(ai_ids), max(hu_ids))
        span = max(max(ai_ids), max(hu_ids)) - min(min(ai_ids), min(hu_ids))
        overlap = max(0, hi - lo) / span if span else 0
        print(f"\nID range  AI: {min(ai_ids)}-{max(ai_ids)}")
        print(f"ID range  human: {min(hu_ids)}-{max(hu_ids)}")
        print(f"Overlap: {overlap:.0%}")
        if overlap < 0.7:
            print("  *** LOW OVERLAP -- drop change_hour/day/month before modeling")

    leak_scan(rows)


if __name__ == "__main__":
    main()
