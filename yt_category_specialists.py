"""CLI tool to discover and analyze Korean YouTube category specialists."""
from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Set

import requests

API_BASE_URL = "https://www.googleapis.com/youtube/v3"
SEARCH_MAX_RESULTS = 50
CHANNELS_BATCH_SIZE = 50
VIDEOS_BATCH_SIZE = 50
PLAYLIST_ITEMS_MAX_RESULTS = 50
DEFAULT_PUBLISHED_WINDOW_DAYS = 14
DEFAULT_BACKOFF_BASE = 30.0
BUDGET_HIGH_DEFAULT = 10_000_000
SPECIALIST_THRESHOLD = 0.80


class BudgetExceeded(RuntimeError):
    """Raised when the API call budget has been exhausted."""


@dataclass
class VideoRecord:
    fetched_at: str
    video_id: str
    channel_id: str
    channel_title: str
    published_at: str
    title: str
    category_id: Optional[str]
    region: str
    lang_hint: Optional[str]


@dataclass
class ChannelRecord:
    channel_id: str
    channel_title: str
    subscriber_count: int
    country: Optional[str]
    uploads_playlist_id: Optional[str]
    kept_by_subs: bool


@dataclass
class AnalysisResult:
    channel_id: str
    channel_title: str
    uploads_sampled: int
    target_hits: int
    target_share: float
    is_specialist: bool


class KeyManager:
    """Rotates API keys when quota issues occur."""

    def __init__(self, keys: Sequence[str], logger: logging.Logger) -> None:
        if not keys:
            raise ValueError("At least one API key is required")
        self._keys = list(keys)
        self._index = 0
        self._rounds = 0
        self._logger = logger
        self._backoff_seconds = DEFAULT_BACKOFF_BASE

    def current(self) -> str:
        return self._keys[self._index]

    def rotate(self, reason: str) -> None:
        self._logger.warning("Rotating API key due to %s", reason)
        self._index = (self._index + 1) % len(self._keys)
        if self._index == 0:
            self._rounds += 1
            backoff = self._backoff_seconds * (2 ** (self._rounds - 1))
            jitter = random.uniform(0, backoff / 2)
            total_sleep = backoff + jitter
            self._logger.warning(
                "All API keys exhausted; sleeping for %.2f seconds before retry", total_sleep
            )
            time.sleep(total_sleep)

    def reset(self) -> None:
        self._rounds = 0
        self._backoff_seconds = DEFAULT_BACKOFF_BASE


class ApiContext:
    def __init__(self, budget: Optional[int]) -> None:
        self.budget = budget if budget and budget > 0 else None
        self.call_count = 0

    def register_call(self) -> None:
        self.call_count += 1
        if self.budget is not None and self.call_count > self.budget:
            raise BudgetExceeded("API call budget exceeded")


def load_api_keys(path: Path) -> List[str]:
    """Return non-empty API keys read from a plaintext file."""
    with path.open("r", encoding="utf-8") as handle:
        keys = [line.strip() for line in handle if line.strip()]
    return keys


def is_quota_error(response: requests.Response) -> bool:
    if response.status_code in (403, 429):
        return True
    try:
        payload = response.json()
    except ValueError:
        return False
    error = payload.get("error")
    if not error:
        return False
    reasons = []
    for err in error.get("errors", []):
        reason = err.get("reason")
        if reason:
            reasons.append(reason)
    quota_reasons = {
        "quotaExceeded",
        "dailyLimitExceeded",
        "rateLimitExceeded",
        "keyInvalid",
    }
    return any(reason in quota_reasons for reason in reasons)


def youtube_get(
    session: requests.Session,
    key_manager: KeyManager,
    context: ApiContext,
    endpoint: str,
    params: Dict[str, str],
    logger: logging.Logger,
) -> Dict:
    """Call a YouTube Data API endpoint with retries, key rotation, and quota tracking."""
    url = f"{API_BASE_URL}/{endpoint}"
    attempt = 0
    while True:
        attempt += 1
        key = key_manager.current()
        merged_params = dict(params)
        merged_params["key"] = key
        context.register_call()
        try:
            response = session.get(url, params=merged_params, timeout=15)
        except requests.RequestException as exc:
            wait = min(60.0, 2 ** attempt)
            jitter = random.uniform(0, 1.0)
            logger.warning("Network error on %s: %s; retrying in %.2fs", endpoint, exc, wait + jitter)
            time.sleep(wait + jitter)
            continue

        if response.status_code == 200:
            try:
                data = response.json()
            except ValueError as exc:
                logger.error("Failed to decode JSON for %s: %s", endpoint, exc)
                wait = min(60.0, 2 ** attempt)
                time.sleep(wait)
                continue
            return data

        if response.status_code in (500, 502, 503, 504):
            wait = min(120.0, 2 ** attempt)
            jitter = random.uniform(0, 1.0)
            logger.warning(
                "Server error %s on %s; retrying in %.2fs", response.status_code, endpoint, wait + jitter
            )
            time.sleep(wait + jitter)
            continue

        if is_quota_error(response):
            try:
                payload = response.json()
                reason = payload.get("error", {}).get("message", str(response.status_code))
            except ValueError:
                reason = f"HTTP {response.status_code}"
            key_manager.rotate(reason)
            continue

        try:
            payload = response.json()
        except ValueError:
            payload = {"error": {"message": response.text}}
        logger.error("HTTP %s error for %s: %s", response.status_code, endpoint, payload)
        response.raise_for_status()


def write_csv_row(path: Path, header: Sequence[str], row: Sequence[object]) -> None:
    file_exists = path.exists()
    with path.open("a" if file_exists else "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(row)


def parse_published_after(value: Optional[str]) -> str:
    """Compute the published-after timestamp, defaulting to a 14-day lookback."""
    if value:
        return value
    now = datetime.now(timezone.utc)
    window_start = now - timedelta(days=DEFAULT_PUBLISHED_WINDOW_DAYS)
    return window_start.isoformat().replace("+00:00", "Z")


def discover_videos(
    session: requests.Session,
    key_manager: KeyManager,
    context: ApiContext,
    category_id: str,
    max_videos: int,
    region: str,
    language: str,
    published_after: str,
    logger: logging.Logger,
    checkpoint: Dict,
) -> Iterator[VideoRecord]:
    """Yield discovered video records while updating pagination checkpoint data."""
    produced = 0
    next_page_token: Optional[str] = checkpoint.get("last_page_token")
    discovered_count = checkpoint.get("discovered_video_count", 0)
    seen_videos: Set[str] = set()
    while produced < max_videos:
        params = {
            "part": "snippet",
            "type": "video",
            "order": "date",
            "videoCategoryId": category_id,
            "regionCode": region,
            "relevanceLanguage": language,
            "publishedAfter": published_after,
            "maxResults": str(SEARCH_MAX_RESULTS),
        }
        if next_page_token:
            params["pageToken"] = next_page_token

        logger.info(
            "Requesting search.list (category=%s, pageToken=%s)", category_id, next_page_token or "<initial>"
        )
        response = youtube_get(session, key_manager, context, "search", params, logger)
        fetched_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        items = response.get("items", [])
        if not items:
            logger.info("No more search results returned; stopping discovery")
            next_page_token = None
            break

        video_ids = [item.get("id", {}).get("videoId") for item in items if item.get("id", {}).get("videoId")]
        snippets = fetch_videos_snippets(session, key_manager, context, video_ids, logger)

        for item in items:
            video_id = item.get("id", {}).get("videoId")
            snippet = item.get("snippet", {})
            if not video_id or not snippet:
                continue
            if video_id in seen_videos:
                continue
            video_snippet = snippets.get(video_id, {})
            record = VideoRecord(
                fetched_at=fetched_at,
                video_id=video_id,
                channel_id=snippet.get("channelId", ""),
                channel_title=snippet.get("channelTitle", ""),
                published_at=snippet.get("publishedAt", ""),
                title=snippet.get("title", ""),
                category_id=video_snippet.get("categoryId"),
                region=region,
                lang_hint=video_snippet.get("defaultLanguage")
                or video_snippet.get("defaultAudioLanguage")
                or snippet.get("defaultLanguage")
                or snippet.get("defaultAudioLanguage"),
            )
            produced += 1
            seen_videos.add(video_id)
            discovered_count += 1
            checkpoint["discovered_video_count"] = discovered_count
            checkpoint["last_page_token"] = next_page_token
            yield record
            if produced >= max_videos:
                break

        next_page_token = response.get("nextPageToken")
        checkpoint["last_page_token"] = next_page_token
        if not next_page_token:
            logger.info("Search pagination exhausted after %d videos", produced)
            break



def fetch_channels_stats(
    session: requests.Session,
    key_manager: KeyManager,
    context: ApiContext,
    channel_ids: Sequence[str],
    logger: logging.Logger,
) -> List[ChannelRecord]:
    """Fetch channel metadata and statistics for the provided channel identifiers."""
    records: List[ChannelRecord] = []
    for i in range(0, len(channel_ids), CHANNELS_BATCH_SIZE):
        batch_ids = channel_ids[i : i + CHANNELS_BATCH_SIZE]
        params = {
            "part": "snippet,contentDetails,statistics",
            "id": ",".join(batch_ids),
            "maxResults": str(len(batch_ids)),
        }
        logger.debug("Fetching channels.list for %d channels", len(batch_ids))
        response = youtube_get(session, key_manager, context, "channels", params, logger)
        for item in response.get("items", []):
            statistics = item.get("statistics", {})
            snippet = item.get("snippet", {})
            content_details = item.get("contentDetails", {})
            subs_text = statistics.get("subscriberCount")
            try:
                subs = int(subs_text)
            except (TypeError, ValueError):
                subs = 0
            if statistics.get("hiddenSubscriberCount"):
                subs = 0
            uploads_playlist_id = content_details.get("relatedPlaylists", {}).get("uploads")
            record = ChannelRecord(
                channel_id=item.get("id", ""),
                channel_title=snippet.get("title", ""),
                subscriber_count=subs,
                country=snippet.get("country"),
                uploads_playlist_id=uploads_playlist_id,
                kept_by_subs=False,
            )
            records.append(record)
    return records


def fetch_recent_uploads_ids(
    session: requests.Session,
    key_manager: KeyManager,
    context: ApiContext,
    playlist_id: str,
    max_items: int,
    logger: logging.Logger,
) -> List[str]:
    """Return up to ``max_items`` recent upload video IDs from a channel playlist."""
    collected: List[str] = []
    page_token: Optional[str] = None
    while len(collected) < max_items:
        params = {
            "part": "snippet,contentDetails",
            "playlistId": playlist_id,
            "maxResults": str(min(PLAYLIST_ITEMS_MAX_RESULTS, max_items - len(collected))),
        }
        if page_token:
            params["pageToken"] = page_token
        response = youtube_get(session, key_manager, context, "playlistItems", params, logger)
        items = response.get("items", [])
        if not items:
            break
        for item in items:
            video_id = item.get("contentDetails", {}).get("videoId")
            if not video_id:
                continue
            if video_id not in collected:
                collected.append(video_id)
            if len(collected) >= max_items:
                break
        page_token = response.get("nextPageToken")
        if not page_token:
            break
    return collected


def fetch_videos_snippets(
    session: requests.Session,
    key_manager: KeyManager,
    context: ApiContext,
    video_ids: Sequence[str],
    logger: logging.Logger,
) -> Dict[str, Dict]:
    """Fetch snippet payloads for the requested video IDs keyed by ID."""
    result: Dict[str, Dict] = {}
    clean_ids = [vid for vid in video_ids if vid]
    for i in range(0, len(clean_ids), VIDEOS_BATCH_SIZE):
        batch_ids = clean_ids[i : i + VIDEOS_BATCH_SIZE]
        params = {
            "part": "snippet",
            "id": ",".join(batch_ids),
            "maxResults": str(len(batch_ids)),
        }
        response = youtube_get(session, key_manager, context, "videos", params, logger)
        for item in response.get("items", []):
            vid = item.get("id")
            snippet = item.get("snippet", {})
            if vid:
                result[vid] = snippet
    return result


def analyze_channel_mix(
    session: requests.Session,
    key_manager: KeyManager,
    context: ApiContext,
    channel: ChannelRecord,
    uploads_to_sample: int,
    target_category_id: str,
    logger: logging.Logger,
) -> AnalysisResult:
    """Analyze a channel's recent uploads to calculate target-category share."""
    if not channel.uploads_playlist_id:
        logger.warning("Channel %s missing uploads playlist; treating as zero sample", channel.channel_id)
        return AnalysisResult(
            channel_id=channel.channel_id,
            channel_title=channel.channel_title,
            uploads_sampled=0,
            target_hits=0,
            target_share=0.0,
            is_specialist=False,
        )

    video_ids = fetch_recent_uploads_ids(
        session, key_manager, context, channel.uploads_playlist_id, uploads_to_sample, logger
    )
    snippets = fetch_videos_snippets(session, key_manager, context, video_ids, logger)
    target_hits = 0
    valid_sample = 0
    for vid in video_ids:
        snippet = snippets.get(vid)
        if not snippet:
            continue
        category_id = snippet.get("categoryId")
        if category_id:
            valid_sample += 1
            if category_id == target_category_id:
                target_hits += 1
    uploads_sampled = valid_sample
    target_share = (target_hits / uploads_sampled) if uploads_sampled else 0.0
    is_specialist = target_share >= SPECIALIST_THRESHOLD
    logger.info(
        "Channel %s: %d/%d uploads in category %s (share=%.2f) -> %s",
        channel.channel_id,
        target_hits,
        uploads_sampled,
        target_category_id,
        target_share,
        "SPECIALIST" if is_specialist else "general",
    )
    return AnalysisResult(
        channel_id=channel.channel_id,
        channel_title=channel.channel_title,
        uploads_sampled=uploads_sampled,
        target_hits=target_hits,
        target_share=target_share,
        is_specialist=is_specialist,
    )


def load_checkpoint(path: Path) -> Dict:
    """Load checkpoint state from disk if available."""
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_checkpoint(path: Path, data: Dict) -> None:
    """Persist checkpoint state to disk with an updated timestamp."""
    data["updated_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def configure_logging(level: str) -> logging.Logger:
    logger = logging.getLogger("yt_category_specialists")
    logger.setLevel(level.upper())
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    handler.setFormatter(formatter)
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Discover Korean YouTube category specialist channels.")
    parser.add_argument("--api-keys", required=True, help="Path to file containing API keys, one per line.")
    parser.add_argument("--category-id", required=True, help="YouTube category ID to target, e.g. '17'.")
    parser.add_argument("--max-videos", type=int, default=5000, help="Maximum number of videos to discover (1-5000).")
    parser.add_argument("--min-subs", type=int, default=10000, help="Minimum subscriber count for channels.")
    parser.add_argument(
        "--uploads-to-sample",
        type=int,
        default=50,
        help="Number of recent uploads per channel to analyze (max 200).",
    )
    parser.add_argument("--region", default="KR", help="Region code for video discovery (default: KR).")
    parser.add_argument("--language", default="ko", help="Language hint for video discovery (default: ko).")
    parser.add_argument(
        "--output-dir",
        default="./out",
        help="Directory where CSV outputs and checkpoints will be written.",
    )
    parser.add_argument(
        "--published-after",
        default=None,
        help="ISO8601 timestamp for earliest video publication (default: last 14 days).",
    )
    parser.add_argument(
        "--budget-calls",
        type=int,
        default=BUDGET_HIGH_DEFAULT,
        help="Optional cap on total API calls; defaults to a very high number.",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from existing checkpoint if available.")
    parser.add_argument("--dry-run", action="store_true", help="Run without API calls; print planned actions.")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO).",
    )
    return parser.parse_args(argv)


def dry_run_preview(args: argparse.Namespace, logger: logging.Logger) -> None:
    """Print the parameters that would be used for the first few discovery requests."""
    published_after = parse_published_after(args.published_after)
    base_params = {
        "part": "snippet",
        "type": "video",
        "order": "date",
        "videoCategoryId": args.category_id,
        "regionCode": args.region,
        "relevanceLanguage": args.language,
        "publishedAfter": published_after,
        "maxResults": str(SEARCH_MAX_RESULTS),
    }
    logger.info("Dry run: would initiate search.list with parameters:")
    logger.info(base_params)
    for i in range(1, 3):
        params = dict(base_params)
        params["pageToken"] = f"token_{i}"
        logger.info("Dry run page %d params: %s", i + 1, params)
    logger.info("Dry run complete; exiting without network calls.")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logger = configure_logging(args.log_level)

    if not (1 <= args.max_videos <= 5000):
        logger.error("--max-videos must be between 1 and 5000")
        return 2
    if not (1 <= args.uploads_to_sample <= 200):
        logger.error("--uploads-to-sample must be between 1 and 200")
        return 2

    api_key_path = Path(args.api_keys)
    if not api_key_path.exists():
        logger.error("API key file not found: %s", api_key_path)
        return 2

    keys = load_api_keys(api_key_path)
    if not keys:
        logger.error("No API keys found in %s", api_key_path)
        return 2

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    videos_csv = output_dir / "videos_raw.csv"
    channels_csv = output_dir / "channels_checked.csv"
    specialists_csv = output_dir / "specialist_channels.csv"
    analysis_csv = output_dir / "analysis_per_channel.csv"
    checkpoint_path = output_dir / "checkpoint.json"

    if args.dry_run:
        dry_run_preview(args, logger)
        return 0

    if not args.resume:
        for path in [videos_csv, channels_csv, specialists_csv, analysis_csv, checkpoint_path]:
            if path.exists():
                logger.info("Removing previous output: %s", path)
                path.unlink()

    checkpoint = load_checkpoint(checkpoint_path) if args.resume else {}
    checked_channels: Set[str] = set(checkpoint.get("checked_channels", []))
    analyzed_channels: Set[str] = set(checkpoint.get("analyzed_channels", []))

    key_manager = KeyManager(keys, logger)
    context = ApiContext(args.budget_calls)

    session = requests.Session()

    published_after = parse_published_after(args.published_after)

    def persist_checkpoint() -> None:
        checkpoint["checked_channels"] = sorted(checked_channels)
        checkpoint["analyzed_channels"] = sorted(analyzed_channels)
        save_checkpoint(
            checkpoint_path,
            {
                "last_page_token": checkpoint.get("last_page_token"),
                "discovered_video_count": checkpoint.get("discovered_video_count", 0),
                "checked_channels": sorted(checked_channels),
                "analyzed_channels": sorted(analyzed_channels),
            },
        )

    logger.info("Starting discovery up to %d videos", args.max_videos)
    discovered_videos: List[VideoRecord] = []
    try:
        for record in discover_videos(
            session,
            key_manager,
            context,
            args.category_id,
            args.max_videos,
            args.region,
            args.language,
            published_after,
            logger,
            checkpoint,
        ):
            discovered_videos.append(record)
            write_csv_row(
                videos_csv,
                [
                    "fetched_at",
                    "video_id",
                    "channel_id",
                    "channel_title",
                    "published_at",
                    "title",
                    "category_id",
                    "region",
                    "lang_hint",
                ],
                [
                    record.fetched_at,
                    record.video_id,
                    record.channel_id,
                    record.channel_title,
                    record.published_at,
                    record.title,
                    record.category_id,
                    record.region,
                    record.lang_hint,
                ],
            )
    except BudgetExceeded:
        persist_checkpoint()
        raise


    unique_channels: List[str] = []
    seen_channels: Set[str] = set()
    for record in discovered_videos:
        if record.channel_id and record.channel_id not in seen_channels:
            seen_channels.add(record.channel_id)
            unique_channels.append(record.channel_id)

    logger.info(
        "Discovered %d videos across %d unique channels", len(discovered_videos), len(unique_channels)
    )

    channels_to_check = [cid for cid in unique_channels if cid and cid not in checked_channels]
    logger.info("Need to fetch stats for %d channels", len(channels_to_check))

    kept_channels: List[ChannelRecord] = []
    try:
        for record in fetch_channels_stats(session, key_manager, context, channels_to_check, logger):
            kept = record.subscriber_count >= args.min_subs
            record.kept_by_subs = kept
            checked_channels.add(record.channel_id)
            write_csv_row(
                channels_csv,
                [
                    "checked_at",
                    "channel_id",
                    "channel_title",
                    "subscriber_count",
                    "country",
                    "uploads_playlist_id",
                    "kept_by_subs",
                ],
                [
                    datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                    record.channel_id,
                    record.channel_title,
                    record.subscriber_count,
                    record.country,
                    record.uploads_playlist_id,
                    kept,
                ],
            )
            if kept:
                kept_channels.append(record)
    except BudgetExceeded:
        persist_checkpoint()
        raise

    logger.info("Channels meeting subscriber threshold: %d", len(kept_channels))

    persist_checkpoint()

    specialists_found = 0
    analyzed_count = 0

    for channel in kept_channels:
        if channel.channel_id in analyzed_channels:
            logger.debug("Skipping already analyzed channel %s", channel.channel_id)
            continue
        try:
            analysis = analyze_channel_mix(
                session,
                key_manager,
                context,
                channel,
                args.uploads_to_sample,
                args.category_id,
                logger,
            )
        except BudgetExceeded:
            persist_checkpoint()
            raise
        analyzed_channels.add(channel.channel_id)
        analyzed_count += 1
        write_csv_row(
            analysis_csv,
            [
                "evaluated_at",
                "channel_id",
                "channel_title",
                "uploads_sampled",
                "target_hits",
                "target_share",
            ],
            [
                datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                analysis.channel_id,
                analysis.channel_title,
                analysis.uploads_sampled,
                analysis.target_hits,
                f"{analysis.target_share:.4f}",
            ],
        )
        if analysis.is_specialist:
            specialists_found += 1
            write_csv_row(
                specialists_csv,
                [
                    "evaluated_at",
                    "channel_id",
                    "channel_title",
                    "subscriber_count",
                    "uploads_sampled",
                    "target_category_id",
                    "target_share",
                    "is_specialist",
                ],
                [
                    datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                    channel.channel_id,
                    channel.channel_title,
                    channel.subscriber_count,
                    analysis.uploads_sampled,
                    args.category_id,
                    f"{analysis.target_share:.4f}",
                    True,
                ],
            )

        persist_checkpoint()

    persist_checkpoint()

    logger.info(
        "Summary: videos=%d, unique_channels=%d, checked=%d, analyzed=%d, specialists=%d, api_calls=%d",
        len(discovered_videos),
        len(unique_channels),
        len(checked_channels),
        analyzed_count,
        specialists_found,
        context.call_count,
    )

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BudgetExceeded:
        print("API call budget reached; checkpoint saved.")
        sys.exit(0)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Fatal error: {exc}", file=sys.stderr)
        sys.exit(1)
