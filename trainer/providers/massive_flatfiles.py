from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable, Iterator
from zoneinfo import ZoneInfo

import boto3
from botocore.config import Config

from trainer.providers.base import ProviderError


MASSIVE_S3_ENDPOINT = "https://files.massive.com"
MASSIVE_S3_BUCKET = "flatfiles"
MINUTE_AGGS_DATASET = "us_stocks_sip/minute_aggs_v1"
DAY_AGGS_DATASET = "us_stocks_sip/day_aggs_v1"
SUPPORTED_DATASETS = {MINUTE_AGGS_DATASET, DAY_AGGS_DATASET}
SOURCE = "MASSIVE_FLATFILE"
ET = ZoneInfo("America/New_York")


class MassiveFlatFileError(ProviderError):
    """Raised when a Massive flat file cannot be located or admitted."""


@dataclass(frozen=True)
class FlatFileObject:
    dataset: str
    trading_date: str
    prefix: str
    key: str
    size: int
    etag: str | None


@dataclass(frozen=True)
class CachedFlatFile:
    dataset: str
    trading_date: str
    object_key: str
    local_path: str
    size: int
    sha256: str
    etag: str | None
    downloaded: bool

    def to_manifest(self) -> dict[str, Any]:
        return asdict(self)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_header(fieldnames: list[str] | None) -> set[str]:
    return {
        str(field).lstrip("\ufeff").strip().lower()
        for field in (fieldnames or [])
        if field is not None
    }


def _validate_csv_gzip(path: Path) -> None:
    required = {"ticker", "volume", "open", "close", "high", "low"}
    try:
        with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            header = _normalized_header(reader.fieldnames)
            if not required.issubset(header):
                raise MassiveFlatFileError(
                    f"Flat file {path} is missing CSV columns: "
                    f"{sorted(required - header)}"
                )
            if not ({"window_start", "timestamp", "t"} & header):
                raise MassiveFlatFileError(
                    f"Flat file {path} has no timestamp column."
                )
            next(reader, None)
    except (OSError, EOFError, UnicodeError, csv.Error) as exc:
        raise MassiveFlatFileError(
            f"Flat file {path} is not a valid gzip CSV: {exc}"
        ) from exc


def _parse_flatfile_timestamp(raw: str) -> datetime:
    value = str(raw).strip()
    try:
        numeric = int(value)
    except ValueError:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise MassiveFlatFileError(
                f"Invalid flat-file timestamp: {raw}"
            ) from exc
        if parsed.tzinfo is None:
            raise MassiveFlatFileError(
                "Flat-file ISO timestamps require a timezone."
            )
        return parsed.astimezone(ET)

    absolute = abs(numeric)
    if absolute >= 100_000_000_000_000_000:
        divisor = 1_000_000_000  # nanoseconds
    elif absolute >= 100_000_000_000_000:
        divisor = 1_000_000  # microseconds
    elif absolute >= 100_000_000_000:
        divisor = 1_000  # milliseconds
    else:
        divisor = 1  # seconds
    return datetime.fromtimestamp(
        numeric / divisor,
        tz=timezone.utc,
    ).astimezone(ET)


def _session(timestamp: datetime) -> str:
    observed = timestamp.timetz().replace(tzinfo=None)
    if time(4, 0) <= observed < time(9, 30):
        return "PREMARKET"
    if time(9, 30) <= observed < time(16, 0):
        return "REGULAR"
    return "AFTERHOURS"


def _row_value(row: dict[str, str], *names: str) -> str | None:
    normalized = {
        str(key).lstrip("\ufeff").strip().lower(): value
        for key, value in row.items()
        if key is not None
    }
    for name in names:
        value = normalized.get(name)
        if value is not None and str(value).strip() != "":
            return str(value).strip()
    return None


def parse_flatfile_row(
    row: dict[str, str],
    *,
    dataset: str,
    file_trading_date: str,
) -> dict[str, Any]:
    """Map a header-addressed CSV row to the provider-neutral bar shape."""
    if dataset not in SUPPORTED_DATASETS:
        raise MassiveFlatFileError(f"Unsupported Massive dataset: {dataset}")
    ticker = _row_value(row, "ticker", "symbol")
    raw_timestamp = _row_value(row, "window_start", "timestamp", "t")
    if ticker is None or raw_timestamp is None:
        raise MassiveFlatFileError("Flat-file row is missing ticker or timestamp.")
    timestamp = _parse_flatfile_timestamp(raw_timestamp)
    try:
        result = {
            "ticker": ticker.upper(),
            "timestamp": timestamp.isoformat(),
            "trading_date": file_trading_date,
            "open": float(_row_value(row, "open", "o") or "nan"),
            "high": float(_row_value(row, "high", "h") or "nan"),
            "low": float(_row_value(row, "low", "l") or "nan"),
            "close": float(_row_value(row, "close", "c") or "nan"),
            "volume": float(_row_value(row, "volume", "v") or "nan"),
            "transactions": (
                int(float(_row_value(row, "transactions", "n") or 0))
            ),
            "session": (
                "DAILY" if dataset == DAY_AGGS_DATASET else _session(timestamp)
            ),
            "source": SOURCE,
        }
    except ValueError as exc:
        raise MassiveFlatFileError(
            f"Flat-file row for {ticker} contains a non-numeric aggregate."
        ) from exc
    prices = [result[name] for name in ("open", "high", "low", "close")]
    if any(value <= 0 or value != value for value in prices):
        raise MassiveFlatFileError(
            f"Flat-file row for {ticker} contains an invalid price."
        )
    if result["volume"] < 0 or result["volume"] != result["volume"]:
        raise MassiveFlatFileError(
            f"Flat-file row for {ticker} contains invalid volume."
        )
    if result["high"] < max(prices) or result["low"] > min(prices):
        raise MassiveFlatFileError(
            f"Flat-file row for {ticker} contains inconsistent OHLC."
        )
    return result


class MassiveFlatFileStore:
    """S3 discovery, verified local caching, and CSV parsing for Massive."""

    provider_name = "MASSIVE"
    feed_version = "massive_sip_flatfiles_v1"

    def __init__(
        self,
        s3_client: Any,
        *,
        cache_root: Path = Path("data/flatfiles"),
        bucket: str = MASSIVE_S3_BUCKET,
    ) -> None:
        self.s3_client = s3_client
        self.cache_root = Path(cache_root)
        self.bucket = bucket

    @classmethod
    def from_environment(
        cls,
        *,
        cache_root: Path = Path("data/flatfiles"),
    ) -> "MassiveFlatFileStore":
        access_key = os.getenv("MASSIVE_S3_ACCESS_KEY_ID", "")
        secret_key = os.getenv("MASSIVE_S3_SECRET_ACCESS_KEY", "")
        missing = [
            name
            for name, value in (
                ("MASSIVE_S3_ACCESS_KEY_ID", access_key),
                ("MASSIVE_S3_SECRET_ACCESS_KEY", secret_key),
            )
            if not value
        ]
        if missing:
            raise MassiveFlatFileError(
                "Missing required Massive S3 environment variables: "
                + ", ".join(missing)
            )
        session = boto3.Session(
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )
        client = session.client(
            "s3",
            endpoint_url=MASSIVE_S3_ENDPOINT,
            config=Config(signature_version="s3v4"),
        )
        return cls(client, cache_root=cache_root)

    @staticmethod
    def month_prefix(dataset: str, trading_date: str) -> str:
        if dataset not in SUPPORTED_DATASETS:
            raise MassiveFlatFileError(
                f"Unsupported Massive dataset: {dataset}"
            )
        day = date.fromisoformat(trading_date)
        return f"{dataset}/{day.year}/{day.month:02d}/"

    def resolve_object(self, dataset: str, trading_date: str) -> FlatFileObject:
        """Resolve a dated object from a month listing, not a fixed key."""
        prefix = self.month_prefix(dataset, trading_date)
        target = re.compile(
            rf"(^|[^0-9]){re.escape(trading_date)}([^0-9]|$)"
        )
        matches: list[dict[str, Any]] = []
        paginator = self.s3_client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for item in page.get("Contents", []):
                key = str(item.get("Key", ""))
                if key.endswith(".csv.gz") and target.search(Path(key).name):
                    matches.append(item)
        if not matches:
            raise MassiveFlatFileError(
                f"No Massive flat file for {trading_date}; listed prefix "
                f"s3://{self.bucket}/{prefix}"
            )
        if len(matches) != 1:
            keys = sorted(str(item.get("Key")) for item in matches)
            raise MassiveFlatFileError(
                f"Ambiguous Massive flat files for {trading_date} under "
                f"s3://{self.bucket}/{prefix}: {keys}"
            )
        item = matches[0]
        return FlatFileObject(
            dataset=dataset,
            trading_date=trading_date,
            prefix=prefix,
            key=str(item["Key"]),
            size=int(item["Size"]),
            etag=(
                str(item.get("ETag", "")).strip('"') or None
            ),
        )

    def cache_path(self, dataset: str, trading_date: str) -> Path:
        day = date.fromisoformat(trading_date)
        return (
            self.cache_root
            / Path(dataset)
            / f"{day.year:04d}"
            / f"{trading_date}.csv.gz"
        )

    @staticmethod
    def metadata_path(path: Path) -> Path:
        return path.with_suffix(path.suffix + ".metadata.json")

    def _verified_cached(
        self,
        local_path: Path,
        remote: FlatFileObject,
    ) -> CachedFlatFile | None:
        if not local_path.is_file() or local_path.stat().st_size != remote.size:
            return None
        try:
            _validate_csv_gzip(local_path)
        except MassiveFlatFileError:
            return None
        digest = _sha256(local_path)
        metadata_path = self.metadata_path(local_path)
        if metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return None
            if (
                metadata.get("object_key") != remote.key
                or metadata.get("size") != remote.size
                or metadata.get("sha256") != digest
            ):
                return None
        cached = CachedFlatFile(
            dataset=remote.dataset,
            trading_date=remote.trading_date,
            object_key=remote.key,
            local_path=str(local_path),
            size=remote.size,
            sha256=digest,
            etag=remote.etag,
            downloaded=False,
        )
        _atomic_json(metadata_path, cached.to_manifest())
        return cached

    def ensure_day(self, dataset: str, trading_date: str) -> CachedFlatFile:
        remote = self.resolve_object(dataset, trading_date)
        local_path = self.cache_path(dataset, trading_date)
        verified = self._verified_cached(local_path, remote)
        if verified is not None:
            return verified

        local_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = local_path.with_suffix(local_path.suffix + ".part")
        try:
            self.s3_client.download_file(
                self.bucket,
                remote.key,
                str(temporary),
            )
            actual_size = temporary.stat().st_size
            if actual_size != remote.size:
                raise MassiveFlatFileError(
                    f"Downloaded size mismatch for {remote.key}: "
                    f"expected {remote.size}, received {actual_size}."
                )
            _validate_csv_gzip(temporary)
            digest = _sha256(temporary)
            temporary.replace(local_path)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise

        cached = CachedFlatFile(
            dataset=dataset,
            trading_date=trading_date,
            object_key=remote.key,
            local_path=str(local_path),
            size=remote.size,
            sha256=digest,
            etag=remote.etag,
            downloaded=True,
        )
        _atomic_json(self.metadata_path(local_path), cached.to_manifest())
        return cached

    def cache_status(self, dataset: str, trading_date: str) -> str:
        path = self.cache_path(dataset, trading_date)
        if not path.exists():
            return "MISSING"
        if not path.is_file() or path.stat().st_size <= 0:
            return "CORRUPT"
        try:
            _validate_csv_gzip(path)
            metadata_path = self.metadata_path(path)
            if metadata_path.exists():
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                if metadata.get("size") != path.stat().st_size:
                    return "CORRUPT"
                if metadata.get("sha256") != _sha256(path):
                    return "CORRUPT"
        except (MassiveFlatFileError, json.JSONDecodeError, OSError):
            return "CORRUPT"
        return "PRESENT"

    def coverage_report(
        self,
        trading_dates: Iterable[str],
        *,
        datasets: Iterable[str] = (
            MINUTE_AGGS_DATASET,
            DAY_AGGS_DATASET,
        ),
    ) -> dict[str, Any]:
        dates = sorted({date.fromisoformat(value).isoformat() for value in trading_dates})
        report: dict[str, Any] = {"datasets": {}}
        for dataset in datasets:
            statuses = {
                value: self.cache_status(dataset, value) for value in dates
            }
            report["datasets"][dataset] = {
                "present": [value for value, status in statuses.items() if status == "PRESENT"],
                "missing": [value for value, status in statuses.items() if status == "MISSING"],
                "corrupt": [value for value, status in statuses.items() if status == "CORRUPT"],
            }
        return report

    def iter_bars(
        self,
        dataset: str,
        trading_date: str,
        *,
        tickers: set[str] | None = None,
    ) -> Iterator[dict[str, Any]]:
        path = self.cache_path(dataset, trading_date)
        if self.cache_status(dataset, trading_date) != "PRESENT":
            raise MassiveFlatFileError(
                f"Cached flat file is not present and valid: {path}"
            )
        ticker_filter = (
            {ticker.upper() for ticker in tickers} if tickers is not None else None
        )
        try:
            with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                for row_number, row in enumerate(reader, start=2):
                    try:
                        bar = parse_flatfile_row(
                            row,
                            dataset=dataset,
                            file_trading_date=trading_date,
                        )
                    except MassiveFlatFileError as exc:
                        raise MassiveFlatFileError(
                            f"{path}:{row_number}: {exc}"
                        ) from exc
                    if ticker_filter is None or bar["ticker"] in ticker_filter:
                        yield bar
        except (OSError, EOFError, UnicodeError, csv.Error) as exc:
            raise MassiveFlatFileError(
                f"Unable to parse cached flat file {path}: {exc}"
            ) from exc
