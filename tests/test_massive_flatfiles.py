from __future__ import annotations

from datetime import datetime
import gzip
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from trainer.flatfile_inspect import inspect_cached_flatfile
from trainer.providers.massive_flatfiles import (
    MINUTE_AGGS_DATASET,
    MassiveFlatFileError,
    MassiveFlatFileStore,
    SOURCE,
)


ET = ZoneInfo("America/New_York")


def nanos(value: datetime) -> int:
    return int(value.timestamp() * 1_000_000_000)


def fixture_bytes() -> bytes:
    rows = [
        "close,ticker,window_start,volume,low,open,transactions,high",
        f"10.1,TEST,{nanos(datetime(2018, 1, 2, 6, 0, tzinfo=ET))},100,9.9,10.0,4,10.2",
        f"10.2,TEST,{nanos(datetime(2018, 1, 2, 9, 30, tzinfo=ET))},200,10.0,10.1,5,10.3",
        f"10.3,TEST,{nanos(datetime(2018, 1, 2, 16, 1, tzinfo=ET))},300,10.1,10.2,6,10.4",
    ]
    return gzip.compress(("\n".join(rows) + "\n").encode("utf-8"))


class FakePaginator:
    def __init__(self, objects):
        self.objects = objects
        self.calls = []

    def paginate(self, **kwargs):
        self.calls.append(kwargs)
        yield {"Contents": self.objects}


class FakeS3:
    def __init__(self, objects, payload):
        self.paginator = FakePaginator(objects)
        self.payload = payload
        self.downloads = []

    def get_paginator(self, operation):
        assert operation == "list_objects_v2"
        return self.paginator

    def download_file(self, bucket, key, filename):
        self.downloads.append((bucket, key, filename))
        Path(filename).write_bytes(self.payload)


def test_flatfile_parser_uses_header_names_and_classifies_sessions(tmp_path: Path):
    store = MassiveFlatFileStore(object(), cache_root=tmp_path)
    path = store.cache_path(MINUTE_AGGS_DATASET, "2018-01-02")
    path.parent.mkdir(parents=True)
    path.write_bytes(fixture_bytes())

    bars = list(store.iter_bars(MINUTE_AGGS_DATASET, "2018-01-02"))

    assert [bar["session"] for bar in bars] == [
        "PREMARKET",
        "REGULAR",
        "AFTERHOURS",
    ]
    assert bars[0]["timestamp"] == "2018-01-02T06:00:00-05:00"
    assert bars[0]["open"] == 10.0
    assert bars[0]["high"] == 10.2
    assert bars[0]["source"] == SOURCE


def test_flatfile_inspector_reports_raw_timestamps_and_session_buckets(
    tmp_path: Path,
):
    store = MassiveFlatFileStore(object(), cache_root=tmp_path)
    path = store.cache_path(MINUTE_AGGS_DATASET, "2018-01-02")
    path.parent.mkdir(parents=True)
    path.write_bytes(fixture_bytes())

    result = inspect_cached_flatfile(
        MINUTE_AGGS_DATASET,
        "2018-01-02",
        "TEST",
        cache_root=tmp_path,
    )

    assert result["total_rows"] == 3
    assert result["first_row"]["raw_timestamp"] == str(
        nanos(datetime(2018, 1, 2, 6, 0, tzinfo=ET))
    )
    assert result["first_row"]["parsed_et_timestamp"] == (
        "2018-01-02T06:00:00-05:00"
    )
    assert result["last_row"]["parsed_et_timestamp"] == (
        "2018-01-02T16:01:00-05:00"
    )
    assert result["session_row_counts"] == {
        "04:00-07:00": 1,
        "07:00-09:30": 0,
        "09:30-16:00": 1,
        "16:00-20:00": 1,
    }
    assert len(result["first_three_positive_volume_before_09_30_et"]) == 1


def test_s3_object_is_resolved_from_month_listing_and_verified_once(tmp_path: Path):
    payload = fixture_bytes()
    key = (
        "us_stocks_sip/minute_aggs_v1/2018/01/"
        "revision-2018-01-02-bars.csv.gz"
    )
    client = FakeS3(
        [{"Key": key, "Size": len(payload), "ETag": '"abc"'}],
        payload,
    )
    store = MassiveFlatFileStore(client, cache_root=tmp_path)

    first = store.ensure_day(MINUTE_AGGS_DATASET, "2018-01-02")
    second = store.ensure_day(MINUTE_AGGS_DATASET, "2018-01-02")

    assert first.object_key == key
    assert first.downloaded is True
    assert second.downloaded is False
    assert first.sha256 == second.sha256
    assert len(client.downloads) == 1
    assert client.paginator.calls[0]["Prefix"] == (
        "us_stocks_sip/minute_aggs_v1/2018/01/"
    )


def test_missing_date_names_the_listed_prefix(tmp_path: Path):
    client = FakeS3([], b"")
    store = MassiveFlatFileStore(client, cache_root=tmp_path)

    with pytest.raises(
        MassiveFlatFileError,
        match=r"us_stocks_sip/minute_aggs_v1/2018/01/",
    ):
        store.resolve_object(MINUTE_AGGS_DATASET, "2018-01-02")


def test_coverage_reports_present_missing_and_corrupt(tmp_path: Path):
    store = MassiveFlatFileStore(object(), cache_root=tmp_path)
    present = store.cache_path(MINUTE_AGGS_DATASET, "2018-01-02")
    present.parent.mkdir(parents=True)
    present.write_bytes(fixture_bytes())
    corrupt = store.cache_path(MINUTE_AGGS_DATASET, "2018-01-03")
    corrupt.write_bytes(b"not-gzip")

    report = store.coverage_report(
        ["2018-01-02", "2018-01-03", "2018-01-04"],
        datasets=[MINUTE_AGGS_DATASET],
    )["datasets"][MINUTE_AGGS_DATASET]

    assert report["present"] == ["2018-01-02"]
    assert report["corrupt"] == ["2018-01-03"]
    assert report["missing"] == ["2018-01-04"]
