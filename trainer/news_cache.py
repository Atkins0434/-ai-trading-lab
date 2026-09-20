"""Immutable per-ticker/day Massive news and bounded parallel enrichment."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re
import sys
from uuid import uuid4

from trainer.catalyst_events import _parse_aware
from trainer.flatfile_snapshot import _TickerProgress
from trainer.rate_control import load_massive_plan, load_reference_fetch_workers

ROOT = Path(__file__).resolve().parents[1]


def load_news_policy():
    return json.loads((ROOT / 'config/catalyst_policy_alpha_v1.json').read_text())


class NewsCache:
    def __init__(self, root=Path('data/news_cache')):
        self.root = Path(root)

    def path(self, ticker, trading_date):
        date.fromisoformat(trading_date)
        if not re.fullmatch(r'[A-Z][A-Z0-9.-]{0,14}', ticker):
            raise ValueError('Invalid news-cache ticker')
        return self.root / trading_date / f'{ticker}.json'

    def get(self, provider, ticker, trading_date, freeze, lookback_days):
        path = self.path(ticker, trading_date)
        start = _parse_aware(freeze, 'freeze') - timedelta(days=lookback_days)
        request = {'ticker': ticker, 'trading_date': trading_date, 'start': start.isoformat(), 'freeze': freeze}
        hit = path.exists()
        if not hit:
            response = provider.get_news_response(ticker, start.isoformat(), freeze)
            payload = {'request': request, 'fetched_at': datetime.now(timezone.utc).isoformat(),
                       'provider': provider.provider_name, 'feed_version': provider.feed_version, 'response': response}
            # Validate before storing; an unsuccessful query must not become an empty success.
            self.records(payload, request)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(f'.{path.name}.{uuid4().hex}.tmp')
            try:
                temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
                try:
                    os.link(temporary, path)  # Never replace a successful cached query.
                except FileExistsError:
                    hit = True
            finally:
                temporary.unlink(missing_ok=True)
        payload = json.loads(path.read_text())
        self.records(payload, request)
        return payload, hit

    @staticmethod
    def records(payload, request=None):
        if request is not None and payload['request'] != request:
            raise ValueError('News cache request differs; cached file is immutable')
        start = _parse_aware(payload['request']['start'], 'start')
        freeze = _parse_aware(payload['request']['freeze'], 'freeze')
        pages = payload['response']['pages']
        if not isinstance(pages, list) or not pages:
            raise ValueError('News response needs at least one completed page')
        result = []
        for page in pages:
            if not isinstance(page.get('results'), list):
                raise ValueError('News results must be an array')
            for record in page['results']:
                published = _parse_aware(record.get('published_utc'), 'published_utc')
                if start <= published <= freeze:
                    result.append(record)
        return result


def enrich_snapshot_news(snapshot, provider, *, cache_root=Path('data/news_cache')):
    from trainer.news_alpha import normalize_news_snapshot

    config = json.loads((ROOT / 'config/scout_alpha_v1.json').read_text())
    policy = load_news_policy()
    def real_bars(security):
        count = security.get('market_data', {}).get('real_bar_count_60m', {}).get('value')
        if count is not None:
            return count
        freeze = _parse_aware(snapshot['freeze_timestamp'], 'freeze')
        start = freeze - timedelta(minutes=60)
        return sum(start <= _parse_aware(bar['timestamp'], 'timestamp') < freeze
                   for bar in security.get('premarket_bars', []))

    securities = [s for s in snapshot.get('securities', [])
                  if real_bars(s) >= config['shadow_min_premarket_bars']]
    by_ticker = {}
    for security in securities:
        by_ticker.setdefault(security['ticker'], []).append(security)
    summary = {'hits': 0, 'fetches': 0, 'errors': 0, 'tickers_with_zero_articles': 0}
    if not by_ticker:
        return summary
    workers = 1 if load_massive_plan()['rest_calls_per_minute'] is not None else load_reference_fetch_workers()
    cache = NewsCache(cache_root)
    progress = _TickerProgress('news', len(by_ticker))
    print(f'[news] tickers=0/{len(by_ticker)} workers={workers}', file=sys.stderr, flush=True)

    def fetch(ticker):
        hit = False
        try:
            hit = cache.path(ticker, snapshot['trading_date']).exists()
            payload, hit = cache.get(provider, ticker, snapshot['trading_date'], snapshot['freeze_timestamp'], policy['lookback_days'])
            records = cache.records(payload)
            frozen = normalize_news_snapshot(records, ticker, snapshot['trading_date'], snapshot['freeze_timestamp'], payload, policy)
            return ticker, hit, {'status': 'AVAILABLE', 'article_count': len(records), 'snapshot': frozen}
        except Exception as exc:
            # Provider/cached-record failures affect only these three data-dependent metrics.
            print(f'[news] ticker={ticker} error={type(exc).__name__}', file=sys.stderr, flush=True)
            return ticker, hit, {'status': 'MISSING', 'article_count': None, 'error': type(exc).__name__}

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(fetch, ticker) for ticker in sorted(by_ticker)]
        for completed, future in enumerate(as_completed(futures), 1):
            ticker, hit, context = future.result()
            summary['hits' if hit else 'fetches'] += 1
            summary['errors'] += context['status'] == 'MISSING'
            summary['tickers_with_zero_articles'] += context['article_count'] == 0
            for security in by_ticker[ticker]:
                security['news_context'] = context
            progress.update(completed)
    return summary
