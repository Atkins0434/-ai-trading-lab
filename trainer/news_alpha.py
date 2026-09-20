"""Research catalyst adapter over the existing neutral event pipeline."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import re

from trainer.catalyst_events import (
    _admit, _headline_key, build_catalyst_snapshot, calculate_research_catalyst_metrics,
)
from trainer.historical_news import normalize_historical_news_record
from trainer.validate_contracts import validate_contract

ROOT = Path(__file__).resolve().parents[1]
CATALYST_METRIC_IDS = ('catalyst_quality', 'catalyst_verification_confidence', 'catalyst_freshness_relevance')
NEWS_FIELDS = ('news_article_count', 'news_fetch_status', 'admitted_event_count', 'best_event_type', 'source_tier', 'verification_status', 'freshest_event_age_minutes', 'sentiment', 'keyword_sentiment_hint')


def normalize_news_snapshot(records, ticker, trading_date, freeze, envelope, policy):
    sources = json.loads((ROOT / 'config/news_sources.json').read_text())
    canonical = lambda text: re.sub(r'[^a-z0-9]', '', text.lower())
    tiers = {canonical(name): tier for name, tier in sources['publisher_tiers'].items()}
    events = []
    for record in records:
        event = normalize_historical_news_record(record, ticker=ticker, provider_name=envelope['provider'], feed_version=envelope['feed_version'], retrieved_at=envelope['fetched_at'])
        event['event_type_source'] = 'KEYWORDS'
        # News about a filing is not a verified SEC filing; there is no EDGAR input.
        if event['event_type'] == 'SEC_FILING':
            event['event_type'] = 'OTHER'
        has_insight = any(str(insight.get('ticker', '')).upper() == ticker.upper()
                          for insight in record.get('insights', []) or [])
        event['sentiment_source'] = 'MASSIVE_INSIGHTS' if has_insight else 'UNAVAILABLE'
        text = ' '.join([str(record.get('title', '')), str(record.get('description', '')), ' '.join(record.get('keywords') or [])]).lower()
        negative = any(word in text for word in sources['sentiment_keywords']['negative'])
        positive = any(word in text for word in sources['sentiment_keywords']['positive'])
        event['keyword_sentiment_hint'] = 'MIXED' if negative and positive else 'NEGATIVE' if negative else 'POSITIVE' if positive else 'UNKNOWN'
        event['positive_relevance'] = event['relevance_confidence'] if event['sentiment'] == 'POSITIVE' else 0.0
        event['source_tier'] = tiers.get(canonical(event['source_name']), sources['default_tier'])
        event['duplicate_group_id'] = f"{ticker}:{_headline_key(event['headline'])}"
        events.append(event)
    groups = defaultdict(list)
    for event in events:
        if _admit(event, freeze, policy)[0] == 'ADMITTED':
            groups[event['duplicate_group_id']].append(event)
    for group in groups.values():
        primary = any(e['source_tier'] == 'PRIMARY' for e in group)
        multiple = len({canonical(e['source_name']) for e in group}) >= 2
        for event in group:
            event['verification_status'] = 'VERIFIED_PRIMARY' if primary else 'VERIFIED_MULTI_SOURCE' if multiple else 'SINGLE_SOURCE'
    frozen = build_catalyst_snapshot(events, replay_id=f'{trading_date}-news-{ticker}', trading_date=trading_date, freeze_timestamp=freeze, policy_version=policy['policy_version'], policy=policy)
    validate_contract('catalyst_snapshot', frozen)
    return frozen


def news_scores(security, freeze):
    from trainer.news_cache import load_news_policy
    context = security.get('news_context', {})
    if context.get('status') != 'AVAILABLE':
        return {'components': {metric: {'status': 'MISSING', 'score': None, 'raw_value': None, 'maximum_score': 4, 'as_of_timestamp': None, 'reason_code': 'NEWS_UNAVAILABLE', 'calculation_version': 'research_catalyst_v1.0'} for metric in CATALYST_METRIC_IDS},
                **{key: None for key in NEWS_FIELDS}, 'news_fetch_status': 'MISSING'}
    frozen = context['snapshot']
    if frozen['freeze_timestamp'] != freeze:
        raise ValueError('Catalyst snapshot freeze differs from scoring freeze')
    return {**calculate_research_catalyst_metrics(frozen, security['ticker'], load_news_policy()),
            'news_article_count': context['article_count'], 'news_fetch_status': 'AVAILABLE'}
