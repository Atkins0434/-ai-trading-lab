"""Freshness policy correction; all other metric components remain identical."""
import csv
import hashlib
import json
from pathlib import Path
import runpy

import pytest

from trainer.catalyst_events import EVENT_TYPES, relevant_for_freshness
from trainer.news_cache import load_news_policy
from trainer.news_coverage import count_news_coverage, rollup_news_coverage
from trainer.research_scout_alpha import run_research_scout_alpha
from trainer.scorable_outcomes import export_daily_outcomes

ROOT = Path(__file__).resolve().parents[1]


def fixtures():
    return runpy.run_path(str(ROOT / 'tests/test_news_alpha.py'))


def test_earnings_unknown_at_three_hours_scores_three():
    f = fixtures()
    result = f['score']([f['article'](180, insights=[])])
    assert result['sentiment'] == 'UNKNOWN'
    assert result['freshest_event_age_minutes'] == 180
    assert result['components']['catalyst_freshness_relevance']['score'] == 3


@pytest.mark.parametrize('title', ['Company hosts a meeting', 'Company announces offering'])
def test_unknown_other_and_any_offering_are_not_relevant(title):
    f = fixtures()
    for sentiment in ('UNKNOWN', 'POSITIVE', 'MIXED'):
        if title == 'Company hosts a meeting' and sentiment != 'UNKNOWN':
            continue
        result = f['score']([f['article'](180, title=title, insights=[{'ticker':'TEST', 'sentiment':sentiment}])])
        assert result['components']['catalyst_freshness_relevance']['score'] == 0
        assert result['freshest_event_age_minutes'] is None


def test_complete_relevance_table():
    policy = load_news_policy()
    unconditional = {'EARNINGS','GUIDANCE','FDA_REGULATORY','MERGER_ACQUISITION','CONTRACT_AWARD','PRODUCT','SEC_FILING'}
    conditional = {'ANALYST_ACTION','MANAGEMENT','LEGAL_REGULATORY','MACRO_SECTOR','OTHER'}
    assert unconditional | conditional | {'OFFERING_DILUTION'} == EVENT_TYPES
    for event_type in EVENT_TYPES:
        for sentiment in ('UNKNOWN','NEUTRAL','NEGATIVE','POSITIVE','MIXED'):
            expected = event_type in unconditional or (event_type in conditional and sentiment in {'POSITIVE','MIXED'})
            assert relevant_for_freshness({'event_type':event_type,'sentiment':sentiment}, policy) == expected


@pytest.mark.parametrize('sentiment,expected', [(None,0), ('NEGATIVE',0), ('POSITIVE',4), ('MIXED',4)])
def test_keyword_hint_never_supplies_scoring_sentiment(sentiment,expected):
    f = fixtures()
    insights = [] if sentiment is None else [{'ticker':'TEST','sentiment':sentiment}]
    # 'approved' supplies a positive hint, but this headline classifies as OTHER.
    result = f['score']([f['article'](30, title='Company approved for marketplace', insights=insights)])
    assert result['best_event_type'] == 'OTHER'
    assert result['keyword_sentiment_hint'] == 'POSITIVE'
    assert result['sentiment'] == (sentiment or 'UNKNOWN')
    assert result['components']['catalyst_freshness_relevance']['score'] == expected


def test_insights_override_negative_hint():
    f = fixtures()
    result = f['score']([f['article'](30, title='Company misses estimates')])
    assert result['best_event_type'] == 'OTHER'
    assert result['sentiment'] == 'POSITIVE'
    assert result['keyword_sentiment_hint'] == 'NEGATIVE'
    assert result['components']['catalyst_freshness_relevance']['score'] == 4


def coverage(ages):
    candidates = [{
        'ticker':f'T{i}', 'status':'SCORED' if i % 2 else 'SHADOW_SCORED',
        'news_article_count':1 if age is not None else None,
        'admitted_event_count':1 if age is not None else None,
        'news_fetch_status':'AVAILABLE' if age is not None else 'MISSING',
        'freshest_event_age_minutes':age, 'source_tier':'OTHER', 'best_event_type':'EARNINGS',
    } for i, age in enumerate(ages)]
    return count_news_coverage(candidates, [], set())


def test_shares_and_pooled_quartiles_with_strict_age_boundaries():
    result = rollup_news_coverage([coverage([60,None]), coverage([180,720,1440,None])])
    assert result['scorable_count'] == 6
    assert result['share_with_relevant_event_under_12h'] == pytest.approx(2/6)
    assert result['share_with_relevant_event_under_24h'] == pytest.approx(3/6)
    assert result['freshest_relevant_event_age_minutes'] == {'p25':150, 'p50':450, 'p75':900}
    assert result['freshest_relevant_event_ages_minutes'] == [60,180,720,1440]
    assert result['by_price_tier']['UNKNOWN']['freshest_relevant_event_age_minutes'] == result['freshest_relevant_event_age_minutes']
    assert coverage([None])['share_with_relevant_event_under_12h'] == 0
    assert coverage([])['share_with_relevant_event_under_12h'] is None
    assert coverage([])['freshest_relevant_event_age_minutes'] == {'p25':None,'p50':None,'p75':None}


@pytest.mark.parametrize('ticker,gap,expected', [
    ('CDLX',52,'2fbca616d4672623fa2884b711104a36d1c8f4a736ea833360eb889c56fb8ab0'),
    ('SMR',-12,'9a515d1090b271ed71805fd97c049174ce9e72cc5c17dc50eb2c13459dbdd2bd'),
])
def test_all_other_twenty_four_components_byte_identical(ticker,gap,expected,tmp_path):
    # Hashes captured from main 3401659 on synthetic March paths with earnings.
    f = fixtures()
    snapshot = f['fixture_snapshot']()
    snapshot = json.loads(json.dumps(snapshot).replace('2026-09-14','2024-03-15').replace('2026-09-','2024-03-').replace('2026-08-','2024-02-'))
    item = snapshot['securities'][0]
    item['ticker'] = ticker
    item['market_data']['previous_close']['value'] = item['premarket_bars'][-1]['close']/(1+gap/100)
    item['news_context'] = f['context']([f['article'](180,ticker=ticker,insights=[])],ticker)
    result = run_research_scout_alpha(snapshot)
    candidate = result['candidates'][0]
    original = {k:v for k,v in candidate['component_scores'].items() if k != 'catalyst_freshness_relevance'}
    assert hashlib.sha256(json.dumps(original,sort_keys=True).encode()).hexdigest() == expected
    assert candidate['component_scores']['catalyst_freshness_relevance']['score'] == 3
    paths = export_daily_outcomes(tmp_path,snapshot,result,{'outcomes':[],'policy_comparisons':[]},{})
    with paths[0].open() as handle:
        row = next(csv.DictReader(handle))
    assert row['keyword_sentiment_hint'] == 'UNKNOWN'
    assert row['sentiment'] == 'UNKNOWN'
    assert row['freshest_event_age_minutes'] == '180.000000'
