from copy import deepcopy
from datetime import datetime, timedelta
import csv
import hashlib
import json
from pathlib import Path
import runpy

import pytest

from trainer.catalyst_events import VERIFICATION_POINTS
from trainer.news_alpha import CATALYST_METRIC_IDS, normalize_news_snapshot, news_scores
from trainer.news_cache import NewsCache, enrich_snapshot_news, load_news_policy
from trainer.news_coverage import count_news_coverage, rollup_news_coverage
from trainer.research_scout_alpha import run_research_scout_alpha
from trainer.scorable_outcomes import export_daily_outcomes
from trainer.scorable_outcomes_schema import METRIC_IDS, scorable_outcomes_columns

ROOT=Path(__file__).resolve().parents[1]
FREEZE='2024-03-15T09:15:00-04:00'
ENVELOPE={'provider':'MASSIVE','feed_version':'test_news_v1','fetched_at':'2026-09-20T00:00:00+00:00'}


def article(age_minutes=30, publisher='Reuters', title='Company reports earnings', ticker='TEST', **fields):
    return {'id':title+publisher, 'title':title, 'published_utc':(datetime.fromisoformat(FREEZE)-timedelta(minutes=age_minutes)).isoformat(),
            'publisher':{'name':publisher},'tickers':[ticker], 'insights':[{'ticker':ticker,'sentiment':'positive'}], **fields}


def context(records, ticker='TEST'):
    return {'status':'AVAILABLE','article_count':len(records),'snapshot':normalize_news_snapshot(records,ticker,'2024-03-15',FREEZE,ENVELOPE,load_news_policy())}


def score(records):
    return news_scores({'ticker':'TEST','news_context':context(records)},FREEZE)


@pytest.mark.parametrize('minutes,expected',[(15,1),(14.999,0)])
def test_latency_admission_boundary(minutes,expected):
    result=context([article(minutes)])['snapshot']
    assert result['summary']['admitted']==expected
    event=result['events'][0]
    assert event['provider_available_timestamp'] is None
    assert event['admission_reason']==('ASSUMED_AVAILABLE_WITH_LATENCY' if expected else 'AVAILABLE_AFTER_FREEZE')


def test_duplicate_is_counted_once_and_verification_uses_only_admitted_publishers():
    records=[article(publisher='Reuters'),article(publisher='Bloomberg')]
    result=score(records)
    assert result['admitted_event_count']==1
    assert result['verification_status']=='VERIFIED_MULTI_SOURCE'
    assert sum(e['admission_reason']=='SYNDICATED_DUPLICATE' for e in context(records)['snapshot']['events'])==1
    assert score([article(),article(5,publisher='Business Wire')])['verification_status']=='SINGLE_SOURCE'
    assert score([article(),article(publisher='REUTERS')])['verification_status']=='SINGLE_SOURCE'


def test_offering_forces_zero_quality_despite_same_morning_earnings():
    result=score([article(publisher='Business Wire'),article(title='Company announces offering')])
    assert result['admitted_event_count']==2
    assert result['best_event_type']=='OFFERING_DILUTION'
    assert result['components']['catalyst_quality']['score']==0
    assert result['components']['catalyst_verification_confidence']['score']==4


@pytest.mark.parametrize('records,status',[
 ([], 'UNVERIFIED'),([article()], 'SINGLE_SOURCE'),
 ([article(),article(publisher='MarketWatch')], 'VERIFIED_MULTI_SOURCE'),
 ([article(publisher='GlobeNewswire')], 'VERIFIED_PRIMARY'),
])
def test_verification_ladder(records,status):
    result=score(records)
    assert result['verification_status']==status
    assert result['components']['catalyst_verification_confidence']['score']==VERIFICATION_POINTS[status]


@pytest.mark.parametrize('hours,points',[(2,4),(6,3),(12,2),(24,1)])
def test_freshness_ladder_at_boundary(hours,points):
    assert score([article(hours*60)])['components']['catalyst_freshness_relevance']['score']==points
    assert score([article(hours*60+.001)])['components']['catalyst_freshness_relevance']['score']==points-1


def test_freshness_uses_most_recent_positive_relevance():
    result=score([article(360),article(20,title='Newer story',insights=[{'ticker':'TEST','sentiment':'negative'}])])
    assert result['freshest_event_age_minutes']==360
    assert result['components']['catalyst_freshness_relevance']['score']==3
    assert score([article(insights=[{'ticker':'TEST','sentiment':'negative'}])])['components']['catalyst_freshness_relevance']['score']==0


def test_no_catalyst_is_observed_zero_and_fetch_failure_is_missing():
    empty=score([])
    missing=news_scores({'ticker':'TEST','news_context':{'status':'MISSING'}},FREEZE)
    assert all(c['status']=='NO_CATALYST' and c['score']==0 for c in empty['components'].values())
    assert all(c['status']=='MISSING' and c['score'] is None for c in missing['components'].values())
    assert empty['admitted_event_count']==0
    assert missing['admitted_event_count'] is None
    assert all(c['status']=='NO_CATALYST' for c in score([article(5)])['components'].values())


def test_insights_keyword_sources_and_no_sec_filing_events():
    events=context([article(title='Company raises guidance',insights=[]),article(title='Company files 8-K',insights=[{'ticker':'TEST','sentiment':'negative'}])])['snapshot']['events']
    indexed={e['headline']:e for e in events}
    assert indexed['Company raises guidance']['sentiment_source']=='KEYWORDS'
    assert indexed['Company raises guidance']['sentiment']=='POSITIVE'
    assert indexed['Company files 8-K']['sentiment_source']=='MASSIVE_INSIGHTS'
    assert indexed['Company files 8-K']['sentiment']=='NEGATIVE'
    assert all(e['event_type']!='SEC_FILING' for e in events)


class Provider:
    provider_name='MASSIVE'
    feed_version='fake_news_v1'
    def __init__(self,records):self.records=records;self.calls=[]
    def get_news_response(self,ticker,start,end):
        self.calls.append((ticker,start,end))
        if ticker=='FAIL':raise RuntimeError('synthetic outage')
        return {'pages':[{'results':self.records.get(ticker,[]),'status':'OK'}]}


def test_cache_hit_never_fetches_and_request_change_never_overwrites(tmp_path):
    cache=NewsCache(tmp_path);provider=Provider({'TEST':[article()]})
    first,hit=cache.get(provider,'TEST','2024-03-15',FREEZE,3)
    assert not hit
    original=cache.path('TEST','2024-03-15').read_bytes()
    provider.records['TEST'].append(article(title='Later provider addition'))
    again,hit=cache.get(provider,'TEST','2024-03-15',FREEZE,3)
    assert hit and first==again and len(provider.calls)==1
    assert original==cache.path('TEST','2024-03-15').read_bytes()
    with pytest.raises(ValueError,match='immutable'):
        cache.get(provider,'TEST','2024-03-15','2024-03-15T09:00:00-04:00',3)
    assert len(provider.calls)==1


def fixture_snapshot():
    return runpy.run_path(str(ROOT/'tests/test_sector_metrics.py'))['fixture_snapshot']()


def test_only_scored_and_shadow_fetch_with_progress_and_errors(tmp_path,capsys):
    snapshot={'trading_date':'2024-03-15','freeze_timestamp':FREEZE,'securities':[]}
    for ticker,count in [('SCORABLE',30),('SHADOW',10),('SPARSE',9),('FAIL',30)]:
        snapshot['securities'].append({'ticker':ticker,'market_data':{'real_bar_count_60m':{'value':count}}})
    provider=Provider({})
    summary=enrich_snapshot_news(snapshot,provider,cache_root=tmp_path)
    assert {call[0] for call in provider.calls}=={'SCORABLE','SHADOW','FAIL'}
    assert summary=={'hits':0,'fetches':3,'errors':1,'tickers_with_zero_articles':2}
    assert 'news_context' not in snapshot['securities'][2]
    assert snapshot['securities'][3]['news_context']['status']=='MISSING'
    assert 'tickers=3/3' in capsys.readouterr().err
    assert not (tmp_path/'2024-03-15/FAIL.json').exists()


def test_coverage_reconciles_to_csv_and_rolls_up(tmp_path):
    snapshot=fixture_snapshot()
    snapshot=json.loads(json.dumps(snapshot).replace('2026-09-14','2024-03-15').replace('2026-09-','2024-03-').replace('2026-08-','2024-02-'))
    template=snapshot['securities'][0];snapshot['securities']=[]
    for ticker,price,records in [('ONE',1.5,[article(ticker='ONE',publisher='Business Wire')]),('TWO',3,[]),('THREE',7,[article(5,ticker='THREE')]),('FOUR',20,[article(ticker='FOUR')]),('FIVE',30,[])]:
        item=deepcopy(template);item['ticker']=ticker;item['market_data']['last_price']['value']=price
        item['news_context']=context(records,ticker)
        snapshot['securities'].append(item)
    scout=run_research_scout_alpha(snapshot,threshold_pct=0)
    coverage=count_news_coverage(scout['candidates'],snapshot['securities'],{'ONE','THREE'})
    paths=export_daily_outcomes(tmp_path,snapshot,scout,{'outcomes':[],'policy_comparisons':[]},{})
    with paths[0].open() as handle:
        reader=csv.DictReader(handle);rows=list(reader)
        assert reader.fieldnames==list(scorable_outcomes_columns(['execution_policy_v1.0']))
    assert len(METRIC_IDS)==25
    assert coverage['scorable_count']==len(rows)==5
    assert coverage['with_any_article']==sum(int(row['news_article_count'])>0 for row in rows)==3
    assert coverage['with_admitted_event']==sum(int(row['admitted_event_count'])>0 for row in rows)==2
    assert coverage['top_10_movers_with_admitted_event']==1
    assert coverage['selections_with_admitted_event']==sum(row['selected']=='true' and int(row['admitted_event_count'])>0 for row in rows)
    assert coverage['by_source_tier']=={'PRIMARY':1,'HIGH_QUALITY_WIRE':1}
    assert coverage['by_event_type']=={'EARNINGS':2}
    assert sum(t['scorable_count'] for t in coverage['by_price_tier'].values())==5
    doubled=rollup_news_coverage([coverage,coverage])
    assert doubled['with_admitted_event']==4
    assert doubled['by_price_tier']['under_2']['by_source_tier']=={'PRIMARY':2}


@pytest.mark.parametrize('ticker,gap,expected',[
 ('CDLX',52,'943e6dbc42b31691bd5a73beaf9a6b326c39190bcbd5c6e3b529e8fa20fdff1b'),
 ('SMR',-12,'476c1df72a388c5b74fc4498915907df177151866beea9eaaa42ba3e36e49976'),
])
def test_existing_twenty_two_march_components_byte_identical(ticker,gap,expected):
    # Main 549376c golden hashes. Synthetic March paths, not the vendor replay.
    snapshot=fixture_snapshot()
    snapshot=json.loads(json.dumps(snapshot).replace('2026-09-14','2024-03-15').replace('2026-09-','2024-03-').replace('2026-08-','2024-02-'))
    item=snapshot['securities'][0];item['ticker']=ticker
    item['market_data']['previous_close']['value']=item['premarket_bars'][-1]['close']/(1+gap/100)
    item['news_context']=context([article(ticker=ticker)],ticker)
    candidate=run_research_scout_alpha(snapshot)['candidates'][0]
    original={key:value for key,value in candidate['component_scores'].items() if key not in CATALYST_METRIC_IDS}
    assert hashlib.sha256(json.dumps(original,sort_keys=True).encode()).hexdigest()==expected
    assert candidate['reachable_metric_count']==25
    assert candidate['score_pct']==candidate['total_score']
    assert candidate['score_pct_fixed120']==candidate['total_score']/120*100


def test_massive_response_preserves_all_raw_pages():
    from trainer.providers.massive import MassiveClient
    client=MassiveClient.__new__(MassiveClient)
    pages=[{'results':[article()],'next_url':'next','request_id':'first'}, {'results':[],'request_id':'last'}]
    calls=[]
    def page(path,params=None):
        calls.append((path,params));return pages[len(calls)-1]
    client._get_page=page
    result=client.get_news_response('TEST','start','end')
    assert result=={'pages':pages}
    assert calls[0][1]['published_utc.gte']=='start'
    assert calls[1]==('next',None)


def test_postmortem_accepts_no_catalyst_and_records_coverage():
    fixtures=runpy.run_path(str(ROOT/'tests/test_benchmark_postmortem.py'))
    candidate=fixtures['_candidate']()
    news=news_scores({'ticker':'MOVE','news_context':context([], 'MOVE')},FREEZE)
    candidate.update({key:value for key,value in news.items() if key!='components'})
    candidate['status']='SCORED'
    candidate['component_scores'].update(news['components'])
    result=fixtures['_postmortem'](candidate=candidate)
    assert result['news_coverage']['scorable_count']==1
    assert result['news_coverage']['with_admitted_event']==0
    entry=result['missed_opportunities'][0]
    assert sum(c['status']=='NO_CATALYST' for c in entry['component_scores'])==3


def test_news_cache_is_in_all_three_workflows():
    for name in ['day','range','scheduled']:
        workflow=(ROOT/f'.github/workflows/flat-file-replay-{name}.yml').read_text()
        pair='path: |\n            data/reference_cache/\n            data/news_cache/'
        assert workflow.count(pair)==(2 if name=='scheduled' else 1)
