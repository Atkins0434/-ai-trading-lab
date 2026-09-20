from copy import deepcopy
from datetime import datetime, timedelta, timezone
import csv
import hashlib
import json
from pathlib import Path
import runpy

import pytest

from trainer.research_scout_alpha import _load_contract, run_research_scout_alpha
from trainer.news_alpha import CATALYST_METRIC_IDS
from trainer.sector_metrics import (
    SECTOR_METRIC_IDS, availability_metadata, benchmark_symbol,
    build_sector_context, premarket_return, sector_components, sector_key,
)
from trainer.scorable_outcomes import export_daily_outcomes
from trainer.scorable_outcomes_schema import METRIC_IDS, scorable_outcomes_columns

ROOT = Path(__file__).resolve().parents[1]
FREEZE = '2024-03-15T09:15:00-04:00'


def security(ticker, sic='7372', return_pct=4.0, count=30, eligible=True):
    freeze = datetime.fromisoformat(FREEZE)
    bars = [{'timestamp': (freeze - timedelta(minutes=count-i)).isoformat(),
             'open':100, 'close':100+return_pct, 'high':100+return_pct,
             'low':100, 'volume':100, 'source':'MASSIVE_FLATFILES', 'session':'premarket'} for i in range(count)]
    return {'ticker':ticker, 'sic_code':sic, 'eligible':eligible, 'exchange':'XNAS',
            'market_data':{'previous_close':{'value':100}}, 'premarket_bars':bars}


def fixture_snapshot():
    return runpy.run_path(str(ROOT / 'tests/test_extended_alpha_metrics.py'))['snapshot_fixture']()


def sector_snapshot():
    return {'freeze_timestamp':FREEZE,
            'securities':[security(f'S{i}',return_pct=4+i) for i in range(5)] + [security(f'E{i}','1311') for i in range(4)],
            'market_benchmarks':{s:security(s,return_pct=1) for s in ['SPY','QQQ','IWM']}}


def test_sector_median_eligible_population_and_no_double_count():
    config, _, _ = _load_contract()
    snapshot = sector_snapshot()
    snapshot['securities'] += [security('INELIGIBLE',return_pct=100,eligible=False),security('SPARSE',return_pct=100,count=9)]
    context = build_sector_context(snapshot,config)
    assert context['counts'] == {'SOFTWARE':5,'ENERGY':4}
    assert context['medians']['SOFTWARE'] == pytest.approx(6)
    candidate=security('LEADER',return_pct=16)
    components=sector_components(candidate,config,FREEZE,context)
    assert components['sector_strength']['raw_value'] == pytest.approx(5)
    assert components['stock_leadership_vs_sector']['raw_value'] == pytest.approx(10)
    assert components['sector_strength']['score'] == 4
    assert components['stock_leadership_vs_sector']['score'] == 3
    assert sum(components[k]['raw_value'] for k in SECTOR_METRIC_IDS[:2]) == pytest.approx(15)
    # Fewer than five names prevents both sector metrics, but not regime.
    missing=sector_components(security('ENERGY','1311'),config,FREEZE,context)
    assert all(missing[k]['status']=='MISSING' for k in SECTOR_METRIC_IDS[:2])
    assert missing['broad_market_regime_alignment']['status']=='SCORED'
    assert all(sector_components(security('NO_SIC',None),config,FREEZE,context)[k]['status']=='MISSING' for k in SECTOR_METRIC_IDS[:2])


@pytest.mark.parametrize('cap,venue,expected',[(10e9,'XNAS','SPY'),(10e9,'XNYS','SPY'),(3e9,'XNAS','QQQ'),(3e9,'XNYS','IWM'),(3e9-1,'XNAS','IWM'),(None,'XNAS','IWM')])
def test_benchmark_mapping(cap,venue,expected):
    assert benchmark_symbol({'market_cap_usd':{'value':cap},'exchange':venue},_load_contract()[0])==expected


def test_regime_negative_and_threshold_boundaries():
    from trainer.price_volume import score_thresholds
    config,_,_=_load_contract()
    for metric in SECTOR_METRIC_IDS:
        ladder=config['extended_metric_thresholds'][metric]
        for score in range(1,5):
            assert score_thresholds(ladder[str(score)],ladder)==score
            assert score_thresholds(ladder[str(score)]-1e-6,ladder)==score-1
    snapshot=sector_snapshot()
    snapshot['market_benchmarks']['IWM']=security('IWM',return_pct=-1)
    result=sector_components(snapshot['securities'][0],config,FREEZE,build_sector_context(snapshot,config))
    assert result['broad_market_regime_alignment']['score']==0


def test_real_hour_coverage_and_freeze_boundary():
    item=security('SPY',count=30)
    expected=premarket_return(item,FREEZE,30)
    item['premarket_bars'] += [dict(item['premarket_bars'][0],timestamp='2024-03-15T08:14:00-04:00',close=999),dict(item['premarket_bars'][0],timestamp=FREEZE,close=999)]
    assert premarket_return(item,FREEZE,30)==expected
    item['premarket_bars'].pop(0)
    assert premarket_return(item,FREEZE,30) is None
    duplicate=security('SPY'); duplicate['premarket_bars'].append(duplicate['premarket_bars'][0])
    assert premarket_return(duplicate,FREEZE,30) is None


def test_availability_denominator_and_exports(tmp_path):
    snapshot=fixture_snapshot()
    scout=run_research_scout_alpha(snapshot)
    candidate=scout['candidates'][0]
    assert candidate['reachable_metric_count']==25
    assert candidate['score_pct']==candidate['score_pct_reachable']==candidate['total_score']/100*100
    assert candidate['score_pct_fixed120']==candidate['total_score']/120*100
    assert candidate['maximum_possible_score']==100
    assert len(candidate['unavailable_metrics'])==5
    assert len(METRIC_IDS)==25
    paths=export_daily_outcomes(tmp_path,snapshot,scout,{'outcomes':[],'policy_comparisons':[]},{})
    with paths[0].open() as handle:
        reader=csv.DictReader(handle)
        assert reader.fieldnames==list(scorable_outcomes_columns(['execution_policy_v1.0']))
        row=next(reader)
    assert row['score_pct_fixed120']==f"{candidate['score_pct_fixed120']:.6f}"
    assert row['reachable_metric_count']=='25'
    assert row['unavailable_metrics']=='|'.join(candidate['unavailable_metrics'])
    assert row['benchmark_symbol']=='IWM'
    assert all(row[f'{metric}_raw']=='' for metric in SECTOR_METRIC_IDS)


def test_two_unavailable_metrics_are_excluded_from_denominator():
    from trainer.sector_metrics import score_percentages
    registry={'metrics':[{'id':str(i),'availability':status} for i,status in enumerate(['AVAILABLE','DATA_DEPENDENT','UNAVAILABLE','UNAVAILABLE'])]}
    meta=availability_metadata(registry)
    # One available metric earns 4; the data-dependent metric is missing.
    assert meta=={'reachable_metric_count':2,'unavailable_metrics':['2','3']}
    assert score_percentages(4,meta['reachable_metric_count'])=={'score_pct':50,'score_pct_reachable':50,'score_pct_fixed120':4/120*100}


@pytest.mark.parametrize('ticker,gap,expected,alpha12',[
 ('CDLX',52,'600479630b55a901aa4d40733288d766db601d7c69fd8d82ff86c6278b3e88b3',16),
 ('SMR',-12,'7dad5c446989b77a6bb05b7d56496f6a692aab5294af07fbafd54bb4832abcc4',12),
])
def test_march_synthetic_existing_nineteen_components_byte_identical(ticker,gap,expected,alpha12):
    # Hashes captured on main e50b1dc. These are synthetic March fixtures,
    # not a claim to reproduce the original vendor replay.
    snapshot=fixture_snapshot()
    snapshot=json.loads(json.dumps(snapshot).replace('2026-09-14','2024-03-15').replace('2026-09-','2024-03-').replace('2026-08-','2024-02-'))
    item=snapshot['securities'][0]; item['ticker']=ticker
    item['market_data']['previous_close']['value']=item['premarket_bars'][-1]['close']/(1+gap/100)
    candidate=run_research_scout_alpha(snapshot)['candidates'][0]
    components={k:v for k,v in candidate['component_scores'].items() if k not in SECTOR_METRIC_IDS + CATALYST_METRIC_IDS}
    assert hashlib.sha256(json.dumps(components,sort_keys=True).encode()).hexdigest()==expected
    assert candidate['alpha12_total_score']==alpha12
    assert len(candidate['reversal_component_scores'])==12


def test_sic_preserved_in_cache_and_manifest_without_refetch(tmp_path):
    fixtures=runpy.run_path(str(ROOT/'tests/test_universe_builder.py'))
    client=fixtures['FakeReferenceClient']()
    original=client.get_ticker_overview
    client.get_ticker_overview=lambda *args:{**original(*args),'sic_code':'7372','sic_description':'SERVICES-PREPACKAGED SOFTWARE'}
    def build(root):
        return fixtures['build_point_in_time_universe'](client,fixtures['FakeFlatFiles'](),'2018-01-03',root/'universe.json',reference_cache_root=root/'cache')
    first=build(tmp_path)
    assert all(s['sic_code']=='7372' for s in first['securities'])
    assert first['reference_cache_summary']['sic_code_coverage']=={'with':len(first['securities']),'without':0}
    cached=next((tmp_path/'cache').rglob('*.json'))
    assert json.loads(cached.read_text())['overview']['sic_description']=='SERVICES-PREPACKAGED SOFTWARE'
    client.get_ticker_overview=lambda *args:pytest.fail('Must not refetch cached overview')
    assert build(tmp_path)['securities']==first['securities']
    # Legacy cache without SIC stays usable, with missing sector information.
    client.get_ticker_overview=original
    legacy=build(tmp_path/'legacy')
    client.get_ticker_overview=lambda *args:pytest.fail('Must not refetch for SIC enrichment')
    resumed=build(tmp_path/'legacy')
    assert resumed['reference_cache_summary']['sic_code_coverage']=={'with':0,'without':len(legacy['securities'])}


def test_benchmark_bars_loaded_in_existing_file_scans(tmp_path):
    f=runpy.run_path(str(ROOT/'tests/test_flatfile_snapshot.py'))
    store=f['MassiveFlatFileStore'](object(),cache_root=tmp_path)
    prior='2024-03-14'; day='2024-03-15'
    row=f['_row']; write=f['_write_cached_csv']
    symbols=['AAL','SPY','QQQ','IWM']
    write(store.cache_path(f['DAY_AGGS_DATASET'],prior),[row(s,datetime(2024,3,14,tzinfo=timezone.utc),close=100,volume=1_000_000) for s in symbols])
    write(store.cache_path(f['MINUTE_AGGS_DATASET'],prior),[])
    freeze=datetime.fromisoformat(FREEZE)
    write(store.cache_path(f['MINUTE_AGGS_DATASET'],day),[row(s,freeze-timedelta(minutes=i),close=101,volume=1000) for s in symbols for i in range(1,31)])
    calls=[]; original=store.iter_bars
    def read(*args,**kwargs):
        calls.append(args)
        yield from original(*args,**kwargs)
    store.iter_bars=read
    result=f['build_flatfile_snapshot'](day,f['_manifest'](day),store,lookback_sessions=1)
    assert len(calls)==3
    assert [s['ticker'] for s in result.snapshot['securities']]==['AAL']
    assert set(result.outcome_bars)=={'AAL'}
    context=build_sector_context(result.snapshot,_load_contract()[0])
    assert context['benchmark_returns']==pytest.approx({'SPY':1,'QQQ':1,'IWM':1})


def test_scorer_records_market_context_alongside_unchanged_components():
    snapshot=fixture_snapshot()
    original=deepcopy(snapshot['securities'][0])
    # Five identical eligible peers suffice; the sector receives all of them,
    # regardless of whether their primary scores clear selection threshold.
    snapshot['securities']=[]
    for i in range(5):
        item=deepcopy(original);item['ticker']=f'PEER{i}';item['sic_code']='7372'
        snapshot['securities'].append(item)
    benchmark={'premarket_bars':deepcopy(original['premarket_bars']),
               'market_data':{'previous_close':deepcopy(original['market_data']['previous_close'])}}
    for bar in benchmark['premarket_bars']:
        for field in ['open','close','high','low']:bar[field]=benchmark['market_data']['previous_close']['value']
    snapshot['market_benchmarks']={s:deepcopy(benchmark) for s in ['SPY','QQQ','IWM']}
    scored=run_research_scout_alpha(snapshot)
    assert scored['spy_premarket_return_pct']==0
    for candidate in scored['candidates']:
        assert all(candidate['component_scores'][m]['status']=='SCORED' for m in SECTOR_METRIC_IDS)
        assert candidate['component_scores']['broad_market_regime_alignment']['score']==1
        assert candidate['score_pct']==candidate['total_score']/100*100
        assert candidate['threshold_points']==70
    baseline=run_research_scout_alpha(dict(snapshot,market_benchmarks={}))
    for before,after in zip(baseline['candidates'],scored['candidates']):
        assert {k:v for k,v in before['component_scores'].items() if k not in SECTOR_METRIC_IDS + CATALYST_METRIC_IDS}=={k:v for k,v in after['component_scores'].items() if k not in SECTOR_METRIC_IDS + CATALYST_METRIC_IDS}
        assert before['reversal_component_scores']==after['reversal_component_scores']


def test_missing_or_ambiguous_prior_benchmark_close():
    from trainer.flatfile_snapshot import _benchmark_prior_close
    assert _benchmark_prior_close([], '2024-03-14') is None
    bar={'trading_date':'2024-03-14','close':100}
    assert _benchmark_prior_close([bar,bar], '2024-03-14') is None
    item=security('SPY')
    item['market_data']['previous_close']['as_of_timestamp']=FREEZE
    assert premarket_return(item,FREEZE,30) is None
