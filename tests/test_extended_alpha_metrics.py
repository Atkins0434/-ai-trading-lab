from copy import deepcopy
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import runpy

import pytest

from trainer.extended_alpha_metrics import EXTENDED_METRIC_IDS, cutler_window_rsi, extended_components
from trainer.price_volume import score_thresholds
from trainer.research_scout_alpha import _load_contract, _reversal_score, run_research_scout_alpha
from trainer.scorable_outcomes import export_daily_outcomes
from trainer.scorable_outcomes_schema import METRIC_IDS, scorable_outcomes_columns

ROOT = Path(__file__).resolve().parents[1]


def fixture_module():
    return runpy.run_path(str(ROOT / 'tests/test_research_scout_alpha.py'))


def snapshot_fixture():
    fixture = fixture_module()
    daily, bars = fixture['alpha_inputs']()
    return fixture['build_massive_alpha_snapshot']('TEST', '2026-09-14', daily, bars, exchange='NASDAQ')


def security_with_closes(closes, window=15):
    freeze = datetime.fromisoformat('2024-03-15T09:15:00-04:00')
    # Irregularly spaced real bars exercise no-padding behavior.
    spacing = max(1, window // len(closes))
    bars = [{'timestamp': (freeze - timedelta(minutes=window - index * spacing)).isoformat(),
             'open': close, 'high': close, 'low': close, 'close': close, 'volume': 100}
            for index, close in enumerate(closes)]
    return {'premarket_bars': bars, 'market_data': {}}, freeze.isoformat()


def test_cutler_hand_calculated_ten_bars():
    # Nine changes: +2,-1,0,+3,-2,+1,0,-1,+2. Gains=8, losses=4.
    assert cutler_window_rsi([10,12,11,11,14,12,13,13,12,14]) == pytest.approx(100 - 100 / 3)


def test_rsi_window_excludes_old_and_freeze_bars():
    config, _, _ = _load_contract()
    security, freeze = security_with_closes([10,11,12,13,14])
    original = extended_components(security, config, freeze)['relative_strength_index_15m']
    before = dict(security['premarket_bars'][0], timestamp='2024-03-15T08:59:00-04:00', close=1000)
    after = dict(security['premarket_bars'][-1], timestamp=freeze, close=1)
    security['premarket_bars'] = [before, *security['premarket_bars'], after]
    assert extended_components(security, config, freeze)['relative_strength_index_15m'] == original


@pytest.mark.parametrize('closes,expected', [([1,2,3],100), ([3,2,1],0), ([2,2,2],50)])
def test_cutler_edge_cases(closes, expected):
    assert cutler_window_rsi(closes) == expected


@pytest.mark.parametrize('window,minimum', [(15,5),(30,10),(60,20)])
def test_rsi_exact_minimum_and_one_fewer(window, minimum):
    config, _, _ = _load_contract()
    metric = f'relative_strength_index_{window}m'
    for count in (minimum - 1, minimum):
        security, freeze = security_with_closes(list(range(10, 10 + count)), window)
        component = extended_components(security, config, freeze)[metric]
        assert component['calculation_version'] == 'rsi_cutler_window_v1.0'
        assert component['status'] == ('SCORED' if count == minimum else 'MISSING')
        assert component['score'] == (4 if count == minimum else None)
        assert component['raw_value'] == (100 if count == minimum else None)


@pytest.mark.parametrize('metric', EXTENDED_METRIC_IDS)
def test_threshold_ladders(metric):
    config, _, _ = _load_contract()
    thresholds = config['extended_metric_thresholds'][metric]
    for points in range(1, 5):
        boundary = thresholds[str(points)]
        assert score_thresholds(boundary, thresholds) == points
        assert score_thresholds(boundary - 0.000001, thresholds) == points - 1


def test_missing_atr_keeps_fixed_denominator_and_alpha12_bytes():
    snapshot = snapshot_fixture()
    snapshot['securities'][0]['market_data'].pop('atr_14_usd', None)
    candidate = run_research_scout_alpha(snapshot)['candidates'][0]
    component = candidate['component_scores']['atr_pct_opportunity']
    assert component['status'] == 'MISSING'
    assert component['score'] is None
    assert candidate['maximum_possible_score'] == 100
    assert candidate['score_pct'] == candidate['total_score'] / 100 * 100
    # Captured from main at 477f59f before implementing the new metrics.
    score_bytes = json.dumps({'total_score': candidate['alpha12_total_score'], 'score_pct': candidate['alpha12_score_pct']}, sort_keys=True).encode()
    assert score_bytes == b'{"score_pct": 27.083333333333332, "total_score": 13}'
    original = {key: value for key, value in candidate['component_scores'].items() if key in METRIC_IDS[:12]}
    assert hashlib.sha256(json.dumps(original, sort_keys=True).encode()).hexdigest() == '350668260385005ac1248c5157de0cc95f7db24e9045dcfad732a2e69bdf6b57'


def test_smr_reversal_output_bytes_unchanged():
    fixture = fixture_module()
    components = fixture['_reversal_components'](
        premarket_gap_strength=-12.0, price_vs_premarket_vwap={'vwap':9.0, 'price_vs_vwap_pct':-3.5},
        price_slope_60m=-0.05, premarket_trend_consistency=35.0,
        price_slope_15m=0.40, price_slope_30m=0.094, relative_volume=3.56,
    )
    result = _reversal_score(components, _load_contract()[2])
    assert hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest() == 'c531b86a0f4e164c7b6948ecb7f0cd6dce187e3110dc0032d547eff409329b65'


def test_scalar_metrics_and_absolute_range_without_expansion_history():
    config, _, _ = _load_contract()
    security, freeze = security_with_closes([100,101,102,103,110])
    security['market_data'] = {key: {'value':value, 'as_of_timestamp':freeze} for key,value in {
        'previous_close':100.0, 'atr_14_usd':7.0,
        'average_daily_dollar_volume':25_000_000.0, 'premarket_dollar_volume':1_000_000.0,
    }.items()}
    results = extended_components(security, config, freeze)
    for key, raw in [('atr_pct_opportunity',7), ('current_range_pct_opportunity',10),
                     ('average_daily_dollar_volume_quality',25_000_000), ('premarket_dollar_volume_quality',1_000_000)]:
        assert results[key]['raw_value'] == pytest.approx(raw)
        assert results[key]['score'] == 3


def test_cdlx_synthetic_march15_exports_both_scores(tmp_path):
    """Synthetic frozen path; not a claim of reproducing the original replay."""
    snapshot = snapshot_fixture()
    # Shift all snapshot timestamps consistently to the requested fixture date.
    snapshot = json.loads(json.dumps(snapshot).replace('2026-09-14', '2024-03-15').replace('2026-09-', '2024-03-').replace('2026-08-', '2024-02-'))
    security = snapshot['securities'][0]
    security['ticker'] = 'CDLX'
    # CDLX's known +52% gap, while retaining a synthetic chronological bar path.
    security['market_data']['previous_close']['value'] = security['premarket_bars'][-1]['close'] / 1.52
    result = run_research_scout_alpha(snapshot)
    candidate = result['candidates'][0]
    assert candidate['rubric_version'] == 'alpha_v1.3_25m'
    assert candidate['alpha12_total_score'] is not None
    assert candidate['score_pct'] == candidate['total_score'] / 100 * 100
    assert candidate['alpha12_score_pct'] == candidate['alpha12_total_score'] / 48 * 100
    assert len(candidate['reversal_component_scores']) == 12
    paths = export_daily_outcomes(tmp_path, snapshot, result, {'outcomes': [], 'policy_comparisons': []}, {})
    import csv
    with paths[0].open() as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == list(scorable_outcomes_columns(['execution_policy_v1.0']))
        assert len([key for key in reader.fieldnames if key.endswith('_raw')]) == 25
        row = next(reader)
    assert row['ticker'] == 'CDLX'
    assert row['rubric_version'] == 'alpha_v1.3_25m'
    assert row['alpha12_total_score'] == str(candidate['alpha12_total_score'])
    assert row['alpha12_score_pct'] == f"{candidate['alpha12_score_pct']:.6f}"


def test_registry_numbers_and_research_only_thresholds():
    config, registry, reversal = _load_contract()
    primary = json.loads((ROOT / 'config/feature_registry_v1.json').read_text())
    added = [m for m in primary['metrics'] if m['id'] in EXTENDED_METRIC_IDS]
    assert [m['number'] for m in added] == [13,14,15,25,26,29,30]
    assert all(m['implemented'] and m['research_thresholds'] for m in added)
    assert len(METRIC_IDS) == 25
    assert registry['maximum_points'] == 100
    assert config['selection_threshold_pct'] == 70
    assert reversal['scoring']['maximum_points'] == 48


def test_postmortem_components_include_all_twenty_five():
    from trainer.postmortem import _component_scores
    candidate = run_research_scout_alpha(snapshot_fixture())['candidates'][0]
    exported = _component_scores(candidate)
    assert {item['metric_id'] for item in exported} == set(METRIC_IDS)
    assert len(exported) == 25
    assert next(item for item in exported if item['metric_id'] == 'relative_strength_index_60m')['calculation_version'] == 'rsi_cutler_window_v1.0'
