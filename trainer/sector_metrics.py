"""Frozen, research-only sector and market context; no provider calls."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timedelta
import json
from pathlib import Path
from statistics import median

from trainer.price_volume import score_thresholds

ROOT = Path(__file__).resolve().parents[1]
SECTOR_METRIC_IDS = ('sector_strength', 'stock_leadership_vs_sector', 'broad_market_regime_alignment')
BENCHMARK_SYMBOLS = ('SPY', 'QQQ', 'IWM')


def availability_metadata(registry=None):
    if registry is None:
        registry = json.loads((ROOT / 'config/feature_registry_alpha_v1.json').read_text())
    return {
        'reachable_metric_count': sum(m['availability'] != 'UNAVAILABLE' for m in registry['metrics']),
        'unavailable_metrics': [m['id'] for m in registry['metrics'] if m['availability'] == 'UNAVAILABLE'],
    }


def score_percentages(points, reachable_metric_count):
    reachable = points / (4 * reachable_metric_count) * 100
    return {"score_pct": reachable, "score_pct_reachable": reachable, "score_pct_fixed120": points / 120 * 100}


def sector_key(sic_code, mapping=None):
    if sic_code is None:
        return None
    code = str(sic_code).strip()
    if not code.isdigit() or not 2 <= len(code) <= 4:
        return None
    # Numeric SIC codes may omit their leading zero.
    major = code.zfill(4)[:2] if len(code) > 2 else code
    mapping = mapping or json.loads((ROOT / 'config/sector_map.json').read_text())
    return mapping['major_groups'].get(major, mapping['default'])


def benchmark_symbol(security, config):
    cap = security.get('market_cap_usd')
    if isinstance(cap, dict):
        cap = cap.get('value')
    venue = security.get('listing_venue', security.get('exchange'))
    for rule in config['sector_context']['mapping_order']:
        if 'minimum_market_cap_usd' in rule and (cap is None or cap < rule['minimum_market_cap_usd']):
            continue
        if 'listing_venue' in rule and venue != rule['listing_venue']:
            continue
        return rule['symbol']
    raise ValueError('Benchmark mapping requires a default')


def premarket_return(security, freeze_timestamp, minimum_bars):
    freeze = datetime.fromisoformat(freeze_timestamp.replace('Z', '+00:00'))
    start = freeze - timedelta(minutes=60)
    bars = [b for b in security.get('premarket_bars', [])
            if start <= datetime.fromisoformat(b['timestamp'].replace('Z', '+00:00')) < freeze]
    # Ambiguous duplicate minutes cannot inflate coverage or choose a close.
    counts = Counter(datetime.fromisoformat(b['timestamp'].replace('Z', '+00:00')).replace(second=0, microsecond=0) for b in bars)
    if any(n > 1 for n in counts.values()) or len(bars) < minimum_bars:
        return None
    observation = security.get('market_data', {}).get('previous_close', {})
    prior_at = observation.get('as_of_timestamp')
    if prior_at and datetime.fromisoformat(prior_at.replace('Z', '+00:00')) >= freeze:
        return None
    prior = observation.get('value')
    if prior is None or prior <= 0:
        return None
    last = max(bars, key=lambda b: datetime.fromisoformat(b['timestamp'].replace('Z', '+00:00')))
    return (float(last['close']) - prior) / prior * 100


def build_sector_context(snapshot, config):
    freeze = snapshot['freeze_timestamp']
    rules = config['sector_context']
    mapping = json.loads((ROOT / 'config/sector_map.json').read_text())
    sectors = defaultdict(list)
    for security in snapshot['securities']:
        sector = sector_key(security.get('sic_code'), mapping)
        value = premarket_return(security, freeze, rules['minimum_real_bars_60m'])
        if security['eligible'] and sector is not None and value is not None:
            sectors[sector].append(value)
    return {
        'sector_map': mapping,
        'medians': {key: median(values) for key, values in sectors.items() if len(values) >= rules['minimum_sector_names']},
        'counts': {key: len(values) for key, values in sectors.items()},
        'benchmark_returns': {symbol: premarket_return(snapshot.get('market_benchmarks', {}).get(symbol, {}), freeze, rules['benchmark_minimum_real_bars_60m']) for symbol in BENCHMARK_SYMBOLS},
    }


def sector_components(security, config, freeze_timestamp, context):
    sector = sector_key(security.get('sic_code'), context['sector_map'])
    sector_median = context['medians'].get(sector)
    benchmark = context['benchmark_returns'].get(benchmark_symbol(security, config))
    stock_return = premarket_return(security, freeze_timestamp, config['sector_context']['minimum_real_bars_60m'])
    available = sector_median is not None and benchmark is not None
    values = {
        'sector_strength': sector_median - benchmark if available else None,
        'stock_leadership_vs_sector': stock_return - sector_median if available and stock_return is not None else None,
        'broad_market_regime_alignment': benchmark,
    }
    return {key: {
        'raw_value': value,
        'score': None if value is None else score_thresholds(value, config['extended_metric_thresholds'][key]),
        'status': 'MISSING' if value is None else 'SCORED',
        'maximum_score': 4,
        'as_of_timestamp': None if value is None else freeze_timestamp,
        'reason_code': 'REQUIRED_SECTOR_INPUT_MISSING' if value is None else f'{key.upper()}_SCORE',
        'calculation_version': 'sector_context_v1.0',
    } for key, value in values.items()}
