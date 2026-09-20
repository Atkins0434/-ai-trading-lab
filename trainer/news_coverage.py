"""Coverage counts over scored (including shadow) rows, reconciled with exports."""
from collections import Counter

PRICE_TIERS = ('under_2', '2_to_5', '5_to_10', '10_to_25', '25_plus', 'UNKNOWN')


def price_tier(price):
    if price is None:
        return 'UNKNOWN'
    for ceiling, name in ((2, 'under_2'), (5, '2_to_5'), (10, '5_to_10'), (25, '10_to_25')):
        if price < ceiling:
            return name
    return '25_plus'


def empty_counts():
    return {'scorable_count': 0, 'with_any_article': 0, 'with_admitted_event': 0,
            'top_10_movers_with_admitted_event': 0, 'selections_with_admitted_event': 0,
            'fetch_errors': 0, 'by_source_tier': {}, 'by_event_type': {}}


def count_news_coverage(candidates, securities, top_movers):
    result = empty_counts()
    result['by_price_tier'] = {tier: empty_counts() for tier in PRICE_TIERS}
    sources = {security['ticker']: security for security in securities}
    for candidate in candidates:
        if candidate.get('status') not in {'SCORED', 'SHADOW_SCORED', 'SELECTED'}:
            continue
        item = sources.get(candidate['ticker'], {})
        price = item.get('market_data', {}).get('last_price', {}).get('value')
        for target in (result, result['by_price_tier'][price_tier(price)]):
            target['scorable_count'] += 1
            target['with_any_article'] += (candidate.get('news_article_count') or 0) > 0
            target['fetch_errors'] += candidate.get('news_fetch_status') == 'MISSING'
            if (candidate.get('admitted_event_count') or 0) == 0:
                continue
            target['with_admitted_event'] += 1
            target['top_10_movers_with_admitted_event'] += candidate['ticker'] in top_movers
            target['selections_with_admitted_event'] += bool(candidate.get('research_selected') or candidate.get('qualification_selected'))
            # One best event per scored ticker, matching the flat CSV columns.
            for field, key in (('source_tier', 'by_source_tier'), ('best_event_type', 'by_event_type')):
                label = candidate[field]
                target[key][label] = target[key].get(label, 0) + 1
    return result


def rollup_news_coverage(coverages):
    result = empty_counts()
    result['by_price_tier'] = {tier: empty_counts() for tier in PRICE_TIERS}
    def add(target, source):
        for key in empty_counts():
            if key.startswith('by_'):
                counts = Counter(target[key]); counts.update(source.get(key, {}))
                target[key] = dict(sorted(counts.items()))
            else:
                target[key] += source.get(key, 0)
    for coverage in coverages:
        add(result, coverage)
        for tier in PRICE_TIERS:
            add(result['by_price_tier'][tier], coverage.get('by_price_tier', {}).get(tier, {}))
    return result
