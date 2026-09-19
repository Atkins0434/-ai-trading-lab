import csv
from copy import deepcopy
import json
from pathlib import Path
import runpy

import pytest

from trainer.execution_costs import cost_fields, load_execution_costs, price_tier
from trainer.trade_engine import simulate_trade, load_execution_policy
from trainer.benchmark import _return_baselines, build_same_universe_benchmark, load_verdict_basis, BenchmarkError
from trainer.outcome_grader import grade_replay_outcomes
from trainer.scorable_outcomes import export_daily_outcomes
from trainer.scorable_outcomes import concatenate_completed_outcomes
from trainer.scorable_outcomes_schema import scorable_outcomes_columns, eligible_outcomes_columns

ROOT = Path(__file__).resolve().parents[1]


def _fill(price, exit_at="2024-03-15T09:30:00-04:00", reason="TRAILING_STOP", exit_price=None):
    exit_price = price * 1.01 if exit_price is None else exit_price
    return dict(trade_executed=True, entry_price=price, exit_price=exit_price,
                position_size_shares=100, position_value_usd=price * 100,
                exit_timestamp=exit_at, exit_reason=reason,
                realized_pnl_usd=(exit_price-price)*100,
                realized_return_pct=(exit_price-price)/price*100)


@pytest.mark.parametrize("price,entry_bps,early,late,end", [
    (1,150,150,75,50), (2.89,100,100,50,35), (7,60,60,30,20),
    (15,35,35,18,12), (30,20,20,10,8),
])
@pytest.mark.parametrize("phase", ["early", "late", "end"])
def test_hand_computed_costs_each_price_and_time_tier(price,entry_bps,early,late,end,phase):
    timestamp = {"early":"09:34:00", "late":"09:35:00", "end":"15:59:00"}[phase]
    trade = _fill(price, f"2024-03-15T{timestamp}-04:00", "SESSION_END" if phase=="end" else "TRAILING_STOP")
    config = load_execution_costs()
    config.update(commission_per_share_usd=.003, commission_minimum_per_order_usd=1)
    result = cost_fields(trade, 18, config)
    entry = price*100*entry_bps/10000
    exit_cost = price*1.01*100*{"early":early,"late":late,"end":end}[phase]/10000
    fees = 100*.000166
    total = entry+exit_cost+2+fees
    assert result["entry_slippage_usd"] == pytest.approx(entry)
    assert result["exit_slippage_usd"] == pytest.approx(exit_cost)
    assert result["commissions_usd"] == 2
    assert result["regulatory_fees_usd"] == pytest.approx(fees)
    assert result["total_cost_usd"] == pytest.approx(total)
    assert result["cost_bps_of_position"] == pytest.approx(total/(price*100)*10000)
    assert result["net_realized_pnl_usd"] == pytest.approx(trade["realized_pnl_usd"]-total)
    assert result["net_realized_return_pct"] == pytest.approx(1-total/(price*100)*100)
    assert result["net_capture_ratio"] == pytest.approx(result["net_realized_return_pct"]/18)


def test_positive_one_percent_gross_fill_at_289_nets_negative():
    # A supplied gross fill is costed without changing the stop/target logic.
    trade = _fill(2.89)
    result = cost_fields(trade, 18)
    assert result["gross_realized_return_pct"] == pytest.approx(1)
    assert result["net_realized_return_pct"] == pytest.approx(-1.01574394463667)
    assert result["entry_slippage_usd"] == pytest.approx(2.89)
    assert result["exit_slippage_usd"] == pytest.approx(2.9189)


@pytest.mark.parametrize("price,tier", [(2,"2_to_5"),(5,"5_to_10"),(10,"10_to_25"),(25,"25_plus")])
def test_fill_price_boundaries(price,tier):
    assert price_tier(price) == tier


def test_each_side_uses_own_fill_tier_and_timezone():
    result = cost_fields(_fill(4.99, "2024-03-15T13:35:00Z", exit_price=5.01))
    assert result["entry_slippage_usd"] == pytest.approx(4.99)
    assert result["exit_slippage_usd"] == pytest.approx(1.503)


def test_per_share_commission_above_minimum_and_no_trade_costs():
    config = load_execution_costs()
    config.update(commission_per_share_usd=.02, commission_minimum_per_order_usd=1)
    trade = _fill(10)
    assert cost_fields(trade, config=config)["commissions_usd"] == 4
    rejected = dict(trade, trade_executed=False, position_value_usd=0, realized_pnl_usd=0, realized_return_pct=0)
    result = cost_fields(rejected, config=config)
    assert result["total_cost_usd"] == result["net_realized_pnl_usd"] == 0
    assert result["net_capture_ratio"] is None


def test_gross_fields_byte_identical_to_main_percent_and_atr():
    fixture = json.loads((ROOT/"tests/fixtures/execution_costs_gross.json").read_text())
    assert len(fixture["results"]) == 2
    for path, expected in fixture["results"].items():
        result = simulate_trade("CDLX",12.5,fixture["bars"],10000,
                                policy=load_execution_policy(path),atr_14_usd=.25)
        gross = {key:result[key] for key in expected}
        assert json.dumps(gross,sort_keys=True).encode() == json.dumps(expected,sort_keys=True).encode()
        assert result["gross_realized_return_pct"] == result["realized_return_pct"]
        assert result["net_realized_return_pct"] < result["realized_return_pct"]


def test_random_and_eligible_baselines_cost_the_identical_draws():
    bars=[dict(timestamp="2024-03-15T09:30:00-04:00",open=10,high=10.1,low=10,close=10.1,volume=1000)]
    outcomes=[dict(ticker=t,intraday_path=bars,maximum_capturable_move_pct=1) for t in ("A","B","C")]
    result = _return_baselines(outcomes,2,10000)
    # 200 shares per position, 35 entry bps + 12 session-end exit bps + fees.
    cost = 200*10*.0035 + 200*10.1*.0012 + 200*.000166
    assert result["random_draw_mean_realized_pnl_usd"]-result["net_random_draw_mean_realized_pnl_usd"] == pytest.approx(2*cost)
    assert result["eligible_basket_expected_pnl_usd"]-result["net_eligible_basket_expected_pnl_usd"] == pytest.approx(2*cost)
    assert result["random_draw_mean_realized_return_pct"]-result["net_random_draw_mean_realized_return_pct"] == pytest.approx(2*cost/10000*100)


def test_verdict_basis_round_trip(tmp_path):
    assert load_verdict_basis() == "GROSS"
    path=tmp_path/"benchmark.json"
    for basis in ("GROSS","NET"):
        path.write_text(json.dumps({"verdict_basis":basis}))
        assert load_verdict_basis(path) == basis
    path.write_text('{"verdict_basis":"OTHER"}')
    with pytest.raises(BenchmarkError,match="verdict_basis"):
        load_verdict_basis(path)


def test_cohorts_and_comparison_baselines_include_costs():
    helper=runpy.run_path(str(ROOT/"tests/test_execution_policy_comparison.py"))
    snapshot=helper["_snapshot"]()
    snapshot["universe_version"]="research_universe_v1.0"
    base=snapshot["securities"][0]
    snapshot["securities"]=[dict(deepcopy(base),ticker=t,eligible=True) for t in ("A","B","C")]
    scout={k:snapshot[k] for k in ("universe_mode","universe_manifest_hash","universe_coverage","research_evidence","promotion_eligible")}
    scout["candidates"]=[dict(ticker=t,research_selected=t!="C",selection_basis=basis,rank=i+1,score_pct=80) for i,(t,basis) in enumerate((("A","QUALIFYING_THRESHOLD"),("B","EXPLORATION_TOP_K"),("C","NOT_SELECTED")))]
    outcome=grade_replay_outcomes(snapshot,scout,{t:helper["_bars"]() for t in ("A","B","C")},2500,[ROOT/"config/execution_policy_atr.json"])
    benchmark=build_same_universe_benchmark(snapshot,scout,outcome,2500)
    for name in ("scout_summary","exploration_summary","combined_summary","benchmark_summary"):
        assert benchmark[name]["net_realized_return_pct"] < benchmark[name]["realized_return_pct"]
    comparison=outcome["policy_comparisons"][0]
    assert comparison["summary"]["net_realized_return_pct"] < comparison["summary"]["realized_return_pct"]
    assert comparison["return_baselines"]["net_random_draw_mean_realized_return_pct"] < comparison["return_baselines"]["random_draw_mean_realized_return_pct"]
    assert benchmark["comparison"]["result_code"] == benchmark["comparison"]["gross_result_code"]
    assert benchmark["comparison"]["net_result_code"] is not None
    from trainer.postmortem import build_postmortem
    scout["scout_version"] = snapshot["scout_version"]
    for candidate in scout["candidates"]:
        candidate["component_scores"] = {}
    postmortem = build_postmortem(snapshot,scout,benchmark,outcome_result=outcome,strategy_capital_usd=2500)
    assert postmortem["cost_model_id"] == "execution_costs_v1.0"
    assert postmortem["net_result_code"] == benchmark["comparison"]["net_result_code"]
    assert postmortem["scout_performance"]["net_realized_pnl_usd"] == benchmark["scout_summary"]["net_realized_pnl_usd"]
    for policy in postmortem["execution_policy_review"]:
        for cohort in policy["cohort_summaries"].values():
            assert cohort["net_realized_return_pct"] < cohort["realized_return_pct"]


def test_cost_csv_columns_and_values(tmp_path):
    helper=runpy.run_path(str(ROOT/"tests/test_scorable_outcomes.py"))
    snapshot,scout,outcome,benchmark=helper["_artifacts"]()
    outcome["cost_model_id"]="execution_costs_v1.0"
    execution=outcome["outcomes"][0]["execution_result"]
    execution.update(net_realized_return_pct=-1.5,cost_bps_of_position=200,net_capture_ratio=-.15)
    paths=export_daily_outcomes(tmp_path,snapshot,scout,outcome,benchmark)
    ids=(helper["PRIMARY"],helper["ATR"])
    for path,expected in zip(paths,(scorable_outcomes_columns(ids),eligible_outcomes_columns(ids))):
        with path.open() as handle:
            reader=csv.DictReader(handle);rows=list(reader)
            assert reader.fieldnames == list(expected)
        assert all(row["cost_model_id"]=="execution_costs_v1.0" for row in rows)
        assert any(row[f"{ids[0]}_net_realized_return_pct"]=="-1.500000" for row in rows)


def test_gross_win_net_miss_and_config_switch(monkeypatch):
    helper=runpy.run_path(str(ROOT/"tests/test_execution_policy_comparison.py"))
    snapshot=helper["_snapshot"]()
    snapshot["universe_version"]="research_universe_v1.0"
    snapshot["securities"]=[dict(ticker=t,eligible=True,market_data={}) for t in ("LOW","HIGH")]
    scout={k:snapshot[k] for k in ("universe_mode","universe_manifest_hash","universe_coverage","research_evidence","promotion_eligible")}
    scout["candidates"]=[dict(ticker=t,research_selected=t=="LOW",rank=i+1,score_pct=80) for i,t in enumerate(("LOW","HIGH"))]
    bars={t:[dict(timestamp="2024-03-15T09:30:00-04:00",open=price,low=price,high=price*(1+ret),close=price*(1+ret),volume=1000)] for t,price,ret in (("LOW",2.89,.015),("HIGH",30,.01))}
    outcome=grade_replay_outcomes(snapshot,scout,bars,10000)
    result=build_same_universe_benchmark(snapshot,scout,outcome,10000)
    assert result["comparison"]["result_code"] == "SCOUT_OUTPERFORMED"
    assert result["comparison"]["net_result_code"] == "SCOUT_UNDERPERFORMED"
    assert result["comparison"]["net_scout_won"] is False
    monkeypatch.setattr("trainer.benchmark.load_verdict_basis",lambda:"NET")
    net_basis=build_same_universe_benchmark(snapshot,scout,outcome,10000)
    assert net_basis["comparison"]["result_code"] == "SCOUT_UNDERPERFORMED"
    assert net_basis["comparison"]["gross_result_code"] == "SCOUT_OUTPERFORMED"
    assert net_basis["promotion_eligible"] == result["promotion_eligible"]


def test_cumulative_csv_accepts_only_additive_cost_columns(tmp_path):
    helper=runpy.run_path(str(ROOT/"tests/test_scorable_outcomes.py"))
    original=None
    for date in ("2024-03-15","2024-03-18"):
        snapshot,scout,outcome,benchmark=helper["_artifacts"]()
        snapshot["trading_date"]=date
        paths=export_daily_outcomes(tmp_path/"days"/date,snapshot,scout,outcome,benchmark)
        if date=="2024-03-18":
            for path in paths:
                with path.open() as handle:
                    reader=csv.DictReader(handle);rows=list(reader);fields=[k for k in reader.fieldnames if k!='cost_model_id' and not k.endswith(('_net_realized_return_pct','_cost_bps','_net_capture_ratio'))]
                with path.open('w',newline='') as handle:
                    writer=csv.DictWriter(handle,fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
            original=[p.read_bytes() for p in paths]
    outputs=concatenate_completed_outcomes(tmp_path,["2024-03-15","2024-03-18"])
    assert [p.read_bytes() for p in paths]==original
    for path in outputs:
        with path.open() as handle:
            reader=csv.DictReader(handle);rows=list(reader)
            assert 'cost_model_id' in reader.fieldnames
            assert all(row['cost_model_id']=='' for row in rows if row['trading_date']=='2024-03-18')


def test_range_and_multi_day_net_aggregates(tmp_path):
    from trainer.output_paths import daily_path
    from trainer.flatfile_range_summary import render_range_summary
    from trainer.multi_day_trainer import _aggregate
    helper=runpy.run_path(str(ROOT/"tests/test_multi_day_trainer.py"))
    records=[]
    for date in ("2024-03-15","2024-03-18"):
        day=tmp_path/"days"/date
        helper["fake_day_runner"](None,[],date,cache_root=tmp_path,output_dir=day,threshold_pct=70,exploration_top_k=0,dataset_partition="DEVELOPMENT",universe_mode="historical_research")
        bp=daily_path(day,date,"benchmark_result")
        benchmark=json.loads(bp.read_text())
        for name in ("scout_summary","exploration_summary","combined_summary"):
            summary=benchmark[name]
            summary["net_realized_return_pct"]=summary["realized_return_pct"]-.5
            summary["net_realized_pnl_usd"]=summary["realized_pnl_usd"]-12.5
        benchmark["return_baselines"].update(net_random_draw_mean_realized_return_pct=1.5,net_random_draw_mean_realized_pnl_usd=37.5)
        benchmark["comparison"].update(result_code="SCOUT_OUTPERFORMED",net_result_code="SCOUT_UNDERPERFORMED")
        bp.write_text(json.dumps(benchmark))
        pp=daily_path(day,date,"postmortem")
        postmortem=json.loads(pp.read_text());postmortem.update(result="WIN",net_result="MISS");pp.write_text(json.dumps(postmortem))
        records.append(dict(trading_date=date,status="COMPLETE",artifact_directory=str(day.relative_to(tmp_path))))
    aggregate,*_= _aggregate(records,tmp_path)
    assert aggregate["scout_cumulative_return_pct"] == pytest.approx(2.01)
    assert aggregate["net_scout_cumulative_return_pct"] == pytest.approx(1.0025)
    assert aggregate["net_scout_total_realized_pnl_usd"] == 25
    assert aggregate["net_misses"] == aggregate["gross_win_to_net_non_win_count"] == 2
    manifest=dict(days=records,requested_dates=[r['trading_date'] for r in records],completed_dates=[r['trading_date'] for r in records])
    summary=render_range_summary(manifest,tmp_path)
    assert 'Net WIN/TIE/MISS: 0/0/2' in summary
    assert 'Gross WIN to net TIE/MISS: 2' in summary
    assert 'net cumulative return: 0.6000%' in summary
