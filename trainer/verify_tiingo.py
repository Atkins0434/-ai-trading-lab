from __future__ import annotations

import argparse

from trainer.providers.tiingo import TiingoClient


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify Tiingo credentials and daily-price access."
    )
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument("--date", default="2018-01-02")
    args = parser.parse_args()

    client = TiingoClient.from_environment()
    records = client.get_daily_prices(
        ticker=args.ticker,
        start_date=args.date,
        end_date=args.date,
    )
    print(
        f"Tiingo connection PASS: {args.ticker} "
        f"{args.date} returned {len(records)} record(s)."
    )


if __name__ == "__main__":
    main()
