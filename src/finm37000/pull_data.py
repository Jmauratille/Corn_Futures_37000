"""Pull a contract's history from Databento and store it for the project."""

import argparse
import logging

import databento as db

from finm37000.data_store import (
    DEFAULT_DATASET,
    default_range,
    history_path,
    pull_history,
)
from finm37000.db_env_util import get_databento_api_key

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point to pull and store a contract's history."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="ZCH7", help="Raw contract symbol.")
    parser.add_argument("--schema", default="trades", help="Databento schema.")
    parser.add_argument("--months", type=int, default=6, help="Months to pull.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument(
        "--overwrite", action="store_true", help="Re-pull an existing file."
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    path = history_path(args.symbol, args.schema)
    if path.exists() and not args.overwrite:
        logger.info("Already stored at %s; pass --overwrite to re-pull.", path)
        return
    client = db.Historical(get_databento_api_key())
    start, end = default_range(client, args.months, args.dataset, args.schema)
    cost = client.metadata.get_cost(
        dataset=args.dataset,
        symbols=[args.symbol],
        stype_in="raw_symbol",
        schema=args.schema,
        start=start,
        end=end,
    )
    logger.info(
        "%s %s [%s, %s): estimated cost $%.4f",
        args.symbol,
        args.schema,
        start,
        end,
        cost,
    )
    pull_history(
        client,
        args.symbol,
        start,
        end,
        schema=args.schema,
        dataset=args.dataset,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
