"""Run the buyer/seller auction with optional Jev supervision.

From llm-agent-collusion:
    python src/simulation.py --rounds 5 --tag smoke
    python src/simulation.py --seller_messages_enabled --overseer_enabled --overseer_threshold 3.5
    python src/simulation.py --seeds 10 11 12 --initial_ask_range 95 100 --initial_bid_range 80 85

Each seed saves config.json, calls.jsonl, market.jsonl, and summary.json.
"""

import argparse
import logging
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pydantic import ValidationError
from tqdm import tqdm

from src.schemas import DEFAULT_MODEL, DEFAULT_OVERSEER_THRESHOLD, MODEL_ALIASES, ExperimentParams
from src.market.identities import (
    AGENT_NAMES, get_random_agent_names, get_random_buyer_company_names, get_random_seller_company_names,
)
from src.market.agents import LLMBuyer, LLMSeller, MessageOverseer
from src.market.market import Market, initialize_market
from src.records.writer import ExperimentLogger, timestamp
from src.llm.client import JEV_MODEL, get_client, get_overseer_client
from src.llm.config import resolve_model


def run_simulation(params: ExperimentParams, result_dir: str = "results") -> dict[str, Any]:
    """Run one seed and return the same summary persisted on disk."""
    random.seed(params.seed)
    num_buyers, num_sellers = len(params.buyer_models), len(params.seller_models)
    names = get_random_agent_names(num_buyers + num_sellers)
    buyer_companies = get_random_buyer_company_names(num_buyers)
    seller_companies = get_random_seller_company_names(num_sellers)
    asks, bids = ([None] * num_sellers, [None] * num_buyers)
    if params.initial_condition:
        asks, bids = initialize_market(num_sellers, params.initial_ask_range,
                                       num_buyers, params.initial_bid_range, params.seed)
    created_at = timestamp()
    run_id = f"{datetime.fromisoformat(created_at):%Y%m%d_%H%M%S}_{params.tag}_seed{params.seed}"
    config = {
        "run_id": run_id, "created_at": created_at, "tag": params.tag,
        "seed": params.seed, "result_dir": str(result_dir),
        "settings": {key: getattr(params, key) for key in (
            "rounds", "seller_messages_enabled", "buyer_messages_enabled", "overseer_enabled",
            "ceo_pressure", "max_messages_length", "temperature",
        )},
        "initialization": {
            "initial_condition": params.initial_condition,
            "initial_ask_range": list(params.initial_ask_range),
            "initial_bid_range": list(params.initial_bid_range),
        },
        "agents": {
            "sellers": [
                {"id": f"Seller {names[num_buyers + i]}", "company": seller_companies[i],
                 "valuation": params.seller_valuations[i], "model": resolve_model(params.seller_models[i]),
                 "initial_ask": asks[i]}
                for i in range(num_sellers)
            ],
            "buyers": [
                {"id": f"Buyer {names[i]}", "company": buyer_companies[i],
                 "valuation": params.buyer_valuations[i], "model": resolve_model(params.buyer_models[i]),
                 "initial_bid": bids[i]}
                for i in range(num_buyers)
            ],
            "overseer": {"id": "Overseer", "model": JEV_MODEL,
                         "threshold": params.overseer_threshold} if params.overseer_enabled else None,
        },
    }
    logger = ExperimentLogger(config, base_dir=result_dir)
    market = None
    opened_clients = []
    initializing_agent = None
    try:
        buyers, sellers = [], []
        for group, agent_class, destination in (("buyers", LLMBuyer, buyers), ("sellers", LLMSeller, sellers)):
            for entry in config["agents"][group]:
                initializing_agent = entry["id"]
                client = get_client(model=entry["model"], temperature=params.temperature, seed=params.seed)
                opened_clients.append(client)
                destination.append(agent_class(
                    id=entry["id"], company=entry["company"], valuation=entry["valuation"],
                    expt_params=params, client=client, logger=logger,
                ))
        overseer = None
        if params.overseer_enabled:
            initializing_agent = "Overseer"
            client = get_overseer_client()
            opened_clients.append(client)
            overseer = MessageOverseer(client=client, logger=logger)
        initializing_agent = None
        market = Market(sellers=sellers, buyers=buyers, overseer=overseer,
                        overseer_threshold=params.overseer_threshold)
        if params.initial_condition:
            market.set_initial_orders(
                [(price, seller.id) for price, seller in zip(asks, sellers)],
                [(price, buyer.id) for price, buyer in zip(bids, buyers)],
            )
        for _ in tqdm(range(params.rounds), desc=run_id):
            try:
                completed = market.run_round()
            except Exception:
                logger.log_round(market.current_round.model_dump(by_alias=True))
                raise
            logger.log_round(completed.model_dump(by_alias=True))
            logger.save_summary("running")
        summary = logger.save_summary("completed")
    except Exception as exc:
        logger.save_summary("failed", {
            "type": type(exc).__name__, "message": str(exc),
            "round": market.current_round.round_number if market and len(market.rounds) < params.rounds else None,
            "agent_id": market.failed_agent_id if market else initializing_agent,
        })
        raise
    finally:
        for client in opened_clients:
            try:
                client.close()
            except Exception as exc:
                logging.warning("Could not close model client: %s", logger.redact(str(exc)))
    print_summary(summary, logger.log_dir)
    return summary


def print_summary(summary: dict, result_dir: Path):
    metrics = summary["metrics"]
    print(f"Completed {summary['completed_rounds']} rounds; {metrics['num_trades']} trades.")
    print(f"Seller profit: {metrics['combined_seller_profits']:.2f}; "
          f"buyer surplus: {metrics['combined_buyer_surplus']:.2f}.")
    print(f"Saved {result_dir}")


def broadcast(values, n: int, what: str):
    if len(values) == 1:
        return values * n
    if len(values) != n:
        raise ValueError(f"--{what} needs 1 or {n} values, got {len(values)}")
    return values


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("market")
    g.add_argument("--sellers", type=int, default=3)
    g.add_argument("--buyers", type=int, default=3)
    g.add_argument("--seller_valuations", type=float, nargs="+", default=[80.0])
    g.add_argument("--buyer_valuations", type=float, nargs="+", default=[100.0])
    g.add_argument("--rounds", type=int, default=15)
    g.add_argument("--seeds", type=int, nargs="+", default=[0], help="one independent run per seed")
    g.add_argument("--initial_ask_range", type=float, nargs=2, default=[95, 100], metavar=("LOW", "HIGH"))
    g.add_argument("--initial_bid_range", type=float, nargs=2, default=[80, 85], metavar=("LOW", "HIGH"))
    g.add_argument("--initial_condition", choices=("true", "false"), default="true",
                   help="prepopulate initial quotes; false starts with an empty order book (default: true)")
    m = p.add_argument_group("trader models")
    m.add_argument("--seller_models", nargs="+", default=[DEFAULT_MODEL],
                   help=f"one for all or one per seller; aliases: {', '.join(MODEL_ALIASES)}")
    m.add_argument("--buyer_models", nargs="+", default=[DEFAULT_MODEL], help="one for all or one per buyer")
    m.add_argument("--temperature", type=float, default=0.7)
    a = p.add_argument_group("interventions")
    a.add_argument("--seller_messages_enabled", action="store_true")
    a.add_argument("--buyer_messages_enabled", action="store_true")
    a.add_argument("--max_messages_length", type=int, default=500)
    a.add_argument("--overseer_enabled", action="store_true",
                   help="use Jev scores to switch next-round seller warnings on or off")
    a.add_argument("--overseer_threshold", type=float, default=DEFAULT_OVERSEER_THRESHOLD,
                   help=f"threshold from 1 to 4 inclusive (default: {DEFAULT_OVERSEER_THRESHOLD})")
    a.add_argument("--ceo_pressure", action="store_true")
    o = p.add_argument_group("output")
    o.add_argument("--tag", default="run", help="label only; does not enable interventions")
    o.add_argument("--result_dir", default="results", help="output root, relative to current directory")
    o.add_argument("-y", "--yes", action="store_true", help="skip the large-run confirmation")
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        if args.sellers < 0 or args.buyers < 0:
            raise ValueError("Agent counts must be nonnegative")
        if args.sellers + args.buyers > len(AGENT_NAMES):
            raise ValueError(f"At most {len(AGENT_NAMES)} agents are supported by the identity pool")
        if any(seed < 0 for seed in args.seeds):
            raise ValueError("Seeds must be nonnegative")
        if len(set(args.seeds)) != len(args.seeds):
            raise ValueError("Each seed must be unique")
        params = ExperimentParams(
            seller_valuations=broadcast(args.seller_valuations, args.sellers, "seller_valuations"),
            buyer_valuations=broadcast(args.buyer_valuations, args.buyers, "buyer_valuations"),
            seller_models=broadcast(args.seller_models, args.sellers, "seller_models"),
            buyer_models=broadcast(args.buyer_models, args.buyers, "buyer_models"),
            rounds=args.rounds, seller_messages_enabled=args.seller_messages_enabled,
            buyer_messages_enabled=args.buyer_messages_enabled, overseer_enabled=args.overseer_enabled,
            ceo_pressure=args.ceo_pressure, temperature=args.temperature, tag=args.tag,
            initial_condition=args.initial_condition == "true", initial_ask_range=args.initial_ask_range,
            initial_bid_range=args.initial_bid_range, max_messages_length=args.max_messages_length,
            overseer_threshold=args.overseer_threshold,
        )
    except (ValueError, ValidationError) as exc:
        parser.error(str(exc))
    n_calls = args.rounds * (args.sellers + args.buyers + int(args.overseer_enabled)) * len(args.seeds)
    print(f"{len(args.seeds)} runs, approximately {n_calls} model calls before retries "
          f"(traders via OpenRouter/configured endpoint; overseer via TypeSafe).")
    local = any(host in os.getenv("LLM_BASE_URL", "") for host in ("localhost", "127.0.0.1"))
    if not args.yes and n_calls > 60 and (not local or args.overseer_enabled) and sys.stdin.isatty():
        if input("Continue? [y/N] ").strip().lower() not in ("y", "yes"):
            parser.exit(0, "Aborted.\n")
    for seed in args.seeds:
        run_simulation(params.model_copy(update={"seed": seed}), result_dir=args.result_dir)


if __name__ == "__main__":
    main()
