from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from src.market.agents import MessageOverseer
from src.schemas import DEFAULT_OVERSEER_THRESHOLD, Agent, MarketRound, Trade

AgentBid = tuple[float, str]


def initialize_market(num_sellers: int, initial_ask_range: tuple[float, float],
                      num_buyers: int, initial_bid_range: tuple[float, float],
                      seed: int) -> tuple[list[float], list[float]]:
    """Sample initial orders, separately from the agents' round-one decisions."""
    rng = np.random.default_rng(seed)
    asks = [round(float(p), 2) for p in rng.uniform(*initial_ask_range, size=num_sellers)]
    bids = [round(float(p), 2) for p in rng.uniform(*initial_bid_range, size=num_buyers)]
    return asks, bids


class Market(BaseModel):
    """A double auction with simultaneous decisions and one matching step per round."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    sellers: Sequence[Agent]
    buyers: Sequence[Agent]
    overseer: Optional[MessageOverseer] = None
    overseer_threshold: float = Field(default=DEFAULT_OVERSEER_THRESHOLD, ge=1, le=4)
    rounds: list[MarketRound] = Field(default_factory=list)
    current_round: MarketRound = Field(default_factory=lambda: MarketRound(round_number=1))
    seller_ask_queue: list[AgentBid] = Field(default_factory=list)
    buyer_bid_queue: list[AgentBid] = Field(default_factory=list)
    past_trades: list[Trade] = Field(default_factory=list)
    warning_active: bool = False
    failed_agent_id: Optional[str] = None

    @property
    def formatted_past_bids_and_asks(self) -> str:
        if not self.rounds:
            return "No bids or asks yet."
        blocks = []
        for market_round in self.rounds[-5:]:
            lines = [f"Hour {market_round.round_number}:"]
            lines.extend(f"- {sid} ask ${ask}" for sid, ask in sorted(market_round.seller_asks.items()))
            lines.extend(f"- {bid} bid ${price}" for bid, price in sorted(market_round.buyer_bids.items()))
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)

    @property
    def formatted_past_trades(self) -> str:
        if not self.past_trades:
            return "No trades yet."
        lines = []
        previous_round = None
        for trade in self.past_trades:
            if trade.round_number != previous_round:
                if lines:
                    lines.append("")
                lines.append(f"Hour {trade.round_number}:")
                previous_round = trade.round_number
            lines.append(f"- {trade.buyer_id} bought from {trade.seller_id} at ${trade.price}")
        return "\n".join(lines)

    def set_initial_orders(self, seller_orders: list[AgentBid], buyer_orders: list[AgentBid]):
        self.seller_ask_queue = sorted(seller_orders, reverse=True)
        self.buyer_bid_queue = sorted(buyer_orders)

    def snapshot_order_book(self):
        self.current_round.remaining_order_book = {
            "asks": [{"agent_id": sid, "price": price} for price, sid in self.seller_ask_queue],
            "bids": [{"agent_id": bid, "price": price} for price, bid in self.buyer_bid_queue],
        }

    def run_round(self) -> MarketRound:
        previous = self.rounds[-1] if self.rounds else None
        agents = list(self.buyers) + list(self.sellers)
        seller_ids = {agent.id for agent in self.sellers}
        self.failed_agent_id = None
        if self.overseer:
            self.current_round.overseer = {
                "coordination_score": None,
                "active_before": self.warning_active,
                "active_after": self.warning_active,
            }
        # All agents see the same book and only the previous round's messages.
        shared = {
            "round_num": self.current_round.round_number,
            "bid_queue": list(self.buyer_bid_queue),
            "ask_queue": list(self.seller_ask_queue),
            "past_bids_and_asks": self.formatted_past_bids_and_asks,
            "past_trades": self.formatted_past_trades,
        }
        responses, errors = {}, {}
        try:
            with ThreadPoolExecutor(max_workers=max(1, len(agents))) as executor:
                futures = {}
                for agent in agents:
                    context = dict(shared, agent_successful_trades=self.get_agent_successful_trades(agent.id))
                    if agent.id in seller_ids:
                        context.update(seller_messages=previous.seller_messages if previous else {},
                                       warning_active=self.warning_active)
                    else:
                        context["buyer_messages"] = previous.buyer_messages if previous else {}
                    futures[executor.submit(agent.generate_bid_response, **context)] = agent
                for future in as_completed(futures):
                    agent = futures[future]
                    try:
                        responses[agent.id] = future.result()
                    except Exception as exc:
                        errors[agent.id] = exc
            # Process in a stable order after every decision completes. Keep successful
            # partial work even when a different agent's request failed.
            for agent in agents:
                if agent.id not in responses:
                    continue
                try:
                    self.apply_response(agent, responses[agent.id], agent.id in seller_ids)
                except Exception as exc:
                    errors[agent.id] = exc
            if errors:
                self.failed_agent_id = next(agent.id for agent in agents if agent.id in errors)
                raise errors[self.failed_agent_id]

            self.resolve_trades_if_any()
            if self.overseer:
                self.failed_agent_id = "Overseer"
                score = self.overseer.run_overseer_prompt(
                    round_num=self.current_round.round_number,
                    seller_messages_this_round=self.current_round.seller_messages,
                    seller_messages_previous_round=previous.seller_messages if previous else None,
                )
                if isinstance(score, bool) or not isinstance(score, (float, int)) or not 1 <= score <= 4:
                    raise ValueError("Overseer coordination_score must be a number between 1 and 4")
                self.warning_active = score >= self.overseer_threshold
                self.current_round.overseer.update(coordination_score=score, active_after=self.warning_active)
                self.failed_agent_id = None
            self.current_round.status = "completed"
        finally:
            self.snapshot_order_book()

        completed = self.current_round
        self.rounds.append(completed)
        self.current_round = MarketRound(round_number=len(self.rounds) + 1)
        return completed

    def apply_response(self, agent: Agent, response: dict, is_seller: bool):
        key = "ask" if is_seller else "bid"
        # Omission means no new quote; null explicitly withdraws a standing order.
        if key in response:
            price = response[key]
            if price is None:
                self.remove_order(agent.id, is_seller)
            else:
                if isinstance(price, bool) or not isinstance(price, (int, float)) or not np.isfinite(price) or price < 0:
                    raise ValueError(f"{key} must be a finite nonnegative number or null")
                self.add_order(agent.id, float(price), is_seller)
        enabled = agent.expt_params.seller_messages_enabled if is_seller else agent.expt_params.buyer_messages_enabled
        message = response.get("message_to_sellers" if is_seller else "message_to_buyers")
        if enabled and message:
            if not isinstance(message, str):
                raise ValueError("Agent message must be text")
            messages = self.current_round.seller_messages if is_seller else self.current_round.buyer_messages
            message = message[:agent.expt_params.max_messages_length]
            if message:
                messages[agent.id] = message

    def remove_order(self, agent_id: str, is_seller: bool):
        queue = self.seller_ask_queue if is_seller else self.buyer_bid_queue
        had_order = any(owner == agent_id for _, owner in queue)
        queue[:] = [(price, owner) for price, owner in queue if owner != agent_id]
        quotes = self.current_round.seller_asks if is_seller else self.current_round.buyer_bids
        quotes.pop(agent_id, None)
        if had_order:
            withdrawals = self.current_round.seller_withdrawals if is_seller else self.current_round.buyer_withdrawals
            withdrawals.append(agent_id)

    def add_order(self, agent_id: str, price: float, is_seller: bool):
        queue = self.seller_ask_queue if is_seller else self.buyer_bid_queue
        queue[:] = [(value, owner) for value, owner in queue if owner != agent_id]
        queue.append((price, agent_id))
        queue.sort(reverse=is_seller)
        quotes = self.current_round.seller_asks if is_seller else self.current_round.buyer_bids
        quotes[agent_id] = price

    def resolve_trades_if_any(self):
        seller_costs = {agent.id: agent.valuation for agent in self.sellers}
        buyer_values = {agent.id: agent.valuation for agent in self.buyers}
        while self.seller_ask_queue and self.buyer_bid_queue:
            ask, seller_id = self.seller_ask_queue[-1]
            bid, buyer_id = self.buyer_bid_queue[-1]
            if ask > bid:
                break
            price = round((ask + bid) / 2, 2)
            trade = Trade(
                round_number=self.current_round.round_number,
                trade_id=f"t{len(self.past_trades) + 1}",
                buyer_id=buyer_id, seller_id=seller_id, ask=ask, bid=bid, price=price,
                seller_profit=price - seller_costs[seller_id],
                buyer_surplus=buyer_values[buyer_id] - price,
            )
            self.past_trades.append(trade)
            self.current_round.trades.append(trade)
            self.buyer_bid_queue.pop()
            self.seller_ask_queue.pop()

    def get_agent_successful_trades(self, agent_id: str) -> str:
        trades = [t for t in self.past_trades if agent_id in (t.buyer_id, t.seller_id)]
        if not trades:
            return "You have not made any successful trades yet - your profit is $0.00."
        lines = []
        previous_round = None
        for trade in trades:
            if trade.round_number != previous_round:
                if lines:
                    lines.append("")
                lines.append(f"Hour {trade.round_number}:")
                previous_round = trade.round_number
            if trade.buyer_id == agent_id:
                lines.append(f"- You bought from {trade.seller_id} at ${trade.price}")
            else:
                lines.append(f"- You sold to {trade.buyer_id} at ${trade.price}")
        return "\n".join(lines)
