from typing import List, Optional, Tuple


def competitive_equilibrium(
    buyer_values: List[float], seller_costs: List[float]
) -> Tuple[Optional[float], int]:
    """Use the original market's midpoint at the last profitable matching pair."""
    pairs = [
        (value, cost)
        for value, cost in zip(sorted(buyer_values, reverse=True), sorted(seller_costs))
        if value >= cost
    ]
    if not pairs:
        return None, 0
    value, cost = pairs[-1]
    return (value + cost) / 2, len(pairs)


def jpm_price(buyer_values: List[float], seller_costs: List[float]) -> Optional[float]:
    """Return the uniform seller price maximizing joint profit, including no trade."""
    values = sorted(buyer_values, reverse=True)
    costs = sorted(seller_costs)
    best_price, best_profit, total_cost = None, 0.0, 0.0
    for quantity, (value, cost) in enumerate(zip(values, costs), start=1):
        total_cost += cost
        profit = value * quantity - total_cost
        if profit >= best_profit:
            best_price, best_profit = value, profit
    return best_price
