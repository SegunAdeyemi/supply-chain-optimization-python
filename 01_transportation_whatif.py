"""Model 1: Transportation LP with what-if scenarios.
Two warehouses ship to three stores at minimum total shipping cost."""
import numpy as np
from scipy.optimize import linprog

BASE_COST = np.array([[4, 6, 8],     # warehouse A -> stores 1, 2, 3 ($/unit)
                      [5, 3, 7]])    # warehouse B -> stores 1, 2, 3
BASE_SUPPLY = [60, 40]
BASE_DEMAND = [30, 40, 30]


def solve(cost=BASE_COST, supply=BASE_SUPPLY, demand=BASE_DEMAND, lane_caps=None):
    """Minimize shipping cost. lane_caps: {(warehouse, store): max_units}."""
    n_w, n_s = cost.shape
    c = cost.flatten()                       # x[w*n_s + s]
    A_ub = np.zeros((n_w, n_w * n_s))
    for w in range(n_w):
        A_ub[w, w * n_s:(w + 1) * n_s] = 1   # warehouse w ships at most its supply
    A_eq = np.zeros((n_s, n_w * n_s))
    for s in range(n_s):
        for w in range(n_w):
            A_eq[s, w * n_s + s] = 1         # each store gets exactly its demand
    bounds = [(0, None)] * (n_w * n_s)
    for (w, s), cap in (lane_caps or {}).items():
        bounds[w * n_s + s] = (0, cap)
    res = linprog(c, A_ub=A_ub, b_ub=supply, A_eq=A_eq, b_eq=demand, bounds=bounds)
    return res


def show(name, res):
    if res.status == 0:
        plan = res.x.reshape(2, -1).round(0).astype(int).tolist()
        print(f"{name:<46} cost ${res.fun:>6.0f}   plan (rows=A,B): {plan}")
    else:
        print(f"{name:<46} INFEASIBLE (supply cannot meet demand)")


if __name__ == "__main__":
    show("Base case", solve())

    # Naive plan: each store ships from its cheapest warehouse, ignoring supply limits
    naive = BASE_COST.argmin(axis=0)
    naive_cost = sum(BASE_COST[naive[s], s] * BASE_DEMAND[s] for s in range(3))
    b_load = sum(BASE_DEMAND[s] for s in range(3) if naive[s] == 1)
    print(f"{'Naive cheapest-warehouse plan':<46} cost ${naive_cost:>6.0f}   but B must ship {b_load} units, has {BASE_SUPPLY[1]}: infeasible")

    show("Lane B->store 2 capped at 20 (closure)", solve(lane_caps={(1, 1): 20}))
    show("Fuel spike: all B lanes cost +$2", solve(cost=BASE_COST + np.array([[0, 0, 0], [2, 2, 2]])))
    show("Store 3 demand +10 (needs 10 more supply)", solve(supply=[70, 40], demand=[30, 40, 40]))
    show("Warehouse B supply drops to 30", solve(supply=[60, 30]))
    show("Warehouse A->store 3 cost falls $8 -> $5", solve(cost=np.array([[4, 6, 5], [5, 3, 7]])))
