"""Model 2: Eight-node inventory rebalancing with MADE-UP data (no client data).

Each node has demand, lead time and on-hand stock. Safety-stock targets come from
target = mean_demand * lead_time + z * sd * sqrt(lead_time).
Surplus nodes can transfer stock to deficit nodes at a per-unit cost based on distance.
An LP finds the cheapest transfers; any unmet deficit carries a penalty, and any
surplus left over is the 'redundant' stock that carries holding cost."""
import numpy as np
from scipy.optimize import linprog

rng = np.random.default_rng(7)
N = 8
Z = 1.65                  # ~95% cycle service level
UNIT_COST = 25.0          # $ per unit (made up)
HOLD_RATE = 0.20          # annual holding cost as share of unit cost (made up)
SHORT_PENALTY = 60.0      # $ per unit of unmet deficit (made up)
COST_PER_MILE = 0.01      # $ per unit per mile transferred (made up)

mean_d = rng.integers(40, 160, N)              # units/day
sd_d = (0.25 * mean_d).round(1)
lead = rng.integers(3, 8, N)                   # days
target = mean_d * lead + Z * sd_d * np.sqrt(lead)
onhand = (target * rng.uniform(0.6, 1.9, N)).round()
xy = rng.uniform(0, 800, (N, 2))               # node locations in miles
dist = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=2)
tcost = COST_PER_MILE * dist

surplus = np.maximum(onhand - target, 0)
deficit = np.maximum(target - onhand, 0)

# Variables: t[i,j] transfers (i != j), then u[j] unmet deficit
pairs = [(i, j) for i in range(N) for j in range(N) if i != j]
nt = len(pairs)
c = np.concatenate([[tcost[i, j] for i, j in pairs], np.full(N, SHORT_PENALTY)])
A_ub = np.zeros((N, nt + N))                   # outgoing <= surplus
A_eq = np.zeros((N, nt + N))                   # incoming + unmet = deficit
for k, (i, j) in enumerate(pairs):
    A_ub[i, k] = 1
    A_eq[j, k] = 1
for j in range(N):
    A_eq[j, nt + j] = 1
res = linprog(c, A_ub=A_ub, b_ub=surplus, A_eq=A_eq, b_eq=deficit, bounds=(0, None))
assert res.status == 0
t = res.x[:nt]
moved = np.zeros((N, N))
for k, (i, j) in enumerate(pairs):
    moved[i, j] = t[k]
unmet = res.x[nt:]
out = moved.sum(axis=1)
leftover = surplus - out                       # redundant stock after rebalancing

hold = UNIT_COST * HOLD_RATE
print("Node  demand/day  lead  target  on-hand  surplus  deficit  sent  received  leftover")
for n in range(N):
    print(f"{n+1:>4}  {mean_d[n]:>10}  {lead[n]:>4}  {target[n]:>6.0f}  {onhand[n]:>7.0f}  {surplus[n]:>7.0f}  {deficit[n]:>7.0f}  {out[n]:>4.0f}  {moved[:, n].sum():>8.0f}  {leftover[n]:>8.0f}")
print()
print(f"Total surplus before rebalancing : {surplus.sum():.0f} units  -> annual holding cost ${surplus.sum()*hold:,.0f}")
print(f"Total deficit before rebalancing : {deficit.sum():.0f} units")
print(f"Units transferred                : {t.sum():.0f}  (transfer cost ${(t * c[:nt]).sum():,.0f})")
print(f"Unmet deficit after transfers    : {unmet.sum():.0f} units")
print(f"Redundant stock left over        : {leftover.sum():.0f} units  -> annual holding cost ${leftover.sum()*hold:,.0f}")
print()
print("Reading it: surplus nodes fill deficit nodes first. What is still left over is truly")
print("redundant stock: the amount you could stop reordering or release, and its holding cost.")
