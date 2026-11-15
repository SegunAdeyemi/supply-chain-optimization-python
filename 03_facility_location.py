"""Model 3: Capacitated facility location (mixed-integer programming).
Decide which candidate DCs to open (yes/no) and how much each ships to each store,
minimizing fixed opening cost plus shipping cost."""
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

FIXED = np.array([100, 120, 90])
CAP = np.array([80, 90, 70])
DEM = np.array([30, 25, 35, 20])
SHIP = np.array([[2, 4, 5, 3],       # DC1 -> stores 1..4 ($/unit)
                 [3, 2, 4, 5],       # DC2
                 [5, 3, 2, 4]])      # DC3


def solve(fixed=FIXED, cap=CAP, dem=DEM, ship=SHIP):
    nD, nS = ship.shape
    nv = nD + nD * nS                           # y[i] open?, then x[i,j] units
    c = np.concatenate([fixed, ship.flatten()])
    Aeq = np.zeros((nS, nv))                    # each store's demand is met
    for j in range(nS):
        for i in range(nD):
            Aeq[j, nD + i * nS + j] = 1
    Aub = np.zeros((nD, nv))                    # ship only from open DCs, within capacity
    for i in range(nD):
        Aub[i, nD + i * nS:nD + (i + 1) * nS] = 1
        Aub[i, i] = -cap[i]
    res = milp(c,
               constraints=[LinearConstraint(Aeq, dem, dem),
                            LinearConstraint(Aub, -np.inf, 0)],
               integrality=np.concatenate([np.ones(nD), np.zeros(nD * nS)]),
               bounds=Bounds(0, np.concatenate([np.ones(nD), np.full(nD * nS, np.inf)])))
    if not res.success:
        return None
    opened = [i + 1 for i in range(nD) if res.x[i] > 0.5]
    return res.fun, opened


def line(label, out):
    if out is None:
        print(f"{label:<46} INFEASIBLE")
    else:
        print(f"{label:<46} total ${out[0]:>6.0f}   open DCs: {out[1]}")


if __name__ == "__main__":
    line("Base case", solve())
    print("\nSweep: fixed cost of DC2 (base 120)")
    for f in (120, 100, 80, 60, 40):
        line(f"  DC2 fixed cost = {f}", solve(fixed=np.array([100, f, 90])))
    print("\nSweep: demand scaled up")
    for m in (1.0, 1.2, 1.4, 1.6):
        line(f"  demand x{m}", solve(dem=(DEM * m).round()))
    print("\nWhat if DC1 capacity drops to 40?")
    line("  DC1 capacity = 40", solve(cap=np.array([40, 90, 70])))
