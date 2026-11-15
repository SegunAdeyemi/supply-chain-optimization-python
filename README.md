# Supply Chain Optimization: Three Small Models in Python

A learning project applying linear and mixed-integer programming to common logistics decisions. All data is made up for practice. No employer or client data is used.

Tools: Python, NumPy, SciPy (`linprog` and `milp`).

## Models

### 1. Transportation LP with what-if scenarios (`01_transportation_whatif.py`)
**Problem:** Two warehouses ship to three stores. Minimize shipping cost while meeting each store's demand and respecting warehouse supply.
**Model:** Decision variables are units shipped on each lane. The objective is total shipping cost. Constraints are warehouse supply limits and exact store demand.
**Results (see `results/01_transportation_whatif.txt`):**
- Optimal plan costs **$480**.
- The "ship from each store's cheapest warehouse" shortcut looks cheaper ($450) but is infeasible: Warehouse B would have to ship 70 units and has 40.
- Capping the B to store 2 lane at 20 units (a closure) raises cost to **$520**.
- Raising B's lane costs by $2 (fuel spike) raises cost to **$560**.
- Cutting Warehouse B's supply to 30 makes the problem **infeasible**: total supply no longer covers demand.

### 2. Eight-node inventory rebalancing (`02_eight_node_inventory.py`)
**Problem:** Eight distribution nodes hold different amounts of stock against safety-stock targets. Which nodes have redundant stock, and what is the cheapest way to cover shortfalls?
**Model:** Target stock = mean demand x lead time + z x demand standard deviation x sqrt(lead time) (z = 1.65, about a 95% cycle service level). A transshipment LP moves surplus stock to deficit nodes at minimum transfer cost, with a penalty on any unmet deficit.
**Results (seeded random data, see `results/02_eight_node_inventory.txt`):**
- 951 units of surplus and 110 units of deficit across the network.
- The LP covers the full deficit by moving 110 units from the node with the most economical transfer, at a cost of $353.
- **841 units of redundant stock remain**, with an annual holding cost of about $4,207 under the assumed unit cost ($25) and holding rate (20%).
**Takeaway:** looking at the network as a whole shows that most nodes are over-stocked while a few are short, which a node-by-node view hides.

### 3. Capacitated facility location, MIP (`03_facility_location.py`)
**Problem:** Choose which of three candidate distribution centers to open to serve four stores at minimum fixed plus shipping cost.
**Model:** Yes/no variable per DC (integer), shipment variables per DC and store, capacity linked to whether a DC is open, and demand met exactly.
**Results (see `results/03_facility_location.txt`):**
- Base case: open DC1 and DC3, total cost **$455**.
- If DC2's fixed cost falls from 120 to 60, the optimal choice switches to DC2 + DC3 (**$440**).
- Scaling demand up 40% also switches the plan to DC2 + DC3 (**$623**). At 60% growth, all three DCs open.
**Takeaway:** the best network is sensitive to fixed costs and demand growth, so it should be tested with scenarios, not decided from one run.

## Run it
```
pip install -r requirements.txt
python 01_transportation_whatif.py
python 02_eight_node_inventory.py
python 03_facility_location.py
```

## What I learned
(Write 3 or 4 sentences in your own words: what surprised you, which result you would explain to a client, what you would add next.)

## Limitations
Small, simplified, made-up data. Real projects need real cost data, validated inputs, more constraints (service times, labor, multiple products), and review with the people who run the operation.
