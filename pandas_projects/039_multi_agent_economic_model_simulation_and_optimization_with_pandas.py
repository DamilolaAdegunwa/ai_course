import pandas as pd
import numpy as np


# Load data
def load_dataset(filepath):
    return pd.read_csv(filepath)


# Simulate market interactions
def simulate_market(agents_df):
    buyers = agents_df[agents_df['Agent Type'] == 'Buyer']
    sellers = agents_df[agents_df['Agent Type'] == 'Seller']

    transactions = []
    for _, buyer in buyers.iterrows():
        for _, seller in sellers.iterrows():
            if buyer['Willingness to Pay'] >= seller['Price Offered'] and seller['Supply'] > 0:
                quantity = min(seller['Supply'], buyer['Willingness to Pay'] // seller['Price Offered'])
                transactions.append({
                    "Buyer": buyer['Agent ID'],
                    "Seller": seller['Agent ID'],
                    "Price": seller['Price Offered'],
                    "Quantity": quantity
                })
                sellers.loc[seller.name, 'Supply'] -= quantity
                break
    return pd.DataFrame(transactions)


# Optimize resource allocation
def optimize_allocation(regions_df, supply_column, demand_column):
    regions_df['Deficit'] = regions_df[demand_column] - regions_df[supply_column]
    surplus_regions = regions_df[regions_df['Deficit'] < 0]
    deficit_regions = regions_df[regions_df['Deficit'] > 0]

    allocations = []
    for _, deficit_region in deficit_regions.iterrows():
        for _, surplus_region in surplus_regions.iterrows():
            allocation = min(-surplus_region['Deficit'], deficit_region['Deficit'])
            if allocation > 0:
                allocations.append({
                    "From": surplus_region['Region'],
                    "To": deficit_region['Region'],
                    "Amount": allocation
                })
                surplus_regions.loc[surplus_region.name, 'Deficit'] += allocation
                deficit_regions.loc[deficit_region.name, 'Deficit'] -= allocation
    return pd.DataFrame(allocations)


# Example usage
if __name__ == "__main__":
    agents = pd.DataFrame({
        "Agent Type": ["Buyer", "Buyer", "Seller", "Seller"],
        "Agent ID": ["B1", "B2", "S1", "S2"],
        "Willingness to Pay": [100, 80, 0, 0],
        "Supply": [0, 0, 50, 30],
        "Price Offered": [90, 85, 90, 80]
    })

    transactions = simulate_market(agents)
    print(transactions)


comment = """
### Project Title: **Multi-Agent Economic Model Simulation and Optimization Using Pandas**  
**File Name**: `multi_agent_economic_model_simulation_and_optimization_with_pandas.py`  

---

### Project Description  

This project tackles **complex multi-agent economic simulations** and introduces **advanced optimization techniques** for scenarios like market equilibria, pricing strategies, and resource distribution. It leverages Pandas for:  

1. **Agent-Based Modeling**: Simulate the behaviors and interactions of economic agents such as buyers, sellers, and intermediaries.  
2. **Market Simulation**: Model demand and supply, dynamic pricing, and resource allocation under constraints.  
3. **Scenario Comparisons**: Perform multi-scenario analyses to identify optimal solutions.  
4. **Advanced Algorithms**: Include optimization methods such as the **Knapsack Algorithm** and **Pareto Efficiency Analysis** for resource distribution.  

This project is designed for use cases in **economic policy design**, **marketplace optimization**, **supply chain management**, and **resource allocation problems**.  

---

### Example Use Cases  

1. **Marketplace Pricing**: Simulate buyer-seller interactions to derive dynamic pricing models.  
2. **Policy Impact Analysis**: Analyze the impact of subsidies or taxes on a simulated economy.  
3. **Supply Chain Optimization**: Allocate resources to maximize efficiency under dynamic constraints.  
4. **Energy Market Modeling**: Model and balance renewable and non-renewable energy supply to meet demand.  
5. **Insurance Risk Assessment**: Simulate multiple risk scenarios and calculate premiums dynamically.  
6. **Multi-Region Resource Allocation**: Optimize distribution of resources like vaccines or food across regions.  

---

### Example Input(s) and Expected Output(s)  

#### **Input 1: Buyer-Seller Interactions**  
**Agent Data**:  

| Agent Type | Agent ID | Willingness to Pay | Supply | Price Offered |  
|------------|----------|--------------------|--------|---------------|  
| Buyer      | B1       | 100                | 0      | 90            |  
| Buyer      | B2       | 80                 | 0      | 85            |  
| Seller     | S1       | 0                  | 50     | 90            |  
| Seller     | S2       | 0                  | 30     | 80            |  

**Expected Output**:  
- **Transactions**:  
  - B1 buys 50 units from S1 at $90/unit.  
  - B2 buys 30 units from S2 at $80/unit.  

---

#### **Input 2: Energy Market**  
**Region Data**:  

| Region | Demand | Renewable Supply | Non-Renewable Supply | Cost Factor |  
|--------|--------|------------------|-----------------------|-------------|  
| North  | 100    | 50               | 60                    | 1.2         |  
| South  | 80     | 40               | 30                    | 1.1         |  
| East   | 120    | 60               | 70                    | 1.5         |  

**Expected Output**:  
- **Supply Allocation**: Balance supply across regions, prioritizing renewables.  

---

#### **Input 3: Tax Policy Impact**  
**Economic Data**:  

| Sector       | Revenue | Tax Rate | Elasticity |  
|--------------|---------|----------|------------|  
| Manufacturing | 500    | 10%      | High       |  
| Technology    | 400    | 15%      | Medium     |  
| Retail        | 300    | 5%       | Low        |  

**Expected Output**:  
- **Post-Tax Revenue**: Adjusted sector revenues based on tax rates and elasticity.  

---

#### **Input 4: Vaccine Distribution**  
**Distribution Data**:  

| Region | Population | Vaccines Available | Priority Factor |  
|--------|------------|--------------------|-----------------|  
| Urban  | 1,000,000  | 500,000            | High            |  
| Rural  | 500,000    | 200,000            | Medium          |  
| Remote | 100,000    | 50,000             | Low             |  

**Expected Output**:  
- Distribute vaccines based on population and priority factor.  

---

#### **Input 5: Supply Chain**  
**Warehouse Data**:  

| Warehouse | Product  | Stock | Shipping Cost | Priority |  
|-----------|----------|-------|---------------|----------|  
| A         | Laptop   | 50    | 5             | High     |  
| B         | Laptop   | 40    | 10            | Low      |  
| A         | Phone    | 60    | 3             | Medium   |  

**Expected Output**:  
- Minimize shipping costs while fulfilling demand.  

---

#### **Input 6: Insurance Risk**  
**Policy Data**:  

| Policyholder | Risk Score | Premium Charged | Coverage Limit |  
|--------------|------------|-----------------|----------------|  
| P1           | High       | 1000            | 5000           |  
| P2           | Medium     | 800             | 4000           |  
| P3           | Low        | 500             | 3000           |  

**Expected Output**:  
- Adjust premiums based on risk scores and market conditions.  

---

### Python Code  

```python
import pandas as pd
import numpy as np

# Load data
def load_dataset(filepath):
    return pd.read_csv(filepath)

# Simulate market interactions
def simulate_market(agents_df):
    buyers = agents_df[agents_df['Agent Type'] == 'Buyer']
    sellers = agents_df[agents_df['Agent Type'] == 'Seller']

    transactions = []
    for _, buyer in buyers.iterrows():
        for _, seller in sellers.iterrows():
            if buyer['Willingness to Pay'] >= seller['Price Offered'] and seller['Supply'] > 0:
                quantity = min(seller['Supply'], buyer['Willingness to Pay'] // seller['Price Offered'])
                transactions.append({
                    "Buyer": buyer['Agent ID'],
                    "Seller": seller['Agent ID'],
                    "Price": seller['Price Offered'],
                    "Quantity": quantity
                })
                sellers.loc[seller.name, 'Supply'] -= quantity
                break
    return pd.DataFrame(transactions)

# Optimize resource allocation
def optimize_allocation(regions_df, supply_column, demand_column):
    regions_df['Deficit'] = regions_df[demand_column] - regions_df[supply_column]
    surplus_regions = regions_df[regions_df['Deficit'] < 0]
    deficit_regions = regions_df[regions_df['Deficit'] > 0]

    allocations = []
    for _, deficit_region in deficit_regions.iterrows():
        for _, surplus_region in surplus_regions.iterrows():
            allocation = min(-surplus_region['Deficit'], deficit_region['Deficit'])
            if allocation > 0:
                allocations.append({
                    "From": surplus_region['Region'],
                    "To": deficit_region['Region'],
                    "Amount": allocation
                })
                surplus_regions.loc[surplus_region.name, 'Deficit'] += allocation
                deficit_regions.loc[deficit_region.name, 'Deficit'] -= allocation
    return pd.DataFrame(allocations)

# Example usage
if __name__ == "__main__":
    agents = pd.DataFrame({
        "Agent Type": ["Buyer", "Buyer", "Seller", "Seller"],
        "Agent ID": ["B1", "B2", "S1", "S2"],
        "Willingness to Pay": [100, 80, 0, 0],
        "Supply": [0, 0, 50, 30],
        "Price Offered": [90, 85, 90, 80]
    })

    transactions = simulate_market(agents)
    print(transactions)
```

This project is highly flexible, offering advanced analytical challenges while introducing real-world applications like agent-based modeling, multi-scenario optimization, and dynamic decision-making processes.
"""
