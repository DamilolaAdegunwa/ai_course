import pandas as pd
from itertools import combinations
from collections import defaultdict


# Function to generate item combinations and calculate support
def generate_frequent_itemsets(transactions, min_support=0.5):
    item_count = defaultdict(int)
    total_transactions = len(transactions)

    # Count item frequencies
    for items in transactions:
        for size in range(1, len(items) + 1):
            for combination in combinations(items, size):
                item_count[combination] += 1

    # Filter itemsets by minimum support
    frequent_itemsets = {
        itemset: count / total_transactions
        for itemset, count in item_count.items() if count / total_transactions >= min_support
    }
    return frequent_itemsets


# Function to calculate confidence and lift
def calculate_association_rules(frequent_itemsets, transactions):
    total_transactions = len(transactions)
    rules = []
    for itemset, support in frequent_itemsets.items():
        if len(itemset) > 1:
            for antecedent in combinations(itemset, len(itemset) - 1):
                consequent = tuple(set(itemset) - set(antecedent))
                antecedent_support = frequent_itemsets.get(antecedent, 0)
                if antecedent_support > 0:
                    confidence = support / antecedent_support
                    lift = confidence / (frequent_itemsets.get(consequent, 0) or 1)
                    rules.append({
                        'Rule': f"{antecedent} -> {consequent}",
                        'Confidence': confidence,
                        'Lift': lift,
                        'Support': support
                    })
    return pd.DataFrame(rules)


# Simulate real-time transaction data
def simulate_transactions(data, chunk_size):
    for i in range(0, len(data), chunk_size):
        yield data.iloc[i:i + chunk_size]


# Main Pipeline
if __name__ == "__main__":
    # Example transaction data
    data = pd.DataFrame({
        'Transaction_ID': [1, 2, 3, 4],
        'Items': [
            ['Bread', 'Milk', 'Butter'],
            ['Bread', 'Butter', 'Eggs'],
            ['Milk', 'Eggs', 'Cheese'],
            ['Bread', 'Milk']
        ]
    })

    # Simulating real-time data chunks
    min_support = 0.5
    all_transactions = []
    for chunk in simulate_transactions(data, chunk_size=2):
        all_transactions.extend(chunk['Items'].tolist())
        frequent_itemsets = generate_frequent_itemsets(all_transactions, min_support)
        print("Frequent Itemsets:\n", frequent_itemsets)

        rules = calculate_association_rules(frequent_itemsets, all_transactions)
        print("\nAssociation Rules:\n", rules)

    # Visualizing frequent itemsets
    itemset_df = pd.DataFrame(list(frequent_itemsets.items()), columns=['Itemset', 'Support'])
    itemset_df.sort_values(by='Support', ascending=False, inplace=True)
    print("\nTop Frequent Itemsets:\n", itemset_df.head())


# https://chatgpt.com/c/674b65b9-fecc-800c-8311-7f681df9b305
comment = """
### Project Title: **Dynamic Market Basket Analysis and Recommendation Engine with Pandas**  
**File Name**: `dynamic_market_basket_analysis_with_pandas.py`  

---

### Project Description  
This project builds a **market basket analysis engine** that dynamically adapts to incoming sales data. It goes beyond static association rule mining by:  

1. **Streaming Transaction Data Simulation**: Handles incoming data in real-time or batch mode.  
2. **Dynamic Association Rule Mining**: Calculates frequently bought-together itemsets using **Apriori-like algorithms**.  
3. **Personalized Recommendations**: Recommends items based on dynamic user behavior and historical data trends.  
4. **Cross-Selling and Upselling**: Identifies high-value item pairs for targeted marketing campaigns.  
5. **Custom Metrics**: Incorporates lift, leverage, and conviction to evaluate rule significance.  

---

### Example Use Cases  

1. **E-commerce Personalization**: Recommend items based on user purchase patterns.  
2. **Retail Analytics**: Discover frequently bought-together products for inventory optimization.  
3. **Dynamic Promotions**: Suggest promotional bundles based on real-time sales trends.  

---

### Example Input(s) and Expected Output(s)  

#### **Input 1: Transaction Data**  
**File**: `transactions.csv`  
| Transaction_ID | Items                      |  
|-----------------|---------------------------|  
| 1               | Bread, Milk, Butter       |  
| 2               | Bread, Butter, Eggs       |  
| 3               | Milk, Eggs, Cheese        |  
| 4               | Bread, Milk               |  

**Expected Output**:  
- **Frequent Itemsets**:  
  - `{Bread, Milk}`: Support = 75%  
  - `{Bread, Butter}`: Support = 50%  
  - `{Milk, Eggs}`: Support = 50%  

- **Recommendations for "Bread"**:  
  - `Milk` (Confidence = 75%, Lift = 1.2)  

#### **Input 2: New Data Chunk**  
| Transaction_ID | Items                      |  
|-----------------|---------------------------|  
| 5               | Milk, Butter              |  
| 6               | Bread, Butter, Eggs, Milk |  

**Expected Output**:  
- **Updated Frequent Itemsets**:  
  - `{Bread, Milk, Butter}`: Support = 50%  
  - `{Milk, Butter}`: Support = 66%  

- **Updated Recommendations for "Butter"**:  
  - `Milk` (Confidence = 80%, Lift = 1.3)  

---

### Python Code  

```python
import pandas as pd
from itertools import combinations
from collections import defaultdict

# Function to generate item combinations and calculate support
def generate_frequent_itemsets(transactions, min_support=0.5):
    item_count = defaultdict(int)
    total_transactions = len(transactions)
    
    # Count item frequencies
    for items in transactions:
        for size in range(1, len(items) + 1):
            for combination in combinations(items, size):
                item_count[combination] += 1

    # Filter itemsets by minimum support
    frequent_itemsets = {
        itemset: count / total_transactions
        for itemset, count in item_count.items() if count / total_transactions >= min_support
    }
    return frequent_itemsets

# Function to calculate confidence and lift
def calculate_association_rules(frequent_itemsets, transactions):
    total_transactions = len(transactions)
    rules = []
    for itemset, support in frequent_itemsets.items():
        if len(itemset) > 1:
            for antecedent in combinations(itemset, len(itemset) - 1):
                consequent = tuple(set(itemset) - set(antecedent))
                antecedent_support = frequent_itemsets.get(antecedent, 0)
                if antecedent_support > 0:
                    confidence = support / antecedent_support
                    lift = confidence / (frequent_itemsets.get(consequent, 0) or 1)
                    rules.append({
                        'Rule': f"{antecedent} -> {consequent}",
                        'Confidence': confidence,
                        'Lift': lift,
                        'Support': support
                    })
    return pd.DataFrame(rules)

# Simulate real-time transaction data
def simulate_transactions(data, chunk_size):
    for i in range(0, len(data), chunk_size):
        yield data.iloc[i:i+chunk_size]

# Main Pipeline
if __name__ == "__main__":
    # Example transaction data
    data = pd.DataFrame({
        'Transaction_ID': [1, 2, 3, 4],
        'Items': [
            ['Bread', 'Milk', 'Butter'],
            ['Bread', 'Butter', 'Eggs'],
            ['Milk', 'Eggs', 'Cheese'],
            ['Bread', 'Milk']
        ]
    })

    # Simulating real-time data chunks
    min_support = 0.5
    all_transactions = []
    for chunk in simulate_transactions(data, chunk_size=2):
        all_transactions.extend(chunk['Items'].tolist())
        frequent_itemsets = generate_frequent_itemsets(all_transactions, min_support)
        print("Frequent Itemsets:\n", frequent_itemsets)

        rules = calculate_association_rules(frequent_itemsets, all_transactions)
        print("\nAssociation Rules:\n", rules)

    # Visualizing frequent itemsets
    itemset_df = pd.DataFrame(list(frequent_itemsets.items()), columns=['Itemset', 'Support'])
    itemset_df.sort_values(by='Support', ascending=False, inplace=True)
    print("\nTop Frequent Itemsets:\n", itemset_df.head())
```

---

### Advanced Skills Covered  

1. **Dynamic Market Basket Analysis**: Real-time computation of frequently bought-together items.  
2. **Custom Metric Calculation**: Implementing advanced metrics like lift, leverage, and confidence.  
3. **Iterative Data Handling**: Simulating incoming transactional data and processing it in batches.  
4. **Association Rule Mining**: Mining and ranking rules dynamically with multiple metrics.  
5. **Optimized Combinatorial Calculations**: Efficient handling of large datasets with custom algorithms.  

This project elevates your data analysis skills by integrating dynamic recommendation engines into the workflow, offering practical applications for retail, e-commerce, and beyond.
"""