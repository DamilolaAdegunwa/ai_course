import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.metrics import jaccard_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# Function to preprocess data
def preprocess_data(df, relationship_type):
    if relationship_type == "co_purchase":
        df = df.groupby(['Customer ID', 'Product ID']).size().reset_index(name='Frequency')
    elif relationship_type == "communication":
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df['Date'] = df['Timestamp'].dt.date
    elif relationship_type == "supplier_buyer":
        df['Date'] = pd.to_datetime(df['Date'])
    return df


# Function to create network
def create_network(df, nodes_col, edges_col, weight_col):
    G = nx.Graph()
    for _, row in df.iterrows():
        G.add_edge(row[nodes_col], row[edges_col], weight=row[weight_col])
    return G


# Function to calculate network metrics
def network_metrics(G):
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    return degree_centrality, betweenness_centrality


# Function to visualize network
def visualize_network(G, title="Network Graph"):
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=700, node_color="skyblue")
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.title(title)
    plt.show()


# Main function
if __name__ == "__main__":
    # Example: Co-purchase data
    data = {
        'Customer ID': ['C001', 'C001', 'C002', 'C003', 'C001'],
        'Product ID': ['P001', 'P002', 'P001', 'P003', 'P003'],
        'Amount': [20.00, 15.00, 22.00, 18.00, 25.00]
    }
    df = pd.DataFrame(data)

    # Preprocess data
    df = preprocess_data(df, relationship_type="co_purchase")

    # Create network
    G = create_network(df, 'Customer ID', 'Product ID', 'Amount')

    # Calculate metrics
    degree_centrality, betweenness_centrality = network_metrics(G)
    print("Degree Centrality:", degree_centrality)
    print("Betweenness Centrality:", betweenness_centrality)

    # Visualize network
    visualize_network(G, title="Co-Purchase Network")

# 2/2
comment = """
### Project Title: **Dynamic Network Analysis and Visualization Using Pandas**  
**File Name**: `dynamic_network_analysis_and_visualization_with_pandas.py`  

---

### Project Description  
This project uses **Pandas** to analyze dynamic relationships between entities over time and visualize these relationships using **network graphs**. The aim is to model and analyze temporal networks where nodes represent entities (e.g., individuals, companies, or products) and edges represent relationships (e.g., transactions, communications, or co-purchases). The project will involve:

1. **Dynamic Relationship Analysis**: Building temporal networks based on transactional or interaction data.
2. **Node and Edge Metrics**: Computing centrality measures, edge weights, and detecting communities.
3. **Feature Engineering**: Using temporal trends and clustering techniques to predict relationship changes.
4. **Visualization**: Visualizing temporal changes in network structure using graph plotting libraries such as **NetworkX** and **Matplotlib**.
5. **Advanced Modeling**: Predicting future relationships or node importance using machine learning.

This project is suitable for analyzing customer interactions, social networks, or supply chain relationships.

---

### Example Use Cases  
1. **Social Network Analysis**: Understanding how user interactions evolve over time in a social media platform.  
2. **Customer Purchase Behavior**: Analyzing co-purchase networks to suggest frequently bought-together products dynamically.  
3. **Supply Chain Monitoring**: Identifying critical suppliers or customers and predicting potential bottlenecks based on transactional history.

---

### Example Input(s) and Expected Output(s)

#### **Input 1: Transactional Data for Co-Purchase Network**  
| Date       | Customer ID | Product ID | Amount |  
|------------|-------------|------------|--------|  
| 2024-10-01 | C001        | P001       | 20.00  |  
| 2024-10-01 | C001        | P002       | 15.00  |  
| 2024-10-02 | C002        | P001       | 22.00  |  
| 2024-10-03 | C003        | P003       | 18.00  |  
| 2024-10-03 | C001        | P003       | 25.00  |  

**Expected Output**:  
- **Network Visualization**: A co-purchase graph where nodes are products and edges indicate co-purchases weighted by frequency.  
- **Key Metrics**: Product P001 has the highest degree centrality (frequent connections with other products).

#### **Input 2: Communication Network Data**  
| Timestamp            | Sender ID | Receiver ID | Message Length |  
|----------------------|-----------|-------------|----------------|  
| 2024-11-01 10:00:00 | U001      | U002        | 120            |  
| 2024-11-01 10:05:00 | U002      | U003        | 80             |  
| 2024-11-01 10:10:00 | U001      | U003        | 100            |  
| 2024-11-01 10:15:00 | U003      | U001        | 150            |  

**Expected Output**:  
- **Temporal Network**: A directed graph where nodes are users and edges represent communication frequency and message length.  
- **Community Detection**: Identifies {U001, U003} as a tightly connected community.

#### **Input 3: Supplier-Buyer Relationship Data**  
| Date       | Supplier ID | Buyer ID | Volume |  
|------------|-------------|----------|--------|  
| 2024-10-01 | S001        | B001     | 500    |  
| 2024-10-01 | S002        | B002     | 300    |  
| 2024-10-02 | S001        | B002     | 400    |  
| 2024-10-03 | S002        | B001     | 350    |  

**Expected Output**:  
- **Dynamic Supplier-Buyer Network**: An undirected graph with edges weighted by transaction volume.  
- **Critical Nodes**: Supplier S001 identified as the most critical node in the network based on betweenness centrality.

---

### Python Code

```python
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.metrics import jaccard_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Function to preprocess data
def preprocess_data(df, relationship_type):
    if relationship_type == "co_purchase":
        df = df.groupby(['Customer ID', 'Product ID']).size().reset_index(name='Frequency')
    elif relationship_type == "communication":
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df['Date'] = df['Timestamp'].dt.date
    elif relationship_type == "supplier_buyer":
        df['Date'] = pd.to_datetime(df['Date'])
    return df

# Function to create network
def create_network(df, nodes_col, edges_col, weight_col):
    G = nx.Graph()
    for _, row in df.iterrows():
        G.add_edge(row[nodes_col], row[edges_col], weight=row[weight_col])
    return G

# Function to calculate network metrics
def network_metrics(G):
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    return degree_centrality, betweenness_centrality

# Function to visualize network
def visualize_network(G, title="Network Graph"):
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=700, node_color="skyblue")
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.title(title)
    plt.show()

# Main function
if __name__ == "__main__":
    # Example: Co-purchase data
    data = {
        'Customer ID': ['C001', 'C001', 'C002', 'C003', 'C001'],
        'Product ID': ['P001', 'P002', 'P001', 'P003', 'P003'],
        'Amount': [20.00, 15.00, 22.00, 18.00, 25.00]
    }
    df = pd.DataFrame(data)
    
    # Preprocess data
    df = preprocess_data(df, relationship_type="co_purchase")
    
    # Create network
    G = create_network(df, 'Customer ID', 'Product ID', 'Amount')
    
    # Calculate metrics
    degree_centrality, betweenness_centrality = network_metrics(G)
    print("Degree Centrality:", degree_centrality)
    print("Betweenness Centrality:", betweenness_centrality)
    
    # Visualize network
    visualize_network(G, title="Co-Purchase Network")
```

---

This project pushes your skills in **data manipulation**, **graph theory**, and **machine learning-based predictions** for temporal networks. You can extend it further by integrating **real-time network updates** or **predictive modeling of relationship evolution**!
"""