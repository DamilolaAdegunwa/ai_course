import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Function to preprocess data for network creation
def preprocess_network_data(df, source_col, target_col, weight_col=None):
    df = df[[source_col, target_col, weight_col]].dropna() if weight_col else df[[source_col, target_col]]
    df = df.groupby([source_col, target_col]).size().reset_index(name='Weight') if not weight_col else df
    return df

# Function to create a graph
def create_graph(df, source_col, target_col, weight_col=None):
    G = nx.DiGraph()  # Directed graph
    for _, row in df.iterrows():
        G.add_edge(row[source_col], row[target_col], weight=row.get(weight_col, 1))
    return G

# Function to calculate graph metrics
def calculate_graph_metrics(G):
    centrality = nx.degree_centrality(G)
    clustering = nx.clustering(G.to_undirected())
    return centrality, clustering

# Function to visualize the graph
def visualize_graph(G, title):
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=700, node_color='skyblue', edge_color='gray')
    plt.title(title)
    plt.show()

# Main function
if __name__ == "__main__":
    # Example: Social Network Data
    social_data = {
        'UserA': ['Alice', 'Bob', 'Alice', 'Charlie'],
        'UserB': ['Bob', 'Charlie', 'Charlie', 'Alice'],
        'Interaction_Type': ['Message', 'Like', 'Follow', 'Reply'],
        'Timestamp': ['2024-01-01 12:00:00', '2024-01-01 13:00:00', '2024-01-02 14:00:00', '2024-01-02 15:00:00']
    }
    df = pd.DataFrame(social_data)

    # Preprocess the data
    network_df = preprocess_network_data(df, 'UserA', 'UserB')

    # Create graph
    G = create_graph(network_df, 'UserA', 'UserB')

    # Calculate graph metrics
    centrality, clustering = calculate_graph_metrics(G)
    print("Centrality Scores:", centrality)
    print("Clustering Coefficients:", clustering)

    # Visualize the graph
    visualize_graph(G, "Social Network Graph")


comment = """
### Project Title: **Dynamic Network Graphs and Insights Generation with Pandas**  
**File Name**: `dynamic_network_graphs_and_insights_with_pandas.py`  

---

### Project Description  
In this advanced project, we explore the use of **Pandas** to process, analyze, and visualize data as **dynamic network graphs**. By modeling relationships and interactions in data (e.g., social networks, logistics chains, or communication patterns), the project involves:  

1. **Relationship Extraction**: Using Pandas to identify and aggregate relationships within data.  
2. **Graph Construction**: Transforming datasets into nodes and edges for graph analysis.  
3. **Network Metrics**: Calculating key graph metrics such as degree centrality, clustering coefficients, and shortest paths.  
4. **Dynamic Visualizations**: Generating interactive network visualizations using tools like **NetworkX** and **Matplotlib**.  
5. **Insights Generation**: Providing actionable insights from the network’s structure and dynamics.  

---

### Example Use Cases  
1. **Social Media Analysis**: Analyze the connections between users, identify influencers, and detect clusters in social networks.  
2. **Transportation Optimization**: Model logistics or traffic flows and identify bottlenecks in the system.  
3. **Fraud Detection**: Highlight anomalous patterns in transactional data by analyzing connectivity.  

---

### Example Input(s) and Expected Output(s)  

#### **Input 1: Social Network Data**  
| UserA   | UserB   | Interaction_Type | Timestamp           |  
|---------|---------|------------------|---------------------|  
| Alice   | Bob     | Message          | 2024-01-01 12:00:00 |  
| Bob     | Charlie | Like             | 2024-01-01 13:00:00 |  
| Alice   | Charlie | Follow           | 2024-01-02 14:00:00 |  
| Charlie | Alice   | Reply            | 2024-01-02 15:00:00 |  

**Expected Output**:  
- A graph showing connections between users (nodes).  
- Centrality scores highlighting key influencers.  
- Cluster analysis identifying tightly connected groups.  

#### **Input 2: Supply Chain Data**  
| Source    | Destination | Product    | Volume |  
|-----------|-------------|------------|--------|  
| Factory A | Warehouse X | Electronics| 500    |  
| Factory A | Warehouse Y | Textiles   | 300    |  
| Warehouse X | Retail Z  | Electronics| 200    |  
| Warehouse Y | Retail Z  | Textiles   | 150    |  

**Expected Output**:  
- A directed graph showing the flow of goods.  
- Identification of bottlenecks in the supply chain.  
- Visualization of high-volume paths.  

#### **Input 3: Transactional Data**  
| Sender   | Receiver   | Amount | Timestamp           |  
|----------|------------|--------|---------------------|  
| Account1 | Account2   | 100    | 2024-06-01 09:00:00 |  
| Account2 | Account3   | 150    | 2024-06-01 10:00:00 |  
| Account3 | Account1   | 200    | 2024-06-01 11:00:00 |  
| Account1 | Account4   | 300    | 2024-06-01 12:00:00 |  

**Expected Output**:  
- A network visualization of transactions.  
- Detection of cyclical patterns in transactions.  
- Identification of suspiciously high transaction paths.  

---

### Python Code  

```python
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Function to preprocess data for network creation
def preprocess_network_data(df, source_col, target_col, weight_col=None):
    df = df[[source_col, target_col, weight_col]].dropna() if weight_col else df[[source_col, target_col]]
    df = df.groupby([source_col, target_col]).size().reset_index(name='Weight') if not weight_col else df
    return df

# Function to create a graph
def create_graph(df, source_col, target_col, weight_col=None):
    G = nx.DiGraph()  # Directed graph
    for _, row in df.iterrows():
        G.add_edge(row[source_col], row[target_col], weight=row.get(weight_col, 1))
    return G

# Function to calculate graph metrics
def calculate_graph_metrics(G):
    centrality = nx.degree_centrality(G)
    clustering = nx.clustering(G.to_undirected())
    return centrality, clustering

# Function to visualize the graph
def visualize_graph(G, title):
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=700, node_color='skyblue', edge_color='gray')
    plt.title(title)
    plt.show()

# Main function
if __name__ == "__main__":
    # Example: Social Network Data
    social_data = {
        'UserA': ['Alice', 'Bob', 'Alice', 'Charlie'],
        'UserB': ['Bob', 'Charlie', 'Charlie', 'Alice'],
        'Interaction_Type': ['Message', 'Like', 'Follow', 'Reply'],
        'Timestamp': ['2024-01-01 12:00:00', '2024-01-01 13:00:00', '2024-01-02 14:00:00', '2024-01-02 15:00:00']
    }
    df = pd.DataFrame(social_data)

    # Preprocess the data
    network_df = preprocess_network_data(df, 'UserA', 'UserB')

    # Create graph
    G = create_graph(network_df, 'UserA', 'UserB')

    # Calculate graph metrics
    centrality, clustering = calculate_graph_metrics(G)
    print("Centrality Scores:", centrality)
    print("Clustering Coefficients:", clustering)

    # Visualize the graph
    visualize_graph(G, "Social Network Graph")
```

---

### How This Project Advances Your Skills  
1. **Graph Theory**: Gain expertise in modeling relationships and structures using network graphs.  
2. **Dynamic Insights**: Extract meaningful metrics like centrality and clustering for actionable insights.  
3. **Interactive Visualizations**: Learn advanced visualization techniques to present network data intuitively.  
4. **Real-World Applications**: Apply graph-based methods to diverse domains like social media, logistics, and fraud detection.  
5. **Data Engineering**: Handle preprocessing and feature engineering for relationship modeling.  

Enhance this further with **dynamic graphs** using animations or **real-time data updates**. Explore integrations with graph databases like **Neo4j** for scalability.
"""