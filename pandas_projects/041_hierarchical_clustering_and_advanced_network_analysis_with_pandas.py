import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import networkx as nx
import matplotlib.pyplot as plt


# Step 1: Feature Correlation and Clustering
def hierarchical_clustering(data, method='ward', metric='euclidean', threshold=1.5):
    # Perform hierarchical clustering
    linkage_matrix = linkage(data, method=method, metric=metric)
    clusters = fcluster(linkage_matrix, t=threshold, criterion='distance')
    return linkage_matrix, clusters


# Step 2: Network Graph Construction
def create_network(data, node_col1, node_col2, weight_col):
    G = nx.Graph()
    for _, row in data.iterrows():
        G.add_edge(row[node_col1], row[node_col2], weight=row[weight_col])
    return G


# Step 3: Network Metrics
def analyze_network(G):
    metrics = {
        "degree_centrality": nx.degree_centrality(G),
        "clustering_coefficient": nx.clustering(G),
        "modularity_communities": list(nx.community.greedy_modularity_communities(G))
    }
    return metrics


# Step 4: Visualization
def visualize_network(G):
    pos = nx.spring_layout(G)
    edge_weights = nx.get_edge_attributes(G, 'weight')
    nx.draw(G, pos, with_labels=True, node_size=700, node_color="skyblue", font_size=10, font_weight="bold")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_weights)
    plt.show()


# Example Usage
if __name__ == "__main__":
    # Example: User Interaction Data
    data = pd.DataFrame({
        'User1': ['A', 'A', 'B', 'C'],
        'User2': ['B', 'C', 'D', 'D'],
        'Interaction Frequency': [15, 10, 5, 8]
    })

    # Construct Network
    G = create_network(data, 'User1', 'User2', 'Interaction Frequency')

    # Analyze Network
    metrics = analyze_network(G)
    print("Network Metrics:", metrics)

    # Visualize Network
    visualize_network(G)

    # Clustering Example
    interaction_matrix = pd.pivot_table(data, values='Interaction Frequency', index='User1', columns='User2',
                                        fill_value=0)
    interaction_array = interaction_matrix.values
    linkage_matrix, clusters = hierarchical_clustering(interaction_array, method='ward', threshold=7)

    print(f"Clusters: {clusters}")
    dendrogram(linkage_matrix, labels=interaction_matrix.index.tolist())
    plt.title("Hierarchical Clustering Dendrogram")
    plt.show()


comment = """
### Project Title: **Hierarchical Clustering and Advanced Network Analysis with Pandas**  
**File Name**: `hierarchical_clustering_and_advanced_network_analysis_with_pandas.py`  

---

### Project Description  

This project delves into **advanced hierarchical clustering and network analysis** using **Pandas** for feature-rich datasets. It incorporates:  

1. **Feature Correlation Analysis**: Dynamically compute and filter correlations to identify relationships between features.  
2. **Hierarchical Clustering**: Apply advanced clustering methods to group similar data points.  
3. **Network Graph Construction**: Convert clustered data into graph structures for network visualization and analysis.  
4. **Community Detection**: Detect communities within networks using modularity optimization.  
5. **Advanced Metric Evaluation**: Use centrality, clustering coefficients, and edge-weight analysis for insights.  

This is applicable in domains such as **social network analysis**, **biological systems modeling**, **market segmentation**, and **supply chain analysis**.  

---

### Example Use Cases  

1. **Social Media Analysis**: Group users based on interaction patterns and detect influential users within communities.  
2. **Gene Similarity Analysis**: Cluster genes with similar expressions and visualize genetic pathways.  
3. **Market Segmentation**: Group customers based on purchase behavior and detect community-based marketing opportunities.  
4. **Supply Chain Optimization**: Analyze product flow networks and identify bottlenecks.  

---

### Example Input(s) and Expected Output(s)  

#### **Input 1: User Interaction Data**  
| User1 | User2 | Interaction Frequency |  
|-------|-------|------------------------|  
| A     | B     | 15                     |  
| A     | C     | 10                     |  
| B     | D     | 5                      |  
| C     | D     | 8                      |  

**Expected Output**:  
- Clusters of users based on interactions: `{'Cluster 1': ['A', 'B'], 'Cluster 2': ['C', 'D']}`.  
- Network graph with centrality metrics and community detection.  

---

#### **Input 2: Gene Expression Data**  
| Gene1 | Gene2 | Correlation |  
|-------|-------|-------------|  
| G1    | G2    | 0.85        |  
| G1    | G3    | 0.40        |  
| G2    | G4    | 0.90        |  
| G3    | G4    | 0.50        |  

**Expected Output**:  
- Clusters of genes with high correlation: `{'Cluster 1': ['G1', 'G2'], 'Cluster 2': ['G3', 'G4']}`.  
- Network graph of genetic pathways.  

---

#### **Input 3: Customer Purchase Data**  
| Customer1 | Customer2 | Shared Products |  
|-----------|-----------|-----------------|  
| Alice     | Bob       | 5               |  
| Alice     | Charlie   | 3               |  
| Bob       | Charlie   | 4               |  
| Charlie   | David     | 2               |  

**Expected Output**:  
- Clusters of customers with shared product interests: `{'Cluster 1': ['Alice', 'Bob'], 'Cluster 2': ['Charlie', 'David']}`.  
- Centrality metrics highlighting the most influential customer.  

---

#### **Input 4: Product Flow Data**  
| Product1 | Product2 | Transport Volume |  
|----------|----------|------------------|  
| P1       | P2       | 100              |  
| P1       | P3       | 70               |  
| P2       | P4       | 50               |  
| P3       | P4       | 80               |  

**Expected Output**:  
- Clusters of products based on transport volume: `{'Cluster 1': ['P1', 'P2'], 'Cluster 2': ['P3', 'P4']}`.  
- Network graph with edge weights representing transport volume.  

---

### Python Code  

```python
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import networkx as nx
import matplotlib.pyplot as plt

# Step 1: Feature Correlation and Clustering
def hierarchical_clustering(data, method='ward', metric='euclidean', threshold=1.5):
    # Perform hierarchical clustering
    linkage_matrix = linkage(data, method=method, metric=metric)
    clusters = fcluster(linkage_matrix, t=threshold, criterion='distance')
    return linkage_matrix, clusters

# Step 2: Network Graph Construction
def create_network(data, node_col1, node_col2, weight_col):
    G = nx.Graph()
    for _, row in data.iterrows():
        G.add_edge(row[node_col1], row[node_col2], weight=row[weight_col])
    return G

# Step 3: Network Metrics
def analyze_network(G):
    metrics = {
        "degree_centrality": nx.degree_centrality(G),
        "clustering_coefficient": nx.clustering(G),
        "modularity_communities": list(nx.community.greedy_modularity_communities(G))
    }
    return metrics

# Step 4: Visualization
def visualize_network(G):
    pos = nx.spring_layout(G)
    edge_weights = nx.get_edge_attributes(G, 'weight')
    nx.draw(G, pos, with_labels=True, node_size=700, node_color="skyblue", font_size=10, font_weight="bold")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_weights)
    plt.show()

# Example Usage
if __name__ == "__main__":
    # Example: User Interaction Data
    data = pd.DataFrame({
        'User1': ['A', 'A', 'B', 'C'],
        'User2': ['B', 'C', 'D', 'D'],
        'Interaction Frequency': [15, 10, 5, 8]
    })
    
    # Construct Network
    G = create_network(data, 'User1', 'User2', 'Interaction Frequency')
    
    # Analyze Network
    metrics = analyze_network(G)
    print("Network Metrics:", metrics)
    
    # Visualize Network
    visualize_network(G)
    
    # Clustering Example
    interaction_matrix = pd.pivot_table(data, values='Interaction Frequency', index='User1', columns='User2', fill_value=0)
    interaction_array = interaction_matrix.values
    linkage_matrix, clusters = hierarchical_clustering(interaction_array, method='ward', threshold=7)
    
    print(f"Clusters: {clusters}")
    dendrogram(linkage_matrix, labels=interaction_matrix.index.tolist())
    plt.title("Hierarchical Clustering Dendrogram")
    plt.show()
```

This project combines **clustering and network theory** for high-level data analytics. It supports real-world applications like **complex system modeling, market segmentation, and pathway discovery**.
"""