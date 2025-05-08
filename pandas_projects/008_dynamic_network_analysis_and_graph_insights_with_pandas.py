import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from networkx import Graph
from pandas import DataFrame

# Load Titanic Dataset
file_path = "titanic.csv"
df = pd.read_csv(file_path)


# Example Function 1: Create Passenger Network Based on Shared Cabin
def create_cabin_network(df: DataFrame) -> Graph:
    cabin_network: Graph = nx.Graph()
    for _, group in df.groupby("Cabin"):
        passengers = group["Name"].tolist()
        for i in range(len(passengers)):
            for j in range(i + 1, len(passengers)):
                cabin_network.add_edge(passengers[i], passengers[j])
    return cabin_network


# Example Function 2: Calculate Centrality Metrics
def calculate_centrality_metrics(graph) -> DataFrame:
    centrality = nx.degree_centrality(graph)
    return pd.DataFrame(centrality.items(), columns=["Passenger", "Centrality"]).sort_values(by="Centrality", ascending=False)


# Example Function 3: Shortest Path Analysis
def shortest_path_between(graph, source, target):
    if nx.has_path(graph, source, target):
        path = nx.shortest_path(graph, source=source, target=target)
        return path
    else:
        return f"No path exists between {source} and {target}."


# Example Function 4: Visualize Network with Metrics
def visualize_network(graph, centrality=None, title="Passenger Network"):
    pos = nx.spring_layout(graph)
    plt.figure(figsize=(12, 12))
    if centrality:
        nx.draw_networkx_nodes(graph, pos, node_size=[v * 1000 for v in centrality.values()])
    else:
        nx.draw_networkx_nodes(graph, pos, node_size=300)
    nx.draw_networkx_edges(graph, pos, alpha=0.3)
    nx.draw_networkx_labels(graph, pos, font_size=10)
    plt.title(title)
    plt.show()


# Example Function 5: Community Detection
def detect_communities(graph):
    communities = nx.algorithms.community.greedy_modularity_communities(graph)
    return [list(community) for community in communities]


# Test the pipeline
if __name__ == "__main__":
    # Step 1: Create the Cabin Network
    print("Creating cabin-based passenger network...")
    cabin_network = create_cabin_network(df)
    print(f"Number of nodes: {len(cabin_network.nodes)}")
    print(f"Number of edges: {len(cabin_network.edges)}")

    # Step 2: Calculate Centrality
    print("Calculating centrality metrics...")
    centrality_df = calculate_centrality_metrics(cabin_network)
    print("Top Passengers by Centrality:")
    print(centrality_df.head())

    # Step 3: Shortest Path Analysis
    source, target = centrality_df["Passenger"].iloc[0], centrality_df["Passenger"].iloc[1]
    print(f"Finding shortest path between {source} and {target}...")
    path = shortest_path_between(cabin_network, source, target)
    print("Shortest Path:")
    print(path)

    # Step 4: Visualize Network
    print("Visualizing the network...")
    visualize_network(cabin_network, centrality=nx.degree_centrality(cabin_network))

    # Step 5: Detect Communities
    print("Detecting communities in the network...")
    communities = detect_communities(cabin_network)
    print(f"Number of communities detected: {len(communities)}")
    print("Sample community:")
    print(communities[0])


comment = """
### Project Title: Dynamic Network Analysis and Graph-Based Insights with Pandas  
**File Name**: `dynamic_network_analysis_and_graph_insights_with_pandas.py`  

---

### Project Description  
This project pushes your understanding of **Pandas** by incorporating network analysis techniques to analyze and visualize relationships in datasets. Using Pandas in combination with **NetworkX** and **Matplotlib**, you will:  

1. Construct dynamic networks from structured tabular data.  
2. Perform advanced graph-based operations, such as centrality measures, shortest paths, and community detection.  
3. Analyze temporal relationships in dynamic networks.  
4. Visualize the networks and key metrics dynamically.  

The dataset can represent real-world relationships, such as social connections, transportation systems, or trade networks. We'll use the **Titanic dataset** you uploaded to create and analyze passenger relationship networks.  

---

### Example Input(s) and Expected Output(s)  

#### **Input 1**:  
**Dataset**: Titanic dataset.  
**Task**: Create a network where passengers are connected if they share the same cabin.  
**Expected Output**:  
- Graph of passenger relationships with clusters indicating shared cabins.  

#### **Input 2**:  
**Dataset**: Titanic dataset.  
**Task**: Calculate the centrality (importance) of passengers based on shared family names.  
**Expected Output**:  
- A table of passengers ranked by their centrality scores.  

#### **Input 3**:  
**Dataset**: Titanic dataset.  
**Task**: Visualize the shortest path between two selected passengers in the network.  
**Expected Output**:  
- A graph highlighting the shortest path between the two selected passengers.  

---

### Python Code  

```python
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Load Titanic Dataset
file_path = "titanic.csv"
df = pd.read_csv(file_path)

# Example Function 1: Create Passenger Network Based on Shared Cabin
def create_cabin_network(df):
    cabin_network = nx.Graph()
    for _, group in df.groupby("Cabin"):
        passengers = group["Name"].tolist()
        for i in range(len(passengers)):
            for j in range(i + 1, len(passengers)):
                cabin_network.add_edge(passengers[i], passengers[j])
    return cabin_network

# Example Function 2: Calculate Centrality Metrics
def calculate_centrality_metrics(graph):
    centrality = nx.degree_centrality(graph)
    return pd.DataFrame(centrality.items(), columns=["Passenger", "Centrality"]).sort_values(by="Centrality", ascending=False)

# Example Function 3: Shortest Path Analysis
def shortest_path_between(graph, source, target):
    if nx.has_path(graph, source, target):
        path = nx.shortest_path(graph, source=source, target=target)
        return path
    else:
        return f"No path exists between {source} and {target}."

# Example Function 4: Visualize Network with Metrics
def visualize_network(graph, centrality=None, title="Passenger Network"):
    pos = nx.spring_layout(graph)
    plt.figure(figsize=(12, 12))
    if centrality:
        nx.draw_networkx_nodes(graph, pos, node_size=[v * 1000 for v in centrality.values()])
    else:
        nx.draw_networkx_nodes(graph, pos, node_size=300)
    nx.draw_networkx_edges(graph, pos, alpha=0.3)
    nx.draw_networkx_labels(graph, pos, font_size=10)
    plt.title(title)
    plt.show()

# Example Function 5: Community Detection
def detect_communities(graph):
    communities = nx.algorithms.community.greedy_modularity_communities(graph)
    return [list(community) for community in communities]

# Test the pipeline
if __name__ == "__main__":
    # Step 1: Create the Cabin Network
    print("Creating cabin-based passenger network...")
    cabin_network = create_cabin_network(df)
    print(f"Number of nodes: {len(cabin_network.nodes)}")
    print(f"Number of edges: {len(cabin_network.edges)}")
    
    # Step 2: Calculate Centrality
    print("Calculating centrality metrics...")
    centrality_df = calculate_centrality_metrics(cabin_network)
    print("Top Passengers by Centrality:")
    print(centrality_df.head())
    
    # Step 3: Shortest Path Analysis
    source, target = centrality_df["Passenger"].iloc[0], centrality_df["Passenger"].iloc[1]
    print(f"Finding shortest path between {source} and {target}...")
    path = shortest_path_between(cabin_network, source, target)
    print("Shortest Path:")
    print(path)
    
    # Step 4: Visualize Network
    print("Visualizing the network...")
    visualize_network(cabin_network, centrality=nx.degree_centrality(cabin_network))
    
    # Step 5: Detect Communities
    print("Detecting communities in the network...")
    communities = detect_communities(cabin_network)
    print(f"Number of communities detected: {len(communities)}")
    print("Sample community:")
    print(communities[0])
```

---

### Testing Scenarios  

#### **Scenario 1**:  
**Dataset**: Titanic dataset.  
**Task**: Build a network based on passengers sharing the same ticket number.  
**Expected Output**:  
- A graph with edges connecting passengers who share a ticket.  

#### **Scenario 2**:  
**Dataset**: Titanic dataset.  
**Task**: Visualize communities in the cabin-based passenger network.  
**Expected Output**:  
- A graph with nodes colored based on their community.  

#### **Scenario 3**:  
**Dataset**: Titanic dataset.  
**Task**: Identify the passenger with the highest centrality score in the cabin network.  
**Expected Output**:  
- A table showing the passenger with the highest score, indicating their importance in the network.  

---

### Key Learnings  
- **Network Construction**: Learn to represent relationships using graph-based methods.  
- **Centrality and Influence**: Identify key players in a network using centrality metrics.  
- **Community Detection**: Explore how clusters form naturally in networks.  
- **Visualization**: Present complex relationships graphically using libraries like NetworkX and Matplotlib.  

Would you like to extend this with advanced algorithms or additional datasets?
"""
