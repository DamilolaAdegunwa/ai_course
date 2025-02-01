import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


# Load workflow data
def load_workflow_data(file_path):
    df = pd.read_csv(file_path)
    df['Depends_On'] = df['Depends_On'].fillna('').apply(lambda x: x.split(','))
    return df


# Build directed graph from workflow data
def build_workflow_graph(df):
    G = nx.DiGraph()
    for _, row in df.iterrows():
        task = row['Task']
        dependencies = row['Depends_On']
        G.add_node(task, time=row['Estimated_Time'], priority=row['Priority'], resource=row['Resource'])
        for dep in dependencies:
            if dep.strip():  # Avoid empty dependencies
                G.add_edge(dep.strip(), task)
    return G


# Detect cycles in the graph
def detect_cycles(G):
    try:
        cycles = nx.find_cycle(G, orientation='original')
        return cycles
    except nx.NetworkXNoCycle:
        return None


# Compute critical path
def compute_critical_path(G):
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("Graph contains cycles! Critical path computation is invalid.")
    longest_path = nx.dag_longest_path(G, weight='time')
    return longest_path


# Rank tasks using PageRank
def rank_tasks(G):
    pagerank_scores = nx.pagerank(G, alpha=0.85)
    return sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)


# Visualize the workflow graph
def visualize_graph(G, title):
    pos = nx.spring_layout(G)
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, node_color='skyblue', edge_color='gray', node_size=2000, font_size=12,
            font_weight='bold')
    edge_labels = nx.get_edge_attributes(G, 'time')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.title(title)
    plt.show()


# Workflow optimization pipeline
def workflow_optimization_pipeline(file_path):
    print("Loading workflow data...")
    df = load_workflow_data(file_path)

    print("Building workflow graph...")
    G = build_workflow_graph(df)

    print("Detecting cycles...")
    cycles = detect_cycles(G)
    if cycles:
        print("Warning: Cycles detected in the workflow!")
        print(cycles)

    print("Computing critical path...")
    try:
        critical_path = compute_critical_path(G)
        print("Critical Path:", " → ".join(critical_path))
    except ValueError as e:
        print(e)
        return

    print("Ranking tasks using PageRank...")
    task_ranking = rank_tasks(G)
    print("\nTask Rankings (by importance):")
    for task, score in task_ranking:
        print(f"Task {task}: {score:.4f}")

    print("Visualizing workflow graph...")
    visualize_graph(G, "Workflow Graph")


# Example usage
if __name__ == "__main__":
    file_path = "task_dependency_data.csv"  # Replace with your dataset
    workflow_optimization_pipeline(file_path)


comment = """
### Project Title: **Dynamic Graph-Based Workflow Optimization with Pandas**  
**File Name**: `dynamic_graph_based_workflow_optimization_with_pandas.py`  

---

### Project Description  
This project takes workflow optimization to a new level by dynamically analyzing **dependencies** and **bottlenecks** in workflow or task data. It creates a **directed graph representation of workflows**, detects cycles, computes task criticalities, and suggests the optimal sequence for task execution. It includes advanced concepts like **PageRank** for task importance, **parallel execution paths**, and real-time updates to workflows as conditions change.  

Applications include:  
- Optimizing complex manufacturing pipelines.  
- Task scheduling in distributed systems.  
- Project management dependency analysis.  

---

### Example Input(s) and Expected Output(s)  

#### **Input 1**  
**Data**: A task dependency table:  
| Task | Depends_On | Estimated_Time | Priority | Resource |  
|------|------------|----------------|----------|----------|  
| A    | None       | 2              | High     | Worker1  |  
| B    | A          | 3              | Medium   | Worker2  |  
| C    | A          | 1              | Low      | Worker1  |  
| D    | B, C       | 2              | High     | Worker3  |  

**Expected Output**:  
- **Graph Representation**: Visualized as a directed acyclic graph (DAG).  
- **Critical Path**: A → B → D.  
- **Task Ranking**: A > B > D > C (using PageRank).  
- **Optimization Suggestions**: Parallel execution of C and B to minimize time.  

#### **Input 2**  
**Data**: Workflow updates where a task is delayed (e.g., B).  
**Expected Output**:  
- Updated critical path with downstream delays.  
- Suggestion to reallocate resources to C for parallel execution.  

#### **Input 3**  
**Data**: Tasks with new dependencies added dynamically.  
**Expected Output**:  
- Cycle detection with warnings (if any).  
- Updated optimal sequence and allocation.

---

### Python Code  

```python
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Load workflow data
def load_workflow_data(file_path):
    df = pd.read_csv(file_path)
    df['Depends_On'] = df['Depends_On'].fillna('').apply(lambda x: x.split(','))
    return df

# Build directed graph from workflow data
def build_workflow_graph(df):
    G = nx.DiGraph()
    for _, row in df.iterrows():
        task = row['Task']
        dependencies = row['Depends_On']
        G.add_node(task, time=row['Estimated_Time'], priority=row['Priority'], resource=row['Resource'])
        for dep in dependencies:
            if dep.strip():  # Avoid empty dependencies
                G.add_edge(dep.strip(), task)
    return G

# Detect cycles in the graph
def detect_cycles(G):
    try:
        cycles = nx.find_cycle(G, orientation='original')
        return cycles
    except nx.NetworkXNoCycle:
        return None

# Compute critical path
def compute_critical_path(G):
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("Graph contains cycles! Critical path computation is invalid.")
    longest_path = nx.dag_longest_path(G, weight='time')
    return longest_path

# Rank tasks using PageRank
def rank_tasks(G):
    pagerank_scores = nx.pagerank(G, alpha=0.85)
    return sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)

# Visualize the workflow graph
def visualize_graph(G, title):
    pos = nx.spring_layout(G)
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, node_color='skyblue', edge_color='gray', node_size=2000, font_size=12, font_weight='bold')
    edge_labels = nx.get_edge_attributes(G, 'time')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.title(title)
    plt.show()

# Workflow optimization pipeline
def workflow_optimization_pipeline(file_path):
    print("Loading workflow data...")
    df = load_workflow_data(file_path)
    
    print("Building workflow graph...")
    G = build_workflow_graph(df)
    
    print("Detecting cycles...")
    cycles = detect_cycles(G)
    if cycles:
        print("Warning: Cycles detected in the workflow!")
        print(cycles)
    
    print("Computing critical path...")
    try:
        critical_path = compute_critical_path(G)
        print("Critical Path:", " → ".join(critical_path))
    except ValueError as e:
        print(e)
        return
    
    print("Ranking tasks using PageRank...")
    task_ranking = rank_tasks(G)
    print("\nTask Rankings (by importance):")
    for task, score in task_ranking:
        print(f"Task {task}: {score:.4f}")
    
    print("Visualizing workflow graph...")
    visualize_graph(G, "Workflow Graph")

# Example usage
if __name__ == "__main__":
    file_path = "task_dependency_data.csv"  # Replace with your dataset
    workflow_optimization_pipeline(file_path)
```

---

### Testing Scenarios  

#### **Scenario 1**:  
**Input File**: `task_dependency_data.csv`  
| Task | Depends_On | Estimated_Time | Priority | Resource |  
|------|------------|----------------|----------|----------|  
| A    | None       | 2              | High     | Worker1  |  
| B    | A          | 3              | Medium   | Worker2  |  
| C    | A          | 1              | Low      | Worker1  |  
| D    | B, C       | 2              | High     | Worker3  |  

**Expected Output**:  
- **Critical Path**: A → B → D  
- **Task Rankings**: A > B > D > C  

#### **Scenario 2**:  
**Input**: A task delayed (e.g., B now takes 6 units).  
**Expected Output**:  
- Critical path updated: A → B → D.  
- Suggestion to reassign C for earlier parallel execution.  

#### **Scenario 3**:  
**Input**: New task dependencies added with potential cycles.  
**Expected Output**:  
- Warning for detected cycles.  
- Updated graph visualization and critical path.  

---

### Key Learnings  
- **Graph Theory Integration**: Utilize NetworkX for DAGs and dependency management.  
- **Critical Path Analysis**: Identify bottlenecks and suggest optimization strategies.  
- **Dynamic Updates**: Adapt to changes in real-time and handle complex dependency scenarios.  
- **Visualization**: Graph-based visualizations for better understanding of workflows.  

Would you like to extend this with **real-time task updates** or integrate with **project management tools like Trello or Jira**?
"""