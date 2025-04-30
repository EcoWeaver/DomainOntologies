"""
Ontology visualization and querying demonstration script.

This script demonstrates how to visualize and query the INBIO ontology
using Neo4j and matplotlib for visualization.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from neo4j import GraphDatabase
import logging
from typing import Dict, List, Tuple, Optional, Set

# Import utility modules
from config import get_neo4j_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def connect_to_neo4j():
    """
    Connect to Neo4j database.
    
    Returns:
        Neo4j driver instance
    """
    neo4j_config = get_neo4j_config()
    
    uri = neo4j_config['uri']
    username = neo4j_config['username']
    password = neo4j_config['password']
    
    try:
        driver = GraphDatabase.driver(uri, auth=(username, password))
        driver.verify_connectivity()
        logger.info("Successfully connected to Neo4j database")
        return driver
    except Exception as e:
        logger.error(f"Error connecting to Neo4j: {e}")
        raise

def get_ontology_classes(driver, limit: int = 100) -> List[Dict]:
    """
    Get classes from the ontology.
    
    Args:
        driver: Neo4j driver instance
        limit: Maximum number of classes to retrieve
        
    Returns:
        List of dictionaries containing class information
    """
    query = """
    MATCH (c:Class)
    RETURN c.name AS name, c.uri AS uri
    LIMIT $limit
    """
    
    try:
        with driver.session() as session:
            result = session.run(query, limit=limit)
            classes = [dict(record) for record in result]
            logger.info(f"Retrieved {len(classes)} classes from ontology")
            return classes
    except Exception as e:
        logger.error(f"Error retrieving ontology classes: {e}")
        return []

def get_class_hierarchy(driver, root_class: Optional[str] = None, depth: int = 3) -> List[Dict]:
    """
    Get class hierarchy from the ontology.
    
    Args:
        driver: Neo4j driver instance
        root_class: Root class to start from (optional)
        depth: Maximum depth of hierarchy to retrieve
        
    Returns:
        List of dictionaries containing class hierarchy information
    """
    if root_class:
        # Query starting from a specific root class
        query = """
        MATCH path = (c:Class {name: $root_class})-[:SCO*1..{depth}]->(parent:Class)
        RETURN c.name AS child, parent.name AS parent, length(path) AS depth
        UNION
        MATCH path = (c:Class)-[:SCO*1..{depth}]->(parent:Class {name: $root_class})
        RETURN c.name AS child, parent.name AS parent, length(path) AS depth
        """
        params = {"root_class": root_class, "depth": depth}
    else:
        # Query for the entire hierarchy
        query = """
        MATCH path = (c:Class)-[:SCO]->(parent:Class)
        RETURN c.name AS child, parent.name AS parent, 1 AS depth
        LIMIT 1000
        """
        params = {}
    
    try:
        with driver.session() as session:
            result = session.run(query, **params)
            hierarchy = [dict(record) for record in result]
            logger.info(f"Retrieved {len(hierarchy)} class hierarchy relationships")
            return hierarchy
    except Exception as e:
        logger.error(f"Error retrieving class hierarchy: {e}")
        return []

def get_class_relationships(driver, class_name: str) -> List[Dict]:
    """
    Get relationships for a specific class.
    
    Args:
        driver: Neo4j driver instance
        class_name: Name of the class to get relationships for
        
    Returns:
        List of dictionaries containing relationship information
    """
    query = """
    MATCH (c:Class {name: $class_name})-[r]->(target)
    RETURN type(r) AS relationship_type, target.name AS target_name, labels(target) AS target_labels
    UNION
    MATCH (source)-[r]->(c:Class {name: $class_name})
    RETURN type(r) AS relationship_type, source.name AS source_name, labels(source) AS source_labels
    """
    
    try:
        with driver.session() as session:
            result = session.run(query, class_name=class_name)
            relationships = [dict(record) for record in result]
            logger.info(f"Retrieved {len(relationships)} relationships for class '{class_name}'")
            return relationships
    except Exception as e:
        logger.error(f"Error retrieving relationships for class '{class_name}': {e}")
        return []

def visualize_class_hierarchy(hierarchy: List[Dict], output_file: str = 'class_hierarchy.png'):
    """
    Visualize class hierarchy using NetworkX and matplotlib.
    
    Args:
        hierarchy: List of dictionaries containing class hierarchy information
        output_file: Output file path for the visualization
    """
    G = nx.DiGraph()
    
    # Add nodes and edges
    for relation in hierarchy:
        child = relation['child']
        parent = relation['parent']
        G.add_node(child)
        G.add_node(parent)
        G.add_edge(child, parent)
    
    # Set up the plot
    plt.figure(figsize=(12, 10))
    
    # Use hierarchical layout
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    
    # Draw the graph
    nx.draw(
        G, pos, 
        with_labels=True, 
        node_color='skyblue', 
        node_size=1500, 
        edge_color='gray',
        arrows=True,
        arrowsize=20,
        font_size=10,
        font_weight='bold'
    )
    
    # Save the visualization
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Class hierarchy visualization saved to {output_file}")

def visualize_class_network(driver, central_class: str, depth: int = 2, output_file: str = 'class_network.png'):
    """
    Visualize a network of classes centered around a specific class.
    
    Args:
        driver: Neo4j driver instance
        central_class: Central class for the visualization
        depth: Maximum path length to include
        output_file: Output file path for the visualization
    """
    # Query to get the network around the central class
    query = """
    MATCH path = (c:Class {name: $central_class})-[r*1..{depth}]-(related:Class)
    RETURN c.name AS source, type(last(r)) AS relationship, related.name AS target
    UNION
    MATCH path = (related:Class)-[r*1..{depth}]-(c:Class {name: $central_class})
    RETURN related.name AS source, type(last(r)) AS relationship, c.name AS target
    """
    
    try:
        with driver.session() as session:
            result = session.run(query, central_class=central_class, depth=depth)
            relationships = [dict(record) for record in result]
            
        if not relationships:
            logger.warning(f"No relationships found for class '{central_class}' with depth {depth}")
            return
            
        # Create a directed graph
        G = nx.DiGraph()
        
        # Add nodes and edges
        for rel in relationships:
            source = rel['source']
            target = rel['target']
            relationship = rel['relationship']
            
            G.add_node(source)
            G.add_node(target)
            G.add_edge(source, target, label=relationship)
        
        # Set up the plot
        plt.figure(figsize=(14, 12))
        
        # Use a layout that spreads nodes
        pos = nx.spring_layout(G, k=0.3, iterations=50, seed=42)
        
        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos,
            node_color='lightblue',
            node_size=1000,
            alpha=0.8
        )
        
        # Highlight the central class
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=[central_class],
            node_color='red',
            node_size=1500,
            alpha=0.9
        )
        
        # Draw edges
        nx.draw_networkx_edges(
            G, pos,
            width=1.0,
            alpha=0.5,
            arrows=True,
            arrowsize=15
        )
        
        # Draw labels
        nx.draw_networkx_labels(
            G, pos,
            font_size=10,
            font_weight='bold'
        )
        
        # Draw edge labels (relationship types)
        edge_labels = {(u, v): d['label'] for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(
            G, pos,
            edge_labels=edge_labels,
            font_size=8
        )
        
        # Set title
        plt.title(f"Class Network Around '{central_class}'", fontsize=16)
        
        # Remove axis
        plt.axis('off')
        
        # Save the visualization
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Class network visualization saved to {output_file}")
        
    except Exception as e:
        logger.error(f"Error visualizing class network: {e}")

def export_ontology_to_csv(driver, output_dir: str = 'ontology_export'):
    """
    Export ontology data to CSV files.
    
    Args:
        driver: Neo4j driver instance
        output_dir: Directory to save CSV files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Export classes
    classes_query = """
    MATCH (c:Class)
    RETURN c.name AS name, c.uri AS uri
    """
    
    # Export relationships
    relationships_query = """
    MATCH (c1:Class)-[r]->(c2:Class)
    RETURN c1.name AS source, type(r) AS relationship, c2.name AS target
    """
    
    try:
        with driver.session() as session:
            # Export classes
            classes_result = session.run(classes_query)
            classes_df = pd.DataFrame([dict(record) for record in classes_result])
            classes_file = os.path.join(output_dir, 'ontology_classes.csv')
            classes_df.to_csv(classes_file, index=False)
            logger.info(f"Exported {len(classes_df)} classes to {classes_file}")
            
            # Export relationships
            relationships_result = session.run(relationships_query)
            relationships_df = pd.DataFrame([dict(record) for record in relationships_result])
            relationships_file = os.path.join(output_dir, 'ontology_relationships.csv')
            relationships_df.to_csv(relationships_file, index=False)
            logger.info(f"Exported {len(relationships_df)} relationships to {relationships_file}")
    
    except Exception as e:
        logger.error(f"Error exporting ontology to CSV: {e}")

def main():
    """
    Main function to demonstrate ontology visualization and querying.
    """
    try:
        # Connect to Neo4j
        driver = connect_to_neo4j()
        
        # Get ontology classes
        classes = get_ontology_classes(driver)
        print(f"Retrieved {len(classes)} classes from the ontology")
        
        # Get class hierarchy
        hierarchy = get_class_hierarchy(driver)
        print(f"Retrieved {len(hierarchy)} class hierarchy relationships")
        
        # Visualize class hierarchy
        visualize_class_hierarchy(hierarchy)
        
        # Visualize class network for a specific class
        central_class = "invasive species"  # Replace with an actual class from your ontology
        visualize_class_network(driver, central_class)
        
        # Export ontology to CSV
        export_ontology_to_csv(driver)
        
        # Close the driver
        driver.close()
        
        print("Ontology visualization and export completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main function: {e}")

if __name__ == "__main__":
    main()