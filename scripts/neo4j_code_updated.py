"""
Neo4j integration with LangChain for querying the INBIO ontology.

This script connects to a Neo4j database containing the INBIO ontology
and uses LangChain with OpenAI to perform natural language queries.
"""

import os
from neo4j import GraphDatabase
from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain_openai import ChatOpenAI

# Import the config module for environment variables
from config import get_openai_api_key, get_neo4j_config

def main():
    """
    Main function to set up and run Neo4j with LangChain integration.
    """
    # Get API keys and credentials from environment variables
    openai_api_key = get_openai_api_key()
    neo4j_config = get_neo4j_config()
    
    # Extract Neo4j configuration
    neo4j_url = neo4j_config['uri']
    neo4j_user = neo4j_config['username']
    neo4j_password = neo4j_config['password']
    
    # Verify Neo4j connectivity
    try:
        with GraphDatabase.driver(neo4j_url, auth=(neo4j_user, neo4j_password)) as driver:
            driver.verify_connectivity()
            print("Successfully connected to Neo4j database.")
    except Exception as e:
        print(f"Error connecting to Neo4j: {e}")
        return
    
    # Initialize Neo4j graph
    try:
        graph = Neo4jGraph(
            url=neo4j_url,
            username=neo4j_user,
            password=neo4j_password,
            database="inbionew"  # Specify the database name
        )
        print("Neo4j graph initialized successfully.")
    except Exception as e:
        print(f"Error initializing Neo4j graph: {e}")
        return
    
    # Initialize LangChain with OpenAI
    try:
        chain = GraphCypherQAChain.from_llm(
            ChatOpenAI(temperature=0, api_key=openai_api_key),
            graph=graph,
            verbose=True
        )
        print("LangChain initialized successfully.")
    except Exception as e:
        print(f"Error initializing LangChain: {e}")
        return
    
    # Example queries
    example_queries = [
        "List all Classes",
        "List all Properties",
        "Find class with label 'conceptual entity'",
        "Find class with label 'continuant'",
        "What are the relationships available in class label 'continuant'",
        "What classes are available in class label 'continuant'"
    ]
    
    # Run example queries
    for query in example_queries:
        print(f"\nExecuting query: '{query}'")
        try:
            result = chain.invoke({"query": query})
            print(f"Result: {result}")
        except Exception as e:
            print(f"Error executing query: {e}")

if __name__ == "__main__":
    main()