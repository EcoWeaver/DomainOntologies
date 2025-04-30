"""
Configuration module for loading environment variables.

This module provides functions to load and access environment variables
for API keys and other configuration settings.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Try to load environment variables from .env file
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

def get_env_variable(var_name, default=None, required=False):
    """
    Get an environment variable or return a default value.
    
    Args:
        var_name (str): Name of the environment variable.
        default (any, optional): Default value to return if the variable is not set.
        required (bool, optional): If True and the variable is not set, raise an error.
        
    Returns:
        str: The value of the environment variable or the default value.
        
    Raises:
        ValueError: If the variable is required but not set.
    """
    value = os.environ.get(var_name, default)
    if required and value is None:
        raise ValueError(f"Environment variable {var_name} is required but not set. "
                         f"Please create a .env file based on .env.template.")
    return value

# OpenAI API Key
def get_openai_api_key():
    """Get the OpenAI API key from environment variables."""
    return get_env_variable('OPENAI_API_KEY', required=True)

# BioPortal API Key
def get_bioportal_api_key():
    """Get the BioPortal API key from environment variables."""
    return get_env_variable('BIOPORTAL_API_KEY', required=True)

# Neo4j Database Configuration
def get_neo4j_config():
    """Get Neo4j database configuration from environment variables."""
    return {
        'uri': get_env_variable('NEO4J_URI', 'neo4j://localhost:7687'),
        'username': get_env_variable('NEO4J_USERNAME', 'neo4j'),
        'password': get_env_variable('NEO4J_PASSWORD', required=True)
    }

# LangChain Configuration
def get_langchain_config():
    """Get LangChain configuration from environment variables."""
    return {
        'endpoint': get_env_variable('LANGCHAIN_ENDPOINT', 'https://eu.api.smith.langchain.com'),
        'api_key': get_env_variable('LANGCHAIN_API_KEY', required=False),
        'project': get_env_variable('LANGCHAIN_PROJECT', 'thesisV1')
    }

# Check if required environment variables are set
def check_environment():
    """
    Check if all required environment variables are set.
    
    Returns:
        bool: True if all required variables are set, False otherwise.
    """
    try:
        get_openai_api_key()
        get_bioportal_api_key()
        get_neo4j_config()
        return True
    except ValueError as e:
        print(f"Environment configuration error: {e}", file=sys.stderr)
        return False

if __name__ == "__main__":
    # If this script is run directly, check the environment configuration
    if check_environment():
        print("Environment configuration is valid.")
    else:
        print("Please set up your environment variables in a .env file.")
        print("You can use the .env.template file as a starting point.")