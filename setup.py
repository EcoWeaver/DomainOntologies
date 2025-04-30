"""
Setup script for the INBIO ontology evolution project.

This script checks for required dependencies, creates necessary directories,
and provides instructions for setting up the environment variables.
"""

import os
import sys
import subprocess
import shutil
import platform
from pathlib import Path

def print_header(message):
    """Print a header message."""
    print("\n" + "=" * 80)
    print(f" {message}")
    print("=" * 80)

def print_step(message):
    """Print a step message."""
    print(f"\n>> {message}")

def check_python_version():
    """Check if the Python version is compatible."""
    print_step("Checking Python version...")
    
    required_version = (3, 8)
    current_version = sys.version_info
    
    if current_version.major < required_version[0] or (current_version.major == required_version[0] and current_version.minor < required_version[1]):
        print(f"ERROR: Python {required_version[0]}.{required_version[1]} or higher is required.")
        print(f"Current version: {current_version.major}.{current_version.minor}.{current_version.micro}")
        return False
    
    print(f"Python version {current_version.major}.{current_version.minor}.{current_version.micro} is compatible.")
    return True

def check_pip():
    """Check if pip is installed."""
    print_step("Checking pip installation...")
    
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"], check=True, capture_output=True)
        print("pip is installed.")
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        print("ERROR: pip is not installed or not working properly.")
        print("Please install pip: https://pip.pypa.io/en/stable/installation/")
        return False

def check_dependencies():
    """Check if required dependencies are installed."""
    print_step("Checking required dependencies...")
    
    required_packages = [
        "python-dotenv",
        "openai",
        "langchain",
        "neo4j",
        "spacy",
        "pandas",
        "numpy",
        "requests",
        "rdflib",
        "matplotlib",
        "networkx"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} is installed.")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} is not installed.")
    
    if missing_packages:
        print("\nSome required packages are missing. You can install them with:")
        print(f"pip install {' '.join(missing_packages)}")
        
        install = input("\nDo you want to install the missing packages now? (y/n): ")
        if install.lower() == 'y':
            try:
                subprocess.run([sys.executable, "-m", "pip", "install"] + missing_packages, check=True)
                print("Packages installed successfully.")
            except subprocess.SubprocessError:
                print("ERROR: Failed to install packages.")
                return False
    
    return True

def check_spacy_model():
    """Check if the required spaCy model is installed."""
    print_step("Checking spaCy model...")
    
    try:
        import spacy
        
        try:
            spacy.load('en_core_web_sm')
            print("✓ spaCy model 'en_core_web_sm' is installed.")
            return True
        except OSError:
            print("✗ spaCy model 'en_core_web_sm' is not installed.")
            
            install = input("Do you want to install the spaCy model now? (y/n): ")
            if install.lower() == 'y':
                try:
                    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
                    print("spaCy model installed successfully.")
                    return True
                except subprocess.SubprocessError:
                    print("ERROR: Failed to install spaCy model.")
                    return False
            
            return False
    
    except ImportError:
        print("ERROR: spaCy is not installed.")
        return False

def check_neo4j():
    """Check if Neo4j is installed and running."""
    print_step("Checking Neo4j installation...")
    
    # Check if Neo4j is installed
    if platform.system() == "Windows":
        neo4j_cmd = "where neo4j"
    else:
        neo4j_cmd = "which neo4j"
    
    try:
        subprocess.run(neo4j_cmd, shell=True, check=True, capture_output=True)
        print("✓ Neo4j is installed.")
    except subprocess.SubprocessError:
        print("✗ Neo4j is not installed or not in PATH.")
        print("Please install Neo4j: https://neo4j.com/download/")
        print("Note: You can still run most of the code without Neo4j, but some functionality will be limited.")
    
    # Check if Neo4j Python driver is installed
    try:
        import neo4j
        print("✓ Neo4j Python driver is installed.")
    except ImportError:
        print("✗ Neo4j Python driver is not installed.")
        print("You can install it with: pip install neo4j")
    
    return True

def setup_env_file():
    """Set up the .env file."""
    print_step("Setting up environment variables...")
    
    env_template_path = Path('.env.template')
    env_path = Path('.env')
    
    if not env_template_path.exists():
        print("ERROR: .env.template file not found.")
        return False
    
    if env_path.exists():
        overwrite = input(".env file already exists. Do you want to overwrite it? (y/n): ")
        if overwrite.lower() != 'y':
            print("Skipping .env file setup.")
            return True
    
    # Copy the template
    shutil.copy(env_template_path, env_path)
    print(f"Created .env file from template.")
    print("Please edit the .env file to add your API keys and database credentials.")
    
    # Open the file for editing if possible
    try:
        if platform.system() == "Windows":
            os.startfile(env_path)
        elif platform.system() == "Darwin":  # macOS
            subprocess.run(["open", env_path], check=False)
        else:  # Linux
            subprocess.run(["xdg-open", env_path], check=False)
    except:
        pass
    
    return True

def create_directories():
    """Create necessary directories if they don't exist."""
    print_step("Creating necessary directories...")
    
    directories = [
        'data/abstracts_new',
        'results',
        'coverage_results/txt',
        'ontology_export'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")
    
    return True

def main():
    """Main function to run the setup script."""
    print_header("INBIO Ontology Evolution Project Setup")
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Check pip
    if not check_pip():
        return False
    
    # Check dependencies
    if not check_dependencies():
        return False
    
    # Check spaCy model
    if not check_spacy_model():
        return False
    
    # Check Neo4j
    check_neo4j()
    
    # Create directories
    if not create_directories():
        return False
    
    # Set up .env file
    if not setup_env_file():
        return False
    
    print_header("Setup completed successfully!")
    print("\nNext steps:")
    print("1. Edit the .env file to add your API keys and database credentials.")
    print("2. Run the scripts in the 'scripts' directory to process and analyze the ontology.")
    print("3. Check the README.md file for more information on how to use the code.")
    
    return True

if __name__ == "__main__":
    main()