# Ontology Development and Evolution in the Invasion Biology Domain Using Large Language Models

This repository contains the code and resources for the research paper "Ontology Evolution in Invasion Biology Using Large Language Models: A Hybrid Approach" presented at the SemDH2025 conference.

## Overview

This research presents a hybrid approach for ontology evolution that integrates Large Language Models (LLMs)—specifically GPT-4-based pipelines—with classical ontology engineering practices. This integration aims to create dynamic, scalable, and semantically consistent ontologies suitable for representing emergent phenomena in invasion biology.

### Key Components

1. **Concept and Relationship Extraction**: Analysis of hypothesis texts, scholarly abstracts, and curated domain metadata to extract candidate terms.
2. **LLM-driven Pipeline**: Incorporating prompt-engineering and zero-shot learning to generate novel concepts and relationships.
3. **Expert Validation**: Validation of newly proposed classes by domain experts in an iterative loop.

## Repository Structure

```
.
├── scripts/                  # Core processing scripts
│   ├── config.py             # Configuration and environment variables
│   ├── llm_integ_updated.py  # LLM integration for concept extraction
│   ├── neo4j_code_updated.py # Neo4j database integration
│   └── ...                   # Other utility scripts
├── data/                     # Sample data files
│   └── abstracts_new/        # Abstract text files
├── results/                  # Ontology files and analysis results
│   └── INBIO-NEW_v1.owl      # The INBIO ontology file
├── coverage_results/         # Coverage analysis results
├── .env.template             # Template for environment variables
├── .gitignore                # Git ignore file
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- Neo4j Database (for ontology storage and querying)
- OpenAI API key (for LLM integration)
- BioPortal API key (for ontology term validation)

### Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/invasion-biology-ontology.git
   cd invasion-biology-ontology
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:

   ```bash
   cp .env.template .env
   ```

   Edit the `.env` file to add your API keys and database credentials.

4. Install spaCy language model:
   ```bash
   python -m spacy download en_core_web_sm
   ```

## Usage

### Ontology Term Extraction

To extract terms from abstracts and hypothesis files:

```bash
python scripts/llm_integ_updated.py
```

### Neo4j Integration

To query the ontology using natural language with Neo4j and LangChain:

```bash
python scripts/neo4j_code_updated.py
```

### Coverage Analysis

The scripts in this repository can be used to analyze the coverage of the INBIO ontology against terms extracted from abstracts and hypothesis files:

```bash
python scripts/statistical_evaluation.py
```

## Key Results

The methodology demonstrated:

- Extraction of **175 unique triplets** (69 subjects, 63 predicates, 113 objects)
- Addition of **46 new concepts** and **24 relationships** to the ontology
- Validation accuracy exceeding **90%** for most new additions

## Acknowledgements

This research was supervised by **Prof. Dr. Alsayed Algergawy** at the **University of Passau**. Domain expertise was provided by **PD Dr. Tina Heger** from **Uni Jena**, whose insights greatly enhanced the validation framework.

## References

1. **INBIO Ontology** - [BioPortal](https://bioportal.bioontology.org/ontologies/INBIO)
2. **LangChain** - [GitHub Repository](https://github.com/langchain-ai/langchain)
3. **OpenAI GPT-4** - [OpenAI Documentation](https://platform.openai.com/docs/overview)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
