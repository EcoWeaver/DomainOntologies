"""
BioPortal API utilities for ontology term validation and search.

This module provides functions to interact with the BioPortal API
for searching and validating ontology terms.
"""

import requests
import time
from urllib.parse import quote_plus
import logging
from typing import Dict, List, Optional, Set, Tuple, Union

# Import the config module for environment variables
from config import get_bioportal_api_key

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# List of ontologies to search in BioPortal
DEFAULT_ONTOLOGIES = [
    "CCONT", "BFO", "CHEBI", "GEO", "GO", "NCBITAXON", "OBI",
    "PATO", "PCO", "RO", "UBERON", "OBOE-SBC", "INBIO"
]

def get_term_details(
    term: str, 
    preferred_ontologies: Optional[Dict[str, int]] = None
) -> Optional[Dict[str, Union[str, List[str]]]]:
    """
    Get term details from BioPortal with preferred ontologies.
    
    Args:
        term: Term to search for
        preferred_ontologies: Dictionary of preferred ontologies with scores
        
    Returns:
        Dictionary of term details or None if not found
    """
    api_key = get_bioportal_api_key()
    preferred_ontologies = preferred_ontologies or {}
    logger.info(f"Searching for term: '{term}' with preferred ontologies: {preferred_ontologies}")
    
    url = f"http://data.bioontology.org/search?q={term}&apikey={api_key}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        best_match = None
        highest_score = -1  # Initial negative score to ensure any positive score is better
        
        if data.get('totalCount', 0) > 0:
            for term_details in data['collection']:
                score = 0
                ontology_link = term_details['links']['ontology']
                ontology_id = ontology_link.split("/")[-1]  # Extract ontology ID
                
                # Score increment if in preferred ontology list
                score += preferred_ontologies.get(ontology_id, 0)
                
                # Additional scoring based on information completeness
                if 'definition' in term_details:
                    score += 1
                if 'synonym' in term_details:
                    score += 1
                    
                # Term with highest score
                if score > highest_score:
                    highest_score = score
                    best_match = term_details
                    
        if best_match:
            # Extract definition, handling both string and list formats
            definition = best_match.get('definition', ['No definition provided'])
            if isinstance(definition, list):
                definition = definition[0] if definition else 'No definition provided'
            
            term_info = {
                'uri': best_match['@id'],
                'label': best_match['prefLabel'],
                'definition': definition,
                'ontology': best_match['links']['ontology'],
                'synonyms': best_match.get('synonym', []),
                'relationships': best_match.get('relations', [])
            }
            logger.info(f"Found term: '{term}' in ontology: {term_info['ontology']}")
            return term_info
        
        logger.info(f"No matching term found for: '{term}'")
        return None
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Error searching for term '{term}': {e}")
        return None

def search_bioportal(
    term: str, 
    ontologies: Optional[List[str]] = None,
    exact_match: bool = True,
    require_definitions: bool = True
) -> List[Dict[str, Union[str, List[str]]]]:
    """
    Search for a term in BioPortal and retrieve its details.
    
    Args:
        term: The term to search for in BioPortal
        ontologies: List of ontology acronyms to search within
        exact_match: Whether to require exact matches
        require_definitions: Whether to require definitions
        
    Returns:
        A list of dictionaries containing term details
    """
    api_key = get_bioportal_api_key()
    ontologies = ontologies or DEFAULT_ONTOLOGIES
    
    logger.info(f"Searching for term: '{term}' in BioPortal ontologies: {ontologies}")
    encoded_term = quote_plus(term)
    ontologies_param = ','.join(ontologies)
    
    url = (
        f"https://data.bioontology.org/search?"
        f"q={encoded_term}&ontologies={ontologies_param}&apikey={api_key}"
        f"&require_exact_match={'true' if exact_match else 'false'}"
        f"&require_definitions={'true' if require_definitions else 'false'}"
    )
    
    results = []
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        if data.get('totalCount', 0) > 0:
            for term_details in data['collection']:
                # Extract definition, handling both string and list formats
                definition = term_details.get('definition', ['No definition provided'])
                if isinstance(definition, list):
                    definition = definition[0] if definition else 'No definition provided'
                else:
                    definition = str(definition)
                
                result = {
                    'uri': term_details['@id'],
                    'label': term_details.get('prefLabel', term),
                    'definition': definition,
                    'ontology': term_details.get('links', {}).get('ontology', ''),
                    'validation_link': term_details['links'].get('ui', '')
                }
                results.append(result)
                logger.info(f"Found term: '{term}' with details: {result}")
        else:
            logger.info(f"No results found for term: '{term}'")
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Error searching for term '{term}': {e}")
    
    return results

def batch_search_terms(
    terms: Set[str], 
    ontologies: Optional[List[str]] = None,
    batch_size: int = 10,
    delay: float = 1.0
) -> Dict[str, List[Dict[str, Union[str, List[str]]]]]:
    """
    Search for multiple terms in BioPortal with rate limiting.
    
    Args:
        terms: Set of terms to search for
        ontologies: List of ontology acronyms to search within
        batch_size: Number of terms to process before logging progress
        delay: Delay between API requests in seconds
        
    Returns:
        Dictionary mapping terms to their search results
    """
    results = {}
    term_list = list(terms)
    total_terms = len(term_list)
    
    logger.info(f"Batch searching {total_terms} terms in BioPortal")
    
    for i, term in enumerate(term_list):
        results[term] = search_bioportal(term, ontologies)
        
        # Log progress after each batch
        if (i + 1) % batch_size == 0:
            logger.info(f"Processed {i + 1}/{total_terms} terms")
        
        # Respect rate limits
        if i < total_terms - 1:  # Don't delay after the last item
            time.sleep(delay)
    
    logger.info(f"Completed batch search of {total_terms} terms")
    return results

def get_class_label(ontology_id: str) -> str:
    """
    Fetch the preferred label for a given ontology class ID from BioPortal.
    
    Args:
        ontology_id: The ontology class ID (e.g., 'NCBITaxon_33090')
        
    Returns:
        The preferred label of the class, or an error message if not found
    """
    api_key = get_bioportal_api_key()
    
    if not ontology_id or ontology_id == "None":
        return "No ID provided"
    
    # Handle different ID formats
    if ontology_id.startswith('_'):
        # IDs starting with an underscore use the IRI: 'https://w3id.org/inbio#{ontology_id}'
        ontology_acronym = 'INBIO'
        class_iri = f"https://w3id.org/inbio#{ontology_id}"
    elif '_' in ontology_id:
        # Assume the ID is in the format 'PREFIX_ID' (e.g., 'PATO_0000125')
        prefix, local_id = ontology_id.split('_', 1)
        
        # Map prefixes to ontology acronyms
        prefix_to_acronym = {
            'PATO': 'PATO',
            'NCBITaxon': 'NCBITAXON',
            'ENVO': 'ENVO',
            'BFO': 'BFO',
            'GEO': 'GEO',
            'GO': 'GO',
            'CHEBI': 'CHEBI',
            'OBI': 'OBI',
            'RO': 'RO',
            'UBERON': 'UBERON',
            'IAO': 'IAO',
            'NCIT': 'NCIT',
        }
        
        if prefix in prefix_to_acronym:
            ontology_acronym = prefix_to_acronym[prefix]
            class_iri = f"http://purl.obolibrary.org/obo/{ontology_id}"
        else:
            ontology_acronym = prefix.upper()
            class_iri = f"http://purl.obolibrary.org/obo/{ontology_id}"
    else:
        # Use the search endpoint for IDs without an underscore
        return search_for_label(ontology_id)
    
    # URL-encode the class IRI
    encoded_class_iri = quote_plus(class_iri)
    
    # Construct the API URL
    api_url = f"http://data.bioontology.org/ontologies/{ontology_acronym}/classes/{encoded_class_iri}"
    
    # Include the API key in the query parameters
    params = {'apikey': api_key}
    
    try:
        response = requests.get(api_url, params=params)
        response.raise_for_status()
        data = response.json()
        prefLabel = data.get('prefLabel', 'No label available')
        return prefLabel
    except requests.exceptions.HTTPError as http_err:
        if response.status_code == 404:
            # Try the search endpoint as a fallback
            return search_for_label(ontology_id)
        else:
            logger.error(f"HTTP error occurred while fetching label for '{ontology_id}': {http_err}")
            return "Failed to fetch label"
    except Exception as err:
        logger.error(f"An error occurred while fetching label for '{ontology_id}': {err}")
        return "Failed to fetch label"

def search_for_label(ontology_id: str) -> str:
    """
    Search for a label using the BioPortal search endpoint.
    
    Args:
        ontology_id: The ontology ID to search for
        
    Returns:
        The preferred label if found, or an error message
    """
    api_key = get_bioportal_api_key()
    api_url = "http://data.bioontology.org/search"
    
    params = {
        'apikey': api_key,
        'q': ontology_id,
        'include': 'prefLabel',
        'include_properties': 'false',
    }
    
    try:
        response = requests.get(api_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if data['collection']:
            prefLabel = data['collection'][0].get('prefLabel', 'No label available')
            return prefLabel
        else:
            logger.warning(f"No label found for '{ontology_id}'")
            return "Label not found"
    except Exception as e:
        logger.error(f"An error occurred during search for '{ontology_id}': {e}")
        return "Failed to fetch label"

if __name__ == "__main__":
    # Example usage
    test_term = "invasive species"
    result = get_term_details(test_term, {"INBIO": 3, "NCIT": 2, "BFO": 1})
    
    if result:
        print(f"Found term: {test_term}")
        print(f"URI: {result['uri']}")
        print(f"Label: {result['label']}")
        print(f"Definition: {result['definition']}")
        print(f"Ontology: {result['ontology']}")
    else:
        print(f"Term '{test_term}' not found in BioPortal")