"""
Ontology term extraction and validation demonstration script.

This script demonstrates the core functionality of the ontology term extraction
and validation process using LLMs and BioPortal.
"""

import os
import pandas as pd
import spacy
import re
from typing import Dict, List, Set, Tuple, Optional
import logging

# Import utility modules
from config import get_openai_api_key
from bioportal_utils import search_bioportal, get_term_details

# Import OpenAI
from openai import OpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load spaCy model
try:
    nlp = spacy.load('en_core_web_sm')
    logger.info("Loaded spaCy model: en_core_web_sm")
except OSError:
    logger.error("spaCy model not found. Please install it using: python -m spacy download en_core_web_sm")
    raise

# Initialize OpenAI client
client = OpenAI(api_key=get_openai_api_key())

def preprocess_text(text: str) -> List[str]:
    """
    Preprocess text by removing abbreviations, numbers, stopwords, and punctuation.
    
    Args:
        text: The text to preprocess
        
    Returns:
        List of preprocessed tokens
    """
    # Remove specific abbreviations
    text = re.sub(r'\b(e\.g\.|i\.e\.|e\.g|i\.e)\b', '', text)
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Tokenize, remove stopwords and punctuation
    doc = nlp(text)
    tokens = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct and len(token.text) > 2]
    return tokens

def extract_terms_and_phrases(text: str, max_phrase_length: int = 2) -> Set[str]:
    """
    Extract terms and phrases from text.
    
    Args:
        text: The text to extract terms from
        max_phrase_length: Maximum length of phrases to extract
        
    Returns:
        Set of extracted terms and phrases
    """
    preprocessed_tokens = preprocess_text(text)
    doc = nlp(" ".join(preprocessed_tokens))
    
    # Extract single terms
    terms = set(token.text.lower() for token in doc 
                if not token.is_stop and not token.is_punct and len(token.text) > 2)
    
    # Extract noun phrases
    phrases = set(chunk.text.lower() for chunk in doc.noun_chunks 
                 if len(chunk.text.split()) <= max_phrase_length)
    
    # Combine terms and phrases
    all_terms = terms.union(phrases)
    return all_terms

def extract_terms_with_llm(text: str, model: str = "gpt-4o") -> Set[str]:
    """
    Extract relevant terms related to invasion biology using LLM.
    
    Args:
        text: The text to extract terms from
        model: The OpenAI model to use
        
    Returns:
        Set of extracted terms
    """
    prompt = f"""
    Extract relevant terms related to invasion biology from the following text. 
    Focus on domain-specific terminology, concepts, and phrases (up to 3 words).
    Return the terms as a comma-separated list.
    
    Text: {text}
    """
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a domain expert in invasion biology."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0,
        )
        
        terms_text = response.choices[0].message.content.strip()
        terms = {term.strip().lower() for term in terms_text.split(',') if term.strip()}
        logger.info(f"Extracted {len(terms)} terms using LLM")
        return terms
    
    except Exception as e:
        logger.error(f"Error extracting terms with LLM: {e}")
        return set()

def validate_terms_with_bioportal(terms: Set[str]) -> Dict[str, Dict]:
    """
    Validate terms using BioPortal.
    
    Args:
        terms: Set of terms to validate
        
    Returns:
        Dictionary mapping terms to their validation results
    """
    results = {}
    
    for term in terms:
        logger.info(f"Validating term: {term}")
        term_results = search_bioportal(term)
        
        if term_results:
            # Term found in BioPortal
            results[term] = {
                'status': 'found',
                'details': term_results[0]  # Use the first (best) result
            }
        else:
            # Term not found in BioPortal
            results[term] = {
                'status': 'not_found',
                'details': None
            }
    
    # Count statistics
    found_count = sum(1 for result in results.values() if result['status'] == 'found')
    logger.info(f"Validated {len(terms)} terms: {found_count} found in BioPortal, {len(terms) - found_count} not found")
    
    return results

def suggest_ontology_placement(term: str, definition: Optional[str] = None) -> Dict:
    """
    Suggest ontology placement for a new term using LLM.
    
    Args:
        term: The term to place in the ontology
        definition: Optional definition of the term
        
    Returns:
        Dictionary with suggested placement information
    """
    prompt = f"""
    As an ontology expert, suggest where to place the following term in an invasion biology ontology:
    
    Term: {term}
    {f"Definition: {definition}" if definition else ""}
    
    Please provide:
    1. A concise definition (if not provided)
    2. Suggested parent class(es)
    3. Potential relationships with other concepts
    4. Any relevant annotations or synonyms
    
    Format your response as a JSON object with the following structure:
    {{
        "definition": "...",
        "parent_classes": ["class1", "class2"],
        "relationships": [
            {{"predicate": "predicate1", "object": "object1"}},
            {{"predicate": "predicate2", "object": "object2"}}
        ],
        "synonyms": ["synonym1", "synonym2"],
        "annotations": {{"key1": "value1", "key2": "value2"}}
    }}
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an ontology engineering expert."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.2,
        )
        
        suggestion_text = response.choices[0].message.content.strip()
        
        # Extract JSON from the response
        import json
        import re
        
        # Find JSON object in the response
        json_match = re.search(r'({.*})', suggestion_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            try:
                suggestion = json.loads(json_str)
                logger.info(f"Generated ontology placement suggestion for term: {term}")
                return suggestion
            except json.JSONDecodeError:
                logger.error(f"Failed to parse JSON from LLM response for term: {term}")
                return {"error": "Failed to parse JSON from LLM response"}
        else:
            logger.error(f"No JSON found in LLM response for term: {term}")
            return {"error": "No JSON found in LLM response"}
    
    except Exception as e:
        logger.error(f"Error suggesting ontology placement for term '{term}': {e}")
        return {"error": str(e)}

def process_abstract(abstract_text: str) -> Dict:
    """
    Process an abstract to extract, validate, and suggest placement for terms.
    
    Args:
        abstract_text: The abstract text to process
        
    Returns:
        Dictionary with processing results
    """
    logger.info("Processing abstract...")
    
    # Extract terms using spaCy
    spacy_terms = extract_terms_and_phrases(abstract_text)
    logger.info(f"Extracted {len(spacy_terms)} terms using spaCy")
    
    # Extract terms using LLM
    llm_terms = extract_terms_with_llm(abstract_text)
    logger.info(f"Extracted {len(llm_terms)} terms using LLM")
    
    # Combine terms
    all_terms = spacy_terms.union(llm_terms)
    logger.info(f"Combined unique terms: {len(all_terms)}")
    
    # Validate terms with BioPortal
    validation_results = validate_terms_with_bioportal(all_terms)
    
    # Process terms not found in BioPortal
    new_term_suggestions = {}
    for term, result in validation_results.items():
        if result['status'] == 'not_found':
            logger.info(f"Suggesting ontology placement for new term: {term}")
            suggestion = suggest_ontology_placement(term)
            new_term_suggestions[term] = suggestion
    
    # Prepare results
    results = {
        'extracted_terms': {
            'spacy': list(spacy_terms),
            'llm': list(llm_terms),
            'combined': list(all_terms)
        },
        'validation_results': validation_results,
        'new_term_suggestions': new_term_suggestions
    }
    
    return results

def save_results(results: Dict, output_file: str = 'term_extraction_results.json'):
    """
    Save processing results to a JSON file.
    
    Args:
        results: The results to save
        output_file: The output file path
    """
    import json
    
    try:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_file}")
    except Exception as e:
        logger.error(f"Error saving results to {output_file}: {e}")

def main():
    """
    Main function to demonstrate the ontology term extraction and validation process.
    """
    # Example abstract text
    abstract_text = """
    Invasive species pose significant threats to biodiversity and ecosystem functioning worldwide. 
    The enemy release hypothesis suggests that invasive species benefit from reduced pressure from 
    natural enemies in their introduced range. This study examines how propagule pressure and 
    disturbance interact to facilitate the establishment of non-native plant species in island 
    ecosystems. We found that invasive plants with high phenotypic plasticity were more successful 
    in colonizing disturbed habitats, supporting the disturbance hypothesis. Additionally, our 
    results indicate that biotic resistance from native communities can be overcome when invasive 
    species possess traits that differ significantly from native species, aligning with the limiting 
    similarity hypothesis. These findings contribute to our understanding of invasion mechanisms and 
    may inform management strategies for controlling invasive species.
    """
    
    # Process the abstract
    results = process_abstract(abstract_text)
    
    # Save results
    save_results(results)
    
    # Print summary
    print("\nSummary of Results:")
    print(f"Total terms extracted: {len(results['extracted_terms']['combined'])}")
    
    found_terms = sum(1 for result in results['validation_results'].values() if result['status'] == 'found')
    print(f"Terms found in BioPortal: {found_terms}")
    print(f"Terms not found in BioPortal: {len(results['validation_results']) - found_terms}")
    print(f"New term suggestions generated: {len(results['new_term_suggestions'])}")
    
    print("\nProcess completed successfully!")

if __name__ == "__main__":
    main()