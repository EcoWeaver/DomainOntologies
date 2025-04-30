# %%
import pandas as pd
import os
import openpyxl
import spacy
import nltk
import re
from tqdm import tqdm
from nltk.corpus import stopwords
import string

# Import the config module for environment variables
from config import get_openai_api_key, get_bioportal_api_key

# Download NLTK resources if needed
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load spaCy model
nlp = spacy.load('en_core_web_md')

# %%
import rdflib

g = rdflib.Graph()
g.parse("INBIO-1.2.owl", format="xml") 

# %%
# SPARQL queries for ontology extraction
preferred_labels_query = """
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT DISTINCT ?label WHERE {
    ?s rdfs:label ?label .
}
"""

# SPARQL query to extract synonyms
synonyms_query = """
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX oboInOwl: <http://www.geneontology.org/formats/oboInOwl#>
SELECT DISTINCT ?synonym WHERE {
    { ?s skos:altLabel ?synonym . }
    UNION
    { ?s oboInOwl:hasExactSynonym ?synonym . }
    UNION
    { ?s oboInOwl:hasRelatedSynonym ?synonym . }
}
"""

# SPARQL query to extract definitions
definitions_query = """
PREFIX IAO: <http://purl.obolibrary.org/obo/IAO_>
SELECT DISTINCT ?definition WHERE {
    ?s IAO:0000115 ?definition .
}
"""

# %%
# Execute SPARQL queries to extract ontology terms
preferred_labels = set()
for row in g.query(preferred_labels_query):
    preferred_labels.add(str(row.label))

synonyms = set()
for row in g.query(synonyms_query):
    synonyms.add(str(row.synonym))

definitions = set()
for row in g.query(definitions_query):
    definitions.add(str(row.definition))

# Combine preferred labels and synonyms into a single set for concept terms
concept_terms = preferred_labels.union(synonyms)

# %%
from openai import OpenAI

# Initialize OpenAI client with API key from environment variables
client = OpenAI(api_key=get_openai_api_key())

def extract_terms_from_definition(definition, model="gpt-4o"):
    """
    Extract relevant terms related to invasion biology from a definition.
    
    Args:
        definition (str): The definition text to extract terms from
        model (str): The OpenAI model to use
        
    Returns:
        set: A set of extracted terms
    """
    prompt = f"Extract relevant terms related to invasion biology, which can mean together (max 2-3 terms together), from the following definition: {definition}. Ensure to remove any stop words, punctuation, and clean the text. "    
    response = client.chat.completions.create(
        model=model,
        messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
        max_tokens=4000,
        n=1,
        stop=None,
        temperature=0,
    )
    terms = response.choices[0].message.content.strip().split(', ')
    return set(terms)

# %%
# Custom stopwords
custom_stopwords = set(['e.g.', 'i.e.', 'e.g'])

def extract_specific_columns(file_path, columns_dict):
    """
    Extract specific columns from Excel sheets.
    
    Args:
        file_path (str): Path to the Excel file
        columns_dict (dict): Dictionary mapping sheet names to column names
        
    Returns:
        dict: Dictionary containing extracted data
    """
    extracted_data = {}
    workbook = openpyxl.load_workbook(file_path, data_only=True)
    for sheet_name, columns in columns_dict.items():
        if sheet_name in workbook.sheetnames:
            df = pd.read_excel(file_path, sheet_name=sheet_name, usecols=columns)
            extracted_data[sheet_name] = df
        else:
            print(f"Worksheet {sheet_name} not found in file {file_path}")
    return extracted_data

def preprocess_text(text):
    """
    Preprocess text by removing abbreviations, numbers, stopwords, and punctuation.
    
    Args:
        text (str): The text to preprocess
        
    Returns:
        list: List of preprocessed tokens
    """
    # Remove specific abbreviations
    text = re.sub(r'\b(e\.g\.|i\.e\.|e\.g|i\.e)\b', '', text)
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Tokenize, remove stopwords and punctuation
    doc = nlp(text)
    tokens = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct and len(token.text) > 2]

    # Remove custom stopwords
    tokens = [token for token in tokens if token not in custom_stopwords]
    return tokens

def extract_terms_and_phrases_from_text(text, nlp, max_phrase_length=2):
    """
    Extract terms and phrases from text.
    
    Args:
        text (str): The text to extract terms from
        nlp: The spaCy NLP model
        max_phrase_length (int): Maximum length of phrases to extract
        
    Returns:
        set: Set of extracted terms and phrases
    """
    preprocessed_tokens = preprocess_text(text)
    doc = nlp(" ".join(preprocessed_tokens))
    terms = set(token.text.lower() for token in doc if not token.is_stop and not token.is_punct and len(token.text) > 2)
    phrases = set(chunk.text.lower() for chunk in doc.noun_chunks if len(chunk.text.split()) <= max_phrase_length)
    all_terms = terms.union(phrases)
    return all_terms

def extract_terms_and_phrases_from_data_enhanced(data, nlp):
    """
    Extract terms and phrases from all data in a dictionary.
    
    Args:
        data (dict): Dictionary containing DataFrames
        nlp: The spaCy NLP model
        
    Returns:
        set: Set of all extracted terms and phrases
    """
    terms_and_phrases = set()
    for df in data.values():
        for column in df.columns:
            for text in df[column].astype(str):
                terms_and_phrases.update(extract_terms_and_phrases_from_text(text, nlp))
    return terms_and_phrases

def extract_verbs_and_relationships(text, nlp):
    """
    Extract verbs and relationships from text.
    
    Args:
        text (str): The text to extract verbs and relationships from
        nlp: The spaCy NLP model
        
    Returns:
        tuple: A tuple containing sets of verbs and relationships
    """
    doc = nlp(text)
    verbs = set()
    relationships = set()
    for token in doc:
        if token.pos_ == 'VERB':
            verbs.add(token.lemma_.lower())
        if token.dep_ in ('nsubj', 'dobj', 'prep'):
            relationships.add((token.head.lemma_.lower(), token.lemma_.lower()))
    return verbs, relationships

# %%
def calculate_coverage_without_abstract(hypothesis_files, directory, concept_terms, definition_terms, nlp):
    """
    Calculate coverage without abstract.
    
    Args:
        hypothesis_files (dict): Dictionary of hypothesis files
        directory (str): Directory containing hypothesis files
        concept_terms (set): Set of concept terms
        definition_terms (set): Set of definition terms
        nlp: The spaCy NLP model
        
    Returns:
        tuple: A tuple containing DataFrames and sets of terms
    """
    definition_terms = set()
    for definition in definitions:
        definition_terms.update(extract_terms_from_definition(definition))
    
    results = []

    unique_concept_covered_terms = set()
    unique_concept_non_matched_terms = set()
    unique_definition_covered_terms = set()
    unique_definition_non_matched_terms = set()
    verbs_relationships = []
    
    for hypothesis, info in hypothesis_files.items():
        file_path = os.path.join(directory, info['filename'])
        columns_dict = info['columns']
        data = extract_specific_columns(file_path, columns_dict)
        excel_terms_and_phrases = extract_terms_and_phrases_from_data_enhanced(data, nlp)
        statement_terms_and_phrases = extract_terms_and_phrases_from_text(info['statement'], nlp)
        statement_verbs, statement_relationships = extract_verbs_and_relationships(info['statement'], nlp)
        all_terms_and_phrases = excel_terms_and_phrases.union(statement_terms_and_phrases)
        
        # Concept level coverage
        concept_covered_terms = concept_terms.intersection(all_terms_and_phrases)
        concept_non_matched_terms = all_terms_and_phrases.difference(concept_terms)
        concept_coverage_percentage = len(concept_covered_terms) / len(all_terms_and_phrases) * 100 if all_terms_and_phrases else 0
        
        # Definition level coverage
        definition_covered_terms = definition_terms.intersection(all_terms_and_phrases)
        definition_non_matched_terms = all_terms_and_phrases.difference(definition_terms)
        definition_coverage_percentage = len(definition_covered_terms) / len(all_terms_and_phrases) * 100 if all_terms_and_phrases else 0
        
        # Filter definition non-matched terms to remove those matched at concept level
        filtered_definition_non_matched_terms = definition_non_matched_terms.difference(concept_covered_terms)
        print(f"Concept Non Covered Terms: {concept_non_matched_terms}")
        
        # Aggregate unique terms
        unique_concept_covered_terms.update(concept_covered_terms)
        unique_concept_non_matched_terms.update(concept_non_matched_terms)
        unique_definition_covered_terms.update(definition_covered_terms)
        unique_definition_non_matched_terms.update(definition_non_matched_terms)

        results.append({
            'Hypothesis': hypothesis,
            'Concept Covered Terms': list(concept_covered_terms),
            'Definition Covered Terms': list(definition_covered_terms),
            'Concept Non-Matched Terms': list(concept_non_matched_terms),
            'Definition Non-Matched Terms': list(definition_non_matched_terms),
            'Statement': info['statement'],
        })

        verbs_relationships.append({
            'Hypothesis': hypothesis,
            'Verbs': list(statement_verbs),
            'Relationships': list(statement_relationships)
        })
    
    df_results = pd.DataFrame(results)
    df_verbs_relationships = pd.DataFrame(verbs_relationships)

    return df_results, df_verbs_relationships, unique_concept_covered_terms, unique_concept_non_matched_terms, unique_definition_covered_terms, unique_definition_non_matched_terms

# %%
# Hypothesis files configuration
hypothesis_files = {
    'H00 – Enemy release hypothesis': {
        'filename': 'Hi-K_Enemy_release.xlsx',
        'columns': {
            'Glossary': ['Term', 'Definition / Information'],
        },
        'statement': 'The absence of enemies in the exotic range is a cause of invasion success.',
        'label': '0'
    },
    'H01 – Biotic resistance hypothesis': {
        'filename': 'Hi-K_Biotic_resistance.xlsx',
        'columns': {
            'Glossary': ['Term', 'Definition / Information'],
        },
        'statement': 'An ecosystem with high biodiversity is more resistant against non-native species than an ecosystem with lower biodiversity.',
        'label': '1'
    },
    'H02 – Phenotypic plasticity hypothesis': {
        'filename': 'Hi-K_Phenotypic_plasticity.xlsx',
        'columns': {
            'Glossary': ['Term', 'Definition / Information'],
        },
        'statement': 'Invasive species are more phenotypically plastic than non-invasive or native ones.',
        'label': '2'
    },
    "H03 – Darwin's naturalization hypothesis": {
        'filename': 'Hi-K_Darwins_naturalisation.xlsx',
        'columns': {
            'Glossary': ['Term', 'Definition / Information'],
        },
        'statement': 'The invasion success of non-native species is higher in areas that are poor in closely related species than in areas that are rich in closely related species.',
        'label': '3'
    },
    'H04 – Island susceptibility hypothesis': {
        'filename': 'Hi-K_Island_susceptibility.xlsx',
        'columns': {
            'Glossary': ['Term', 'Definition / Information'],
        },
        'statement': 'Non-native species are more likely to become established and have major ecological impacts on islands than on continents.',
        'label': '4'
    },
    'H05 – Limiting similarity hypothesis': {
        'filename': 'Hi-K_Limiting_similarity.xlsx',
        'columns': {
            'Glossary': ['Term', 'Definition / Information'],
        },
        'statement': 'The invasion success of non-native species is high if they strongly differ from native species, and it is low if they are similar to native species.',
        'label': '5'
    },
    'H06 – Propagule pressure hypothesis': {
        'filename': 'Hi-K_Propagule_pressure.xlsx',
        'columns': {
            'Glossary': ['Term', 'Definition / Information'],
        },
        'statement': 'A high propagule pressure (a composite measure consisting of the number of individuals introduced per introduction event and the frequency of introduction events) is a cause of invasion success.',
        'label': '6'
    },
    'H07 – Disturbance hypothesis': {
        'filename': 'Hi-K_Disturbance.xlsx',
        'columns': {
            'Glossary': ['Term', 'Definition / Information'],
        },
        'statement': 'The invasion success of non-native species is higher in highly disturbed than in relatively undisturbed ecosystems.',
        'label': '7'
    },
    'H08 – Invasional meltdown hypothesis': {
        'filename': 'Hi-K_Invasional_meltdown.xlsx',
        'columns': {
            'Glossary': ['Term', 'Definition / Information'],
        },
        'statement': 'The presence of non-native species in an ecosystem facilitates invasion by additional species, increasing their likelihood of survival or ecological impact.',
        'label': '8'
    },
    'H09 – Tens Rule Hypothesis': {
        'filename': 'Hi-K_Tens_rule.xlsx',
        'columns': {
            'Glossary': ['Term', 'Definition / Information']
        },
        'statement': 'After having been released from natural specialist enemies, non-native species will allocate more energy in cheap (energy-inexpensive) defenses against generalist enemies and less energy in expensive defenses against specialist enemies (this re-allocation is due to genetic changes); the energy gained in this way will be invested in growth and/or reproduction, which makes the non-native species more competitive.',
        'label': '9'
    },
}

directory = 'hypothesis_excel_files/'

# %%
# Calculate coverage without abstract
df_without_abstract, df_verbs_relationships_without, unique_concept_covered_terms_without, unique_concept_non_matched_terms_without, unique_definition_covered_terms_without, unique_definition_non_matched_terms_without = calculate_coverage_without_abstract(hypothesis_files, directory, concept_terms, definitions, nlp)

# %%
# Functions for loading and processing abstracts
import os
import pandas as pd
import chardet

hypothesis_mapping = {
    '0': 'Enemy release hypothesis',
    '1': 'Biotic resistance hypothesis',
    '2': 'Phenotypic plasticity hypothesis',
    '3': 'Darwins naturalization hypothesis',
    '4': 'Island susceptibility hypothesis',
    '5': 'Limiting similarity hypothesis',
    '6': 'Propagule pressure hypothesis',
    '7': 'Disturbance hypothesis',
    '8': 'Invasional meltdown hypothesis',
    '9': 'Tens rule hypothesis'
}

def extract_main_hypothesis(labels):
    """
    Extract main hypothesis from labels.
    
    Args:
        labels (str): Comma-separated labels
        
    Returns:
        list: List of main hypotheses
    """
    main_hypotheses = set()
    for label in labels.split(','):
        main_hypothesis = label.split('-')[0]
        main_hypotheses.add(main_hypothesis)
    return list(main_hypotheses)

# Folder containing the text files
folder_path = 'abstracts_new'

def load_abstracts(folder_path):
    """
    Load abstracts from text files.
    
    Args:
        folder_path (str): Path to the folder containing abstract files
        
    Returns:
        pd.DataFrame: DataFrame containing abstract data
    """
    titles = []
    abstracts = []
    main_hypotheses = []
    hypothesis_names = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            
            # Detect file encoding
            with open(file_path, 'rb') as file:
                raw_data = file.read()
                result = chardet.detect(raw_data)
                encoding = result['encoding']
            
            # Read the file with the detected encoding
            with open(file_path, 'r', encoding=encoding) as file:
                lines = file.readlines()
                title = lines[0].strip()
                abstract = lines[1].strip()
                labels = lines[2].strip()
                
                # Extract main hypotheses
                main_hypothesis_labels = extract_main_hypothesis(labels)
                main_hypothesis_names = [hypothesis_mapping[label] for label in main_hypothesis_labels]
                
                # Append data to lists
                titles.append(title)
                abstracts.append(abstract)
                main_hypotheses.append(','.join(main_hypothesis_labels))
                hypothesis_names.append(','.join(main_hypothesis_names))

    # Create a DataFrame
    df_abstract = pd.DataFrame({
        'hypothesis_labels': main_hypotheses,
        'hypothesis_names': hypothesis_names,
        'hypothesis_title': titles,
        'hypothesis_abstract': abstracts
    })
    return df_abstract

# %%
def calculate_coverage_with_abstract(hypothesis_files, directory, concept_terms, definition_terms, nlp, abstract_file):
    """
    Calculate coverage with abstract.
    
    Args:
        hypothesis_files (dict): Dictionary of hypothesis files
        directory (str): Directory containing hypothesis files
        concept_terms (set): Set of concept terms
        definition_terms (set): Set of definition terms
        nlp: The spaCy NLP model
        abstract_file (str): Path to the abstract file
        
    Returns:
        tuple: A tuple containing DataFrames and sets of terms
    """
    definition_terms = set()
    for definition in definitions:
        definition_terms.update(extract_terms_from_definition(definition))
    
    # Read the abstract data
    abstract_data = load_abstracts(folder_path)
    nlp.max_length = 10000000
    results = []
    abstract_verbs_relationships = []
    statement_verbs_relationships = []

    unique_concept_covered_terms = set()
    unique_concept_non_matched_terms = set()
    unique_definition_covered_terms = set()
    unique_definition_non_matched_terms = set()

    for hypothesis, info in hypothesis_files.items():
        file_path = os.path.join(directory, info['filename'])
        columns_dict = info['columns']
        data = extract_specific_columns(file_path, columns_dict)
        excel_terms_and_phrases = extract_terms_and_phrases_from_data_enhanced(data, nlp)
        statement_terms_and_phrases = extract_terms_and_phrases_from_text(info['statement'], nlp)
        statement_verbs, statement_relationships = extract_verbs_and_relationships(info['statement'], nlp)
        
        abstracts = abstract_data[abstract_data['hypothesis_labels'].astype(str).str.contains(info['label'])]['hypothesis_abstract'].values
        abstract_terms_and_phrases = set()
        for abstract in abstracts:
            abstract_terms_and_phrases.update(extract_terms_and_phrases_from_text(abstract, nlp))
        abstract_verbs, abstract_relationships = extract_verbs_and_relationships(' '.join(abstracts), nlp)
        
        # Combine all terms and phrases
        all_terms_and_phrases = excel_terms_and_phrases.union(statement_terms_and_phrases).union(abstract_terms_and_phrases)
        
        # Concept level coverage
        concept_covered_terms = concept_terms.intersection(all_terms_and_phrases)
        concept_non_matched_terms = all_terms_and_phrases.difference(concept_terms)
        concept_coverage_percentage = len(concept_covered_terms) / len(all_terms_and_phrases) * 100 if all_terms_and_phrases else 0
        
        # Definition level coverage
        definition_covered_terms = definition_terms.intersection(all_terms_and_phrases)
        definition_non_matched_terms = all_terms_and_phrases.difference(definition_terms)
        definition_coverage_percentage = len(definition_covered_terms) / len(all_terms_and_phrases) * 100 if all_terms_and_phrases else 0
        
        # Filter definition non-matched terms to remove those matched at concept level
        filtered_definition_non_matched_terms = definition_non_matched_terms.difference(concept_covered_terms)

        # Aggregate unique terms
        unique_concept_covered_terms.update(concept_covered_terms)
        unique_concept_non_matched_terms.update(concept_non_matched_terms)
        unique_definition_covered_terms.update(definition_covered_terms)
        unique_definition_non_matched_terms.update(definition_non_matched_terms)

        results.append({
            'Hypothesis': hypothesis,
            'Concept Covered Terms': list(concept_covered_terms),
            'Definition Covered Terms': list(definition_covered_terms),
            'Concept Non-Matched Terms': list(concept_non_matched_terms),
            'Definition Non-Matched Terms': list(definition_non_matched_terms),
            'Statement': info['statement'],
            'Abstract': ' '.join(abstracts),
        })

        abstract_verbs_relationships.append({
            'Hypothesis': hypothesis,
            'Verbs': list(abstract_verbs),
            'Relationships': list(abstract_relationships),
            'Abstract': ' '.join(abstracts)
        })
        statement_verbs_relationships.append({
            'Hypothesis': hypothesis,
            'Verbs': list(statement_verbs),
            'Relationships': list(statement_relationships),
            'Statement': info['statement']
        })
    
    df_results = pd.DataFrame(results)
    df_abstract_verbs_relationships = pd.DataFrame(abstract_verbs_relationships)
    df_statement_verbs_relationships = pd.DataFrame(statement_verbs_relationships)

    return df_results, df_abstract_verbs_relationships, df_statement_verbs_relationships, unique_concept_covered_terms, unique_concept_non_matched_terms, unique_definition_covered_terms, unique_definition_non_matched_terms

# %%
import os
from datetime import datetime

def write_terms_to_file(terms, base_filename):
    """
    Write terms to a file.
    
    Args:
        terms (set): Set of terms to write
        base_filename (str): Base filename to write to
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{base_filename}__.txt"
    if os.path.exists(filename):
        filename = f"{base_filename}_{timestamp}.txt"
    with open(filename, 'w') as file:
        for term in terms:
            file.write(f"{term}\n")

def combine_and_save_results(df_without_abstract, df_with_abstract, df_verbs_relationships_without, df_verbs_relationships_with, df_statement_verbs_relationships_with, unique_terms_without, unique_terms_with):
    """
    Combine and save results.
    
    Args:
        df_without_abstract (pd.DataFrame): DataFrame of results without abstract
        df_with_abstract (pd.DataFrame): DataFrame of results with abstract
        df_verbs_relationships_without (pd.DataFrame): DataFrame of verb relationships without abstract
        df_verbs_relationships_with (pd.DataFrame): DataFrame of verb relationships with abstract
        df_statement_verbs_relationships_with (pd.DataFrame): DataFrame of statement verb relationships with abstract
        unique_terms_without (tuple): Tuple of unique terms without abstract
        unique_terms_with (tuple): Tuple of unique terms with abstract
    """
    # Combine unique terms
    unique_concept_covered_terms = unique_terms_without[0].union(unique_terms_with[0])
    unique_concept_non_matched_terms = unique_terms_without[1].union(unique_terms_with[1])
    write_terms_to_file(unique_concept_non_matched_terms, 'coverage_results/txt/unique_concept_non_matched_terms')

    unique_definition_covered_terms = unique_terms_without[2].union(unique_terms_with[2])
    unique_definition_non_matched_terms = unique_terms_without[3].union(unique_terms_with[3])
    write_terms_to_file(unique_definition_non_matched_terms, 'coverage_results/txt/unique_definition_non_matched_terms')
    write_terms_to_file(unique_concept_non_matched_terms.union(unique_definition_non_matched_terms), 'coverage_results/txt/allnon_matched_terms')

    # Categorize non-matched terms
    related_terms, unrelated_terms = categorize_non_matched_terms(unique_concept_non_matched_terms.union(unique_definition_non_matched_terms))
    print(f"Related terms: {related_terms}")
    print(f"Unrelated terms: {unrelated_terms}")

    # Create DataFrames for unique terms
    df_unique_concept_covered_terms = pd.DataFrame({'Unique Concept Covered Terms': list(unique_concept_covered_terms)})
    df_unique_concept_non_matched_terms = pd.DataFrame({'Unique Concept Non-Matched Terms': list(unique_concept_non_matched_terms)})
    df_unique_definition_covered_terms = pd.DataFrame({'Unique Definition Covered Terms': list(unique_definition_covered_terms)})
    df_unique_definition_non_matched_terms = pd.DataFrame({'Unique Definition Non-Matched Terms': list(unique_definition_non_matched_terms)})
    df_related_terms = pd.DataFrame({'Related Terms': list(related_terms)})
    df_unrelated_terms = pd.DataFrame({'Unrelated Terms': list(unrelated_terms)})

    with pd.ExcelWriter('coverage_results/combined_results_CATEGORIZED_NEW.xlsx', engine='openpyxl') as writer:
        df_without_abstract.to_excel(writer, sheet_name='Coverage Results Without Abstract', index=False)
        df_with_abstract.to_excel(writer, sheet_name='Coverage Results With Abstract', index=False)
        df_unique_concept_covered_terms.to_excel(writer, sheet_name='Unique Concept Covered Terms', index=False)
        df_unique_concept_non_matched_terms.to_excel(writer, sheet_name='Unique Concept Non-Matched Terms', index=False)
        df_unique_definition_covered_terms.to_excel(writer, sheet_name='Unique Definition Covered Terms', index=False)
        df_unique_definition_non_matched_terms.to_excel(writer, sheet_name='Unique Definition Non-Matched Terms', index=False)
        df_related_terms.to_excel(writer, sheet_name='Non Matched Related Terms', index=False)
        df_unrelated_terms.to_excel(writer, sheet_name='Non Matched Unrelated Terms', index=False)

# %%
def categorize_non_matched_terms(non_matched_terms, model="gpt-3.5-turbo", batch_size=1000):
    """
    Categorize non-matched terms using OpenAI API.
    
    Args:
        non_matched_terms (set): Set of non-matched terms
        model (str): OpenAI model to use
        batch_size (int): Batch size for processing
        
    Returns:
        tuple: Tuple of related and unrelated terms
    """
    related_terms = set()
    unrelated_terms = set()
    for batch in batch_terms(non_matched_terms, batch_size):
        prompt = f"Please append (R) as a postfix to any terms that are related, even slightly, to the domain of invasion biology. If it is not related then leave it as it is. Ensure no terms are skipped or inaccurately included. If you think the word it self is invalid add (-) postfix to it. Provide the categorized terms separated by commas: {', '.join(batch)}."
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=4000,
            temperature=0,
        )
        categorized_terms = response.choices[0].message.content
        print(f"Categorized Terms: {categorized_terms}")
        terms = categorized_terms.split(', ')
        for term in terms:
            if re.search(r'\(R\)$', term):
                related_terms.add(term[:-3].strip())  # Remove the (R) postfix and any trailing spaces
            else:
                unrelated_terms.add(term.strip())

    print(f"Related terms: {related_terms}")
    print(f"Unrelated terms: {unrelated_terms}")

    return related_terms, unrelated_terms

def batch_terms(terms, batch_size):
    """
    Batch terms for processing.
    
    Args:
        terms (set): Set of terms to batch
        batch_size (int): Size of each batch
        
    Yields:
        list: Batch of terms
    """
    terms = list(terms)
    for i in range(0, len(terms), batch_size):
        print(i)
        yield terms[i:i + batch_size]

# %%
# Neo4j database integration
from neo4j import GraphDatabase

# Configuration
URI = "neo4j://localhost:7687"
AUTH = ("neo4j", "12345678")
abstract_file = 'coverage_results/abstracts_OWL.csv'
directory = 'hypothesis_excel_files/'

# Connect to Neo4j
driver = GraphDatabase.driver(URI, auth=AUTH)

# Function to clear existing data
def clear_database():
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")

# Function to set graph configuration
def set_graph_config():
    with driver.session() as session:
        session.run("CALL n10s.graphconfig.set({ handleVocabUris: 'SHORTEN', handleMultival: 'ARRAY' })")

# Function to import ontology
def import_ontology():
    with driver.session() as session:
        session.run("CALL n10s.rdf.import.fetch('file://INBIO-1.2.owl', 'RDF/XML')")

# %%
# Function to find non-matched terms with LLM
def find_non_matched_terms_with_llm(all_terms_and_phrases, ontology_terms):
    """
    Find non-matched terms using LLM.
    
    Args:
        all_terms_and_phrases (set): Set of all terms and phrases
        ontology_terms (set): Set of ontology terms
        
    Returns:
        set: Set of non-matched terms
    """
    ontology_terms_list = list(ontology_terms)
    
    # Split the terms into chunks to avoid exceeding the token limit
    chunk_size = 1000  
    all_terms_chunks = [list(all_terms_and_phrases)[i:i + chunk_size] for i in range(0, len(all_terms_and_phrases), chunk_size)]
    
    non_matched_terms = set()
    
    for chunk in all_terms_chunks:
        prompt = (
                    "You are an expert in invasion biology. Your task is to identify terms from the following list "
                    "that are not present in the provided ontology terms. This will help in understanding the coverage "
                    "of the ontology over the given terms.\n\n"
                    f"Terms to check: {', '.join(chunk)}\n\n"
                    f"Ontology terms: {', '.join(ontology_terms_list)}"
                )        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
        )
        
        non_matched_terms.update(response.choices[0].message.content.strip().split(', '))
    
    return non_matched_terms

# %%
# Function to calculate coverage and print non-matched terms
def calculate_coverage_and_print_non_matched_terms(hypothesis_files, directory, ontology_terms, nlp, abstract_file):
    """
    Calculate coverage and print non-matched terms.
    
    Args:
        hypothesis_files (dict): Dictionary of hypothesis files
        directory (str): Directory containing hypothesis files
        ontology_terms (set): Set of ontology terms
        nlp: The spaCy NLP model
        abstract_file (str): Path to the abstract file
        
    Returns:
        pd.DataFrame: DataFrame of coverage results
    """
    abstract_data = pd.read_csv(abstract_file)
    results = {}
    for hypothesis, info in hypothesis_files.items():
        file_path = os.path.join(directory, info['filename'])
        columns_dict = info['columns']
        data = extract_specific_columns(file_path, columns_dict)
        excel_terms_and_phrases = extract_terms_and_phrases_from_data_enhanced(data, nlp)
        statement_terms_and_phrases = extract_terms_and_phrases_from_text(info['statement'], nlp)
        
        # Get all abstracts for the current hypothesis and concatenate them
        abstracts = abstract_data[abstract_data['hypothesis_labels'].astype(str).str.contains(info['label'])]['hypothesis_abstract'].values
        concatenated_abstracts = ' '.join(abstracts) if len(abstracts) > 0 else 'No abstract available'
        abstract_terms_and_phrases = extract_terms_and_phrases_from_text(concatenated_abstracts, nlp)
        
        # Combine all terms and phrases
        all_terms_and_phrases = excel_terms_and_phrases.union(statement_terms_and_phrases).union(abstract_terms_and_phrases)
        
        # Use LLM to find non-matched terms
        non_matched_terms = find_non_matched_terms_with_llm(all_terms_and_phrases, ontology_terms)
        
        covered_terms_and_phrases = ontology_terms.intersection(all_terms_and_phrases)
        coverage_percentage = len(covered_terms_and_phrases) / len(all_terms_and_phrases) * 100 if all_terms_and_phrases else 0
        
        results[hypothesis] = {
            'hypothesis': hypothesis,
            'Coverage Percentage': coverage_percentage,
            'Matched Terms and Phrases': covered_terms_and_phrases,
            'Non-Matched Terms': non_matched_terms,
            'Statement': info['statement'],
            'Abstract': concatenated_abstracts
        }
    
    df = pd.DataFrame.from_dict(results, orient="index")
    df.columns = ['Hypothesis', 'Coverage Percentage', 'Matched Terms and Phrases', 'Non-Matched Terms', 'Statement', 'Abstract']
    df.to_csv('coverage_results_with_abstracts_LLM.csv', index=True)
    return df

# %%
# BioPortal API integration
import requests
from urllib.parse import quote_plus

def get_term_details(term, api_key):
    """
    Get term details from BioPortal.
    
    Args:
        term (str): Term to search for
        api_key (str): BioPortal API key
        
    Returns:
        dict: Dictionary of term details
    """
    url = f"http://data.bioontology.org/search?q={term}&apikey={api_key}"
    response = requests.get(url)
    data = response.json()
    
    if data.get('totalCount', 0) > 0:
        # Process the first result; adjust as needed if you want to handle multiple results
        term_details = data['collection'][0]
        term_info = {
            'uri': term_details['@id'],
            'label': term_details['prefLabel'],
            'definition': term_details.get('definition', ['No definition provided'])[0],
            'ontology': term_details['links']['ontology'],
            'synonyms': term_details.get('synonym', []),
            'relationships': term_details.get('relations', [])
        }
        return term_info
    else:
        return None

# %%
# Function to get term details with preferred ontologies
def get_term_details(term, api_key, preferred_ontologies=None):
    """
    Get term details from BioPortal with preferred ontologies.
    
    Args:
        term (str): Term to search for
        api_key (str): BioPortal API key
        preferred_ontologies (dict): Dictionary of preferred ontologies with scores
        
    Returns:
        dict: Dictionary of term details
    """
    preferred_ontologies = preferred_ontologies or {}
    print(f"\nPreferred Ontologies: {preferred_ontologies}")
    url = f"http://data.bioontology.org/search?q={term}&apikey={api_key}"
    response = requests.get(url)
    data = response.json()
    
    best_match = None
    highest_score = -1  # Initial negative score to ensure any positive score is better
    
    if data.get('totalCount', 0) > 0:
        for term_details in data['collection']:
            score = 0
            ontology_link = term_details['links']['ontology']
            ontology_id = ontology_link.split("/")[-1]  # Extract ontology ID
            print(ontology_id)
            
            # Score increment if in preferred ontology list
            score += preferred_ontologies.get(ontology_id, 0)
            
            # Additional scoring based on information completeness
            if 'definition' in term_details:
                score += 1
            if 'synonym' in term_details:
                score += 1
                
            # term with highest score
            if score > highest_score:
                highest_score = score
                best_match = term_details
                
    if best_match:
        term_info = {
            'uri': best_match['@id'],
            'label': best_match['prefLabel'],
            'definition': best_match.get('definition', ['No definition provided'])[0],
            'ontology': best_match['links']['ontology'],
            'synonyms': best_match.get('synonym', []),
            'relationships': best_match.get('relations', [])
        }
        return term_info
    
    return None

# %%
# Main execution
if __name__ == "__main__":
    # Example usage
    abstract_file = 'abstracts_OWL.csv'
    df_with_abstract, df_abstract_verbs_relationships, df_statement_verbs_relationships, unique_concept_covered_terms_with, unique_concept_non_matched_terms_with, unique_definition_covered_terms_with, unique_definition_non_matched_terms_with = calculate_coverage_with_abstract(hypothesis_files, directory, concept_terms, definitions, nlp, abstract_file)
    
    # Combine and save results
    combine_and_save_results(df_without_abstract, df_with_abstract, df_verbs_relationships_without, df_abstract_verbs_relationships, df_statement_verbs_relationships, 
                            (unique_concept_covered_terms_without, unique_concept_non_matched_terms_without, unique_definition_covered_terms_without, unique_definition_non_matched_terms_without),
                            (unique_concept_covered_terms_with, unique_concept_non_matched_terms_with, unique_definition_covered_terms_with, unique_definition_non_matched_terms_with))