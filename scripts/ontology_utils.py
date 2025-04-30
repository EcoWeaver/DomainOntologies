# ontology_utils.py

import logging
from rdflib import Graph, URIRef, BNode
from rdflib.namespace import RDF, RDFS, OWL

logger = logging.getLogger(__name__)


def get_formatted_hierarchy(owl_file_path):
    """Generates a formatted ontology hierarchy string from an OWL file."""
    class_hierarchy = fetch_class_hierarchy(owl_file_path)
    formatted_hierarchy = ""
    for cls, relations in class_hierarchy.items():
        formatted_hierarchy += f"Class: {cls}\n"
        if relations['superclasses']:
            formatted_hierarchy += f"  Superclasses: {', '.join(relations['superclasses'])}\n"
        if relations['subclasses']:
            formatted_hierarchy += f"  Subclasses: {', '.join(relations['subclasses'])}\n"
        formatted_hierarchy += "\n"
    return formatted_hierarchy


def fetch_class_hierarchy(owl_file_path):
    """Fetches class hierarchy from an OWL file."""
    g = Graph()
    try:
        g.parse(owl_file_path)
    except Exception as e:
        logger.error(f"Failed to load or parse the OWL file: {e}")
        return {}

    class_hierarchy = {}

    classes = set(g.subjects(RDF.type, RDFS.Class))
    classes.update(g.subjects(RDF.type, OWL.Class))

    if not classes:
        logger.warning("No classes found in the OWL file.")
        return {}

    for class_ref in classes:
        if isinstance(class_ref, BNode):
            continue
        class_label = get_label(g, class_ref)
        if class_label not in class_hierarchy:
            class_hierarchy[class_label] = {'subclasses': set(), 'superclasses': set()}

        # Collect labels of subclasses
        for subj in g.subjects(RDFS.subClassOf, class_ref):
            if isinstance(subj, BNode):
                continue
            subclass_label = get_label(g, subj)
            class_hierarchy[class_label]['subclasses'].add(subclass_label)

        # Collect labels of superclasses
        for obj in g.objects(class_ref, RDFS.subClassOf):
            if isinstance(obj, BNode):
                continue
            superclass_label = get_label(g, obj)
            class_hierarchy[class_label]['superclasses'].add(superclass_label)

    # Convert sets to lists for serialization
    for cls in class_hierarchy:
        class_hierarchy[cls]['subclasses'] = list(class_hierarchy[cls]['subclasses'])
        class_hierarchy[cls]['superclasses'] = list(class_hierarchy[cls]['superclasses'])

    return class_hierarchy


def get_label(g, class_ref):
    """Gets the label of a class."""
    label = None
    for l in g.objects(class_ref, RDFS.label):
        label = str(l)
        break  # Use the first label found
    if label is None:
        # No label found, extract local name from the URI
        if isinstance(class_ref, URIRef):
            label = class_ref.split('#')[-1] if '#' in class_ref else class_ref.split('/')[-1]
        else:
            label = str(class_ref)
    return label