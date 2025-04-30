import rdflib

g = rdflib.Graph()
g.parse("INBIO-1.2.owl", format="xml")

# Query to count classes
query_classes = """
SELECT (COUNT(DISTINCT ?class) AS ?numClasses)
WHERE {
  ?class a owl:Class .
}
"""
num_classes = g.query(query_classes)
for row in num_classes:
    print(f"Number of classes: {row.numClasses}")

# Query to count object properties
query_object_properties = """
SELECT (COUNT(DISTINCT ?property) AS ?numObjectProperties)
WHERE {
  ?property a owl:ObjectProperty .
}
"""
num_object_properties = g.query(query_object_properties)
for row in num_object_properties:
    print(f"Number of object properties: {row.numObjectProperties}")

# Query to count data properties
query_data_properties = """
SELECT (COUNT(DISTINCT ?property) AS ?numDataProperties)
WHERE {
  ?property a owl:DataProperty .
}
"""
num_data_properties = g.query(query_data_properties)
for row in num_data_properties:
    print(f"Number of data properties: {row.numDataProperties}")

# Query to list all classes and their properties
query_class_details = """
SELECT DISTINCT ?class ?property
WHERE {
    ?class a owl:Class .
    OPTIONAL { ?class ?property ?value }
    FILTER (?property != rdf:type)
}
ORDER BY ?class
"""
class_details = g.query(query_class_details)
print("\nClasses and their properties:")
for row in class_details:
    print(f"Class: {row['class']}, Property: {row['property']}")


def execute_query(graph, query):
    return [row for row in graph.query(query)]

# Total axioms count
query_total_axioms = """
SELECT (COUNT(*) AS ?count)
WHERE {
  ?s ?p ?o .
}
"""
total_axioms = execute_query(g, query_total_axioms)[0][0]

# Logical axioms count (assuming all triples are logical axioms)
logical_axioms = total_axioms

# Declaration axioms count
query_declaration_axioms = """
SELECT (COUNT(*) AS ?count)
WHERE {
  ?s rdf:type owl:Class .
}
"""
declaration_axioms = execute_query(g, query_declaration_axioms)[0][0]

# Class count
query_class_count = """
SELECT (COUNT(DISTINCT ?class) AS ?count)
WHERE {
  ?class rdf:type owl:Class .
}
"""
class_count = execute_query(g, query_class_count)[0][0]

# Object property count
query_object_property_count = """
SELECT (COUNT(DISTINCT ?property) AS ?count)
WHERE {
  ?property rdf:type owl:ObjectProperty .
}
"""
object_property_count = execute_query(g, query_object_property_count)[0][0]

# Data property count
query_data_property_count = """
SELECT (COUNT(DISTINCT ?property) AS ?count)
WHERE {
  ?property rdf:type owl:DataProperty .
}
"""
data_property_count = execute_query(g, query_data_property_count)[0][0]

# Individual count
query_individual_count = """
SELECT (COUNT(DISTINCT ?individual) AS ?count)
WHERE {
  ?individual rdf:type owl:NamedIndividual .
}
"""
individual_count = execute_query(g, query_individual_count)[0][0]

# Annotation property count
query_annotation_property_count = """
SELECT (COUNT(DISTINCT ?property) AS ?count)
WHERE {
  ?property rdf:type owl:AnnotationProperty .
}
"""
annotation_property_count = execute_query(g, query_annotation_property_count)[0][0]

# Print results
print(f"Total Axioms: {total_axioms}")
print(f"Logical Axioms: {logical_axioms}")
print(f"Declaration Axioms: {declaration_axioms}")
print(f"Class Count: {class_count}")
print(f"Object Property Count: {object_property_count}")
print(f"Data Property Count: {data_property_count}")
print(f"Individual Count: {individual_count}")
print(f"Annotation Property Count: {annotation_property_count}")