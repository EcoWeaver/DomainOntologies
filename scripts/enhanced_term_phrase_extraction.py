
def extract_terms_and_phrases_from_text(text, nlp):
    doc = nlp(text)
    terms = set(token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct and len(token.lemma_) > 2)
    phrases = set(chunk.text.lower() for chunk in doc.noun_chunks)
    all_terms = terms.union(phrases)
    return all_terms

def extract_terms_and_phrases_from_data_enhanced(data, nlp):
    terms_and_phrases = set()
    for df in data.values():
        for column in df.columns:
            for text in df[column].astype(str):
                terms_and_phrases.update(extract_terms_and_phrases_from_text(text, nlp))
    return terms_and_phrases




# Function to calculate coverage and print non-matched terms
def calculate_coverage_and_print_non_matched_terms(hypothesis_files, directory, ontology_terms, nlp):
    results = {}
    for hypothesis, info in hypothesis_files.items():
        file_path = os.path.join(directory, info['filename'])
        columns_dict = info['columns']
        data = extract_specific_columns(file_path, columns_dict)
        excel_terms_and_phrases = extract_terms_and_phrases_from_data_enhanced(data, nlp)
        statement_terms_and_phrases = extract_terms_and_phrases_from_text(info['statement'], nlp)
        all_terms_and_phrases = excel_terms_and_phrases.union(statement_terms_and_phrases)
        covered_terms_and_phrases = ontology_terms.intersection(all_terms_and_phrases)
        non_matched_terms_in_hypothesis = all_terms_and_phrases.difference(ontology_terms)
        non_matched_terms_in_ontology = ontology_terms.difference(all_terms_and_phrases)
        coverage_percentage = len(covered_terms_and_phrases) / len(all_terms_and_phrases) * 100 if all_terms_and_phrases else 0
        results[hypothesis] = {
            'hypothesis': hypothesis,
            'Coverage Percentage': coverage_percentage,
            'Matched Terms and Phrases': covered_terms_and_phrases,
            'Non-Matched Terms': non_matched_terms_in_hypothesis,
            # 'Non-Matched Terms in Ontology': non_matched_terms_in_ontology,
            'Statement': info['statement']
        }
        

    df = pd.DataFrame.from_dict(results, orient="index")
    df.columns = ['Hypothesis', 'Coverage Percentage', 'Matched Terms and Phrases', 'Non-Matched Terms', 'Statement']

    
    df.to_csv('hypothesis_coverage_results_with_noun_intersection.csv', index=True)


    return df

coverage_results_and_non_matched_terms = calculate_coverage_and_print_non_matched_terms(hypothesis_files, directory, ontology_terms, nlp)

# for index, row in coverage_results_and_non_matched_terms.iterrows():
#     print(f"Hypothesis: {index}")
#     print(f"Statement: {row['Statement']}")
#     print(f"Coverage Percentage: {row['Coverage Percentage']:.2f}%")
#     print(f"Matched Terms and Phrases: {row['Matched Terms and Phrases']}")
#     print(f"Non-Matched Terms in Hypothesis: {row['Non-Matched Terms in Hypothesis']}")

    # print(f"Non-Matched Terms in Ontology: {result['Non-Matched Terms in Ontology']}\n")

coverage_results_and_non_matched_terms

coverage_results_and_non_matched_terms

list(coverage_results_and_non_matched_terms.iloc[0, 2])


import matplotlib.pyplot as plt

# Assuming 'coverage_results_and_non_matched_terms' is the result from the previous function
hypothesis_names = []
non_matched_counts = []

for hypothesis, result in coverage_results_and_non_matched_terms.items():
    hypothesis_names.append(hypothesis)
    non_matched_counts.append(len(result['Non-Matched Terms in Hypothesis']))

# Creating the bar chart
plt.figure(figsize=(10, 8))
plt.barh(hypothesis_names, non_matched_counts, color='skyblue')
plt.xlabel('Count of Non-Matched Terms')
plt.ylabel('Hypothesis')
plt.title('Non-Matched Terms in Hypothesis Data')
plt.tight_layout()

# Display the plot
plt.show()

def df_to_markdown_table_universal(df):

    
    # Make a copy of the DataFrame to avoid modifying the original data
    df_copy = df.copy()
    
    # Round all float columns to 2 decimal places
    float_cols = df_copy.select_dtypes(include=['float']).columns
    df_copy[float_cols] = df_copy[float_cols].round(2)
    
    # Construct the Markdown table
    header = "| " + " | ".join(df_copy.columns) + " |"
    separator = "|---" * len(df_copy.columns) + "|"
    rows = ["| " + " | ".join([str(item) for item in row]) + " |" for index, row in df_copy.iterrows()]
    markdown_table = "\n".join([header, separator] + rows)
    
    return markdown_table

# Convert DataFrame to Markdown Table

# Print the Markdown Table
print(df_to_markdown_table_universal(coverage_results_and_non_matched_terms))

