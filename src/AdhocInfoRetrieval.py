from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
import re
from collections import defaultdict, Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

with open("../data/cran.qry", 'r') as file:
    cran_qry_content = file.read()

with open('../data/stop_list.py', 'r') as file:
    stop_list_content = file.read()

with open("../data/cran.all.1400", 'r') as file:
    cran_all_content = file.read()

# Changing the text we extracted into a format that is desired.
start = stop_list_content.find('[')
end = stop_list_content.find(']', start) + 1  # Search after the start
stop_word_list = eval(stop_list_content[start:end])

# Clean and tokenize text


def clean_tokenize(text, stemmer=PorterStemmer()):
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Remove number
    text = text.translate(str.maketrans(
        '', '', string.punctuation))  # Remove punctuation
    tokens = word_tokenize(text)
    # Remove stop_words
    tokens = [word for word in tokens if word not in stop_word_list]
    # Apply stemming
    tokens = [stemmer.stem(word) for word in tokens]
    return tokens

# Function to parse cran.qry(QUERY) content


def parse_cran_qry(content):
    queries = {}
    current_id = ""
    lines = content.split('\n')
    for line in lines:
        if line.startswith('.I'):  # Start of new query
            current_id = line.split(' ')[1]
            queries[current_id] = ""
        elif line.startswith('.W'):
            continue
        else:
            queries[current_id] += line + " "
    return queries


# Parse cran.qry
queries = parse_cran_qry(cran_qry_content)

# Clean and tokenize each query
tokenized_qry = {qid: clean_tokenize(query) for qid, query in queries.items()}

# Function to parse .W part (abstracts) from cran.all.1400


def parse_cran_abstracts(content):
    abstracts = {}
    current_id = ""
    lines = content.split('\n')
    in_abstract = False
    for line in lines:
        if line.startswith('.I'):
            current_id = line.split(' ')[1]
            in_abstract = False
        elif line.startswith('.W'):
            in_abstract = True
            abstracts[current_id] = ""
        elif in_abstract:
            abstracts[current_id] += line + " "
    return abstracts


# Parse the abstracts from cran.all.1400
abstracts = parse_cran_abstracts(cran_all_content)

# Clean and tokenize each abstract
tokenized_abstr = {doc_id: clean_tokenize(
    abstract) for doc_id, abstract in abstracts.items()}

# Convert tokenized queries and abstracts back to string format for TF-IDF vectorization
queries_str = [" ".join(tokens)
               for tokens in tokenized_qry.values()]  # For each query
abstracts_str = [" ".join(tokens)
                 for tokens in tokenized_abstr.values()]  # For each abstract

# Combine queries and abstracts for TF-IDF vectorization (to ensure vocabulary matches)
combined_texts = queries_str + abstracts_str

# Initialize and fit TF-IDF vectorizer
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(combined_texts)

# Split TF-IDF matrix back into queries and abstracts
tfidf_queries = tfidf_matrix[:len(queries_str)]
tfidf_abstracts = tfidf_matrix[len(queries_str):]

# Compute cosine similarity between each query and all abstracts
cosine_similarities = cosine_similarity(tfidf_queries, tfidf_abstracts)

output_lines = []

# Minimum number of matches to output for each query
min_matches = 100

# Iterate over each query's cosine similarities with abstracts
for query_index, similarities in enumerate(cosine_similarities):
    # Sort the indices of abstracts based on similarity scores in descending order
    sorted_indices = np.argsort(similarities)[::-1]
    # Filter out indices where similarity score is 0, but ensure at least 100
    non_zero_indices = [
        index for index in sorted_indices if similarities[index] > 0]

    # If fewer than min_matches, append indices with zero similarity
    if len(non_zero_indices) < min_matches:
        zero_indices = [
            index for index in sorted_indices if similarities[index] == 0]
        required_zeros = min_matches - len(non_zero_indices)
        non_zero_indices.extend(zero_indices[:required_zeros])

    # Iterate over filtered and possibly extended indices to prepare lines for the output file
    for rank, abstract_index in enumerate(non_zero_indices):
        # Format : query_id, abstract_id, cosine_similarity_score
        line = f"{query_index + 1} {abstract_index + 1} {similarities[abstract_index]:.12f}"
        output_lines.append(line)

output_file_path = 'output.txt'
with open(output_file_path, 'w') as file:
    # Write each line to the file
    for line in output_lines:
        file.write(line + '\n')  # Each entry appears on a new line
