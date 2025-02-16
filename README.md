# Ad Hoc Information Retrieval System using TF-IDF and Cosine Similarity

## Overview
This project implements an **Ad Hoc Information Retrieval System** based on **TF-IDF weighting** and **cosine similarity**. The system is designed to process a set of **queries** and **document abstracts**, compute their **TF-IDF representations**, and rank the documents based on their **cosine similarity** to each query. The dataset used is the **Cranfield Collection**, which contains **225 queries** and **1400 abstracts** from aerodynamics journal articles.

## Features
- **Tokenization and Preprocessing:**
  - Removes punctuation, numbers, and stop words
  - Applies stemming using **PorterStemmer**
- **TF-IDF Computation:**
  - Computes **Inverse Document Frequency (IDF)** for terms in queries and abstracts
  - Generates **TF-IDF vectors** for queries and abstracts
- **Cosine Similarity Calculation:**
  - Computes similarity between each query and all abstracts
  - Ranks abstracts for each query
- **Output Generation:**
  - Outputs ranked lists of **query-document** pairs with similarity scores
  - Formats output to match the expected submission format for grading

## Dataset Files
- **cran.qry**: Contains **225 queries** in a structured format
- **cran.all.1400**: Contains **1400 abstracts** with metadata
- **cranqrel**: Answer key mapping queries to relevant abstracts
- **stop_list.py**: List of stop words to be removed during processing

## Prerequisites
Ensure you have **Python 3.x** installed and the following libraries:

```sh
pip install nltk scikit-learn numpy
```

**Output Generation:**
   - Results will be saved in **output.txt**
   - Each line represents a **query-document** pair ranked by cosine similarity:
     ```
     <query_id> <abstract_id> <cosine_similarity>
     ```

## File Structure
```
├── AdhocInfoRetrieval.py      # Main script for information retrieval
├── cran.qry                   # Query dataset
├── cran.all.1400              # Abstracts dataset
├── cranqrel                   # Answer key
├── stop_list.py               # List of stop words
├── output.txt                 # Output file containing ranked results
├── README.md                  # Project documentation
```

## Output Format
Each line in `output.txt` follows this format:
```
<query_id> <abstract_id> <cosine_similarity>
```
Example output:
```
1 51 0.329346411961
1 359 0.222216450587
1 879 0.196560894061
```

## Evaluation Metric: Mean Average Precision (MAP)
The system is evaluated using **Mean Average Precision (MAP)**, computed as follows:
1. **Compute Precision at 10% recall intervals (10% - 100%)**
2. **Average precision scores across recall levels for each query**
3. **Compute the final MAP score by averaging across all queries**

A **MAP score of 20%** is typical, with **higher scores indicating better retrieval performance**.

## Potential Improvements
- **Stemming/Lemmatization:** To improve term matching
- **Length-normalized TF scores:** Adjust for document length variations
- **Alternative Similarity Measures:** Consider **Jaccard similarity** or **BM25 scoring**
- **Expansion of Query Terms:** Using **synonyms or embeddings** to enhance retrieval

## Author
**Chan Hyun Yoo**

