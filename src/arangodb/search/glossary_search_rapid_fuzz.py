import string
import time
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
from arango import ArangoClient
from rapidfuzz import fuzz, process

from shared_utils.embed_collections import embed_glossary_entries

def preprocess_text(text):
    """Remove punctuation and convert to lowercase."""
    return text.translate(str.maketrans("", "", string.punctuation)).lower()

async def load_glossary_from_arango(use_collection_all=False):
    """
    Load documents from ArangoDB.
    If use_collection_all is True, use collection.all().
    Otherwise, use a simple AQL query.
    """
    client = ArangoClient(hosts='http://localhost:8529')
    db = await asyncio.to_thread(
        client.db,
        'verifaix', username='root', password='openSesame'
    )
    glossary_collection = db.collection('glossary')

    await embed_glossary_entries(db)

    batch_size = 100
    glossary = []

    if use_collection_all:
        # Option A: Use collection.all() 
        cursor = await asyncio.to_thread(glossary_collection.all)
        async for doc in cursor:
            glossary.append({
                "term": doc["term"],
                "definition": doc["definition"],
                "processed_term": preprocess_text(doc["term"])
            })
    else:
        # Option B: Use an AQL query
        cursor = await asyncio.to_thread(
            db.aql.execute,
            f"FOR doc IN {glossary_collection.name} RETURN doc",
            batch_size=batch_size
        )
        async for doc in cursor:
            glossary.append({
                "term": doc["term"],
                "definition": doc["definition"],
                "processed_term": preprocess_text(doc["term"])
            })
    return glossary

def find_longest_glossary_item(question, glossary, threshold=97):
    """
    Returns a list with exactly zero or one item.
    If a match meets the threshold, we return it as a single-element list.
    Otherwise, we return an empty list.
    """
    processed_question = preprocess_text(question)

    # Extract matches using RapidFuzz
    choices = [entry["processed_term"] for entry in glossary]
    results = process.extract(
        processed_question,
        choices,
        scorer=fuzz.token_sort_ratio,
        limit=None
    )

    # Filter matches to find the longest one among those that exceed threshold
    longest_match = None
    longest_length = 0
    for match_text, score, idx in results:
        if score >= threshold:
            term_length = len(glossary[idx]["term"])
            if term_length > longest_length:
                longest_length = term_length
                longest_match = {
                    "term": glossary[idx]["term"],
                    "definition": glossary[idx]["primary_definition"],
                    "score": score
                }

    # Return a list so we don't cause iteration errors in main()
    return [longest_match] if longest_match else []

def process_questions(user_questions, glossary):
    """Submits questions to a ThreadPool, each calling find_longest_glossary_item()."""
    results = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(find_longest_glossary_item, question, glossary)
            for question in user_questions
        ]
        for future in futures:
            results.append(future.result())
    return results

def process_questions_in_batches(user_questions, glossary, batch_size=100):
    """
    Batches the user questions to avoid sending thousands of threads/futures at once.
    Also times each batch for profiling.
    """
    all_results = []
    for i in range(0, len(user_questions), batch_size):
        batch = user_questions[i : i + batch_size]
        start_time = time.time()
        batch_results = process_questions(batch, glossary)
        end_time = time.time()

        execution_time = end_time - start_time
        print(f"Batch {i // batch_size + 1} executed in {execution_time:.4f} seconds")

        all_results.extend(batch_results)
    return all_results

def main():
    # Load from ArangoDB (pick whichever method you want)
    glossary = load_glossary_from_arango(use_collection_all=True)

    # Simulate 5000 questions
    user_questions = [
        "What is an aple?",
        "Tell me about bananana.",
        "What are the benefits of oranges?",
        "Is an apple a fruit?",
        "Can you explain what a bannana is?",
    ] * 1000

    results = process_questions_in_batches(user_questions, glossary, batch_size=100)

    # Prepare a JSON-friendly output
    json_results = []
    for i, question in enumerate(user_questions):
        # results[i] is guaranteed to be a list (either empty or with one dict)
        matches = results[i]
        question_matches = []
        for match in matches:
            question_matches.append({
                "term": match["term"],
                "definition": match["definition"],
                "score": match["score"]
            })
        json_results.append({
            "question": question,
            "matches": question_matches
        })

    # Print JSON results for demonstration
    print(json.dumps(json_results, indent=2))

if __name__ == "__main__":
    main()
