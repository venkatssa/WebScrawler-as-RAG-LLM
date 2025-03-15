import boto3
import json
import numpy as np
import faiss
import time
from typing import List, Dict
from sklearn.metrics.pairwise import cosine_similarity
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Create Bedrock client
session = boto3.Session()
bedrock_runtime = session.client(
    service_name='bedrock-runtime',
    region_name='us-east-1',
)

def get_embeddings(text: str) -> List[float]:
    """Get embeddings using Bedrock's embedding model."""
    for _ in range(3):  # Retry mechanism
        try:
            response = bedrock_runtime.invoke_model(
                modelId="amazon.titan-embed-text-v2:0",
                body=json.dumps({"inputText": text})
            )
            response_body = json.loads(response['body'].read())
            return response_body['embedding']
        except Exception as e:
            print("Error fetching embeddings, retrying...", e)
            time.sleep(2)
    raise Exception("Failed to fetch embeddings after retries.")

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 200) -> List[str]:
    """Splits text into semantically meaningful chunks using RecursiveTextSplitter."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""],  # Prioritizes paragraphs, then sentences
    )
    return text_splitter.split_text(text)

def fetch_and_store_scraped_data(scraped_data: List[Dict]):
    """Process scraped data (URL, title, description, and content) and store in FAISS."""
    if not scraped_data:
        print("No data to process.")
        return None, []
    
    embeddings_list = []
    metadata_list = []
    
    for data in scraped_data:
        if 'error' not in data and data.get('content'):
            text_chunks = chunk_text(data['content'])
            # print("------",text_chunks)
            for chunk in text_chunks:
                text_data = f"{data['url']} {data['title']} {data['description']} {chunk}"
                embedding = get_embeddings(text_data)
                embeddings_list.append(embedding)
                metadata_list.append({
                    "url": data['url'],
                    "title": data['title'],
                    "description": data['description'],
                    "content": chunk,
                    "embedding": embedding
                })   
    
    if not embeddings_list:
        print("No valid embeddings generated.")
        return None, []
    
    embeddings_array = np.array(embeddings_list, dtype='float32')
    d = embeddings_array.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings_array)
    faiss.write_index(index, "faiss_index.index")
    print("FAISS database created successfully with semantic chunking!")
    return index, metadata_list

def retrieve_and_rerank(query_text: str, index, metadata, k: int = 5, top_n: int = 3, min_similarity: float = 0.01):
    """Retrieve top-k similar results from FAISS and rerank using cosine similarity."""
    if index is None:
        print("[ERROR] FAISS index not found.")
        return []
    
    query_embedding = np.array(get_embeddings(query_text)).astype('float32').reshape(1, -1)
    _, indices = index.search(query_embedding, k)
    retrieved_docs = [metadata[i] for i in indices[0] if i < len(metadata)]
    
    if not retrieved_docs:
        print("[INFO] No matching documents found.")
        return []
    
    # Extract embeddings and compute cosine similarity
    doc_embeddings = np.array([doc['embedding'] for doc in retrieved_docs])
    similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
    
    # Attach similarity scores and print for debugging
    for i, doc in enumerate(retrieved_docs):
        doc['similarity'] = similarities[i]
    
    print("[DEBUG] Retrieved Documents with Similarity Scores:")
    for doc in retrieved_docs:
        print(f"Title: {doc['title']}, Similarity: {doc['similarity']:.4f}")
    
    # Filter low-relevance results and sort
    retrieved_docs = [doc for doc in retrieved_docs if doc['similarity'] >= min_similarity]
    retrieved_docs = sorted(retrieved_docs, key=lambda x: x['similarity'], reverse=True)
    
    return retrieved_docs[:top_n]

def query_llm(query_text: str, retrieved_docs: List[dict]):
    """Send input query and retrieved document to LLM."""
    if not retrieved_docs:
        return "No relevant information found."

    # Use only the first (most relevant) retrieved document
    top_doc = retrieved_docs[0]  
    context = f"URL: {top_doc['url']}\nTitle: {top_doc['title']}\nDescription: {top_doc['description']}\nContent: {top_doc['content'][:500]}..."
    source_url = top_doc['url']

    formatted_prompt = f"User Query: {query_text}\n\nRelevant Article:\n{context}\n\nGenerate a concise response based on the query and article. Ensure accuracy."
    
    request_body = {
        "inferenceConfig": {"max_new_tokens": 500},
        "messages": [{"role": "user", "content": [{"text": formatted_prompt}]}]
    }
    for _ in range(3):  # Retry mechanism
        try:
            response = bedrock_runtime.invoke_model(
                modelId="amazon.nova-lite-v1:0",
                body=json.dumps(request_body)
            )
            response_body = json.loads(response['body'].read())

            # Extract the response text safely
            llm_output = response_body.get("output", {}).get("message", {}).get("content", [{}])
            llm_response = llm_output[0].get("text", "No response") if isinstance(llm_output, list) else "No response"
            
            return f"{llm_response}\n\n🔗 Source: {source_url}"
        except Exception as e:
            print("Error querying LLM, retrying...", e)
            time.sleep(2)
    
    raise Exception("Failed to query LLM after retries.")

