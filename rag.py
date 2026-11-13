import os
import re
import math
from collections import Counter
import requests
import numpy as np
from pypdf import PdfReader

#Step 1: Load and extract text from local PDFs
PDF_DIR = "./pdfpath"

def load_files(folder_path):
    documents = []
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Directory `{folder_path} created. Please put inside your PDFs")
        return documents
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            filepath = os.path.join(folder_path, filename)
            try:
                reader = PdfReader(filepath)
                full_text = ""
                for page_num, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text:
                        full_text += text + "\n"
                if full_text.strip():
                    documents.append({"source": filename, "text": full_text})
                    print(f"[Loaded {filename} ({len(full_text)} characters])")
                else:
                    print(f"[Warning {filename} ({len(full_text)}) contained no text.]")
            except Exception as e:
                print(f"[Error] Failed to read {filename}: {e}")
    return documents

#Step 2 : Tokenization & Text chunking
def tokenize(text):
    text = text.lower()
    return re.findall(r'\b[a-zA-ZΑ-Ωα-ω0-9]{3,}\b', text)

def chunk_text(text, source_name, max_words=100, overlap=20):
    words = tokenize(text)
    chunks = []
    for i in range(0, len(words), max_words - overlap):
        chunk_words = words[i:i + max_words]
        if len(chunk_words) > 10:
            chunk_text_str = " ".join(chunk_words)
            chunks.append({"source": source_name, "content":chunk_text_str})
    return chunks

#Step 3 : scratch vector space & embeddings(TF-IDF Vectorizer)
class ScratchVectorizer:
    def __init__(self):
        self.vocabulary = {}
        self.vocab_list = []
        self.idf = None
        self.N = 0
    
    def fit(self, corpus_chunks):
        vocab_set = set()
        for chunk in corpus_chunks:
            for word in tokenize(chunk["content"]):
                vocab_set.add(word)
        self.vocab_list = sorted(list(vocab_set))
        self.vocabulary = {word: idx for idx, word in enumerate(self.vocab_list)}
        vocab_size = len(self.vocab_list)

        self.N = len(corpus_chunks)
        df = np.zeros(vocab_size)
        for chunk in corpus_chunks:
            unique_words = set(tokenize(chunk["content"]))
            for word in unique_words:
                if word in self.vocabulary:
                    df[self.vocabulary[word]] += 1

        self.idf = np.log((self.N + 1.0) / (df + 1.0)) + 1.0
        print(f"\n[Vectorizer Built] Vocabulary size: {vocab_size} unique terms across {self.N} chunks.")

    def transform(self, text):
        words = tokenize(text)
        word_counts = Counter(words)
        vector = np.zeros(len(self.vocab_list))
        total_words = len(words) if len(words) > 0 else 1
        for word, count in word_counts.items():
            if word in self.vocabulary:
                idx = self.vocabulary[word]
                tf = count / total_words
                vector[idx] = tf * self.idf[idx]
        return vector

#Step 4: Cosine similarity & Retrieval engine
def cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)

def retrieve_top_k(query, vectorizer, chunk_embeddings, chunks, k=3):
    query_vector = vectorizer.transform(query)
    similarities = []
    for idx, chunk_emb in enumerate(chunk_embeddings):
        sim = cosine_similarity(query_vector, chunk_emb)
        similarities.append((sim, idx))
    similarities.sort(key=lambda x: x[0], reverse=True)
    results = []
    for sim, idx in similarities[:k]:
        results.append({
            "score": sim,
            "source": chunks[idx]["source"],
            "content": chunks[idx]["content"]
        })
    return results

#Step 5: the LLM 
def ask_llm(query, retrieved_chunks):
    """Στέλνει το context και την ερώτηση στο LM Studio μέσω του OpenAI-compatible API"""
    
    # 1. Δόμηση του context
    context_text = "\n\n".join([f"Source: {chunk['source']}\n{chunk['content']}" for chunk in retrieved_chunks])
    
    # 2. LM Studio URL (η προεπιλεγμένη πόρτα είναι η 1234)
    url = "http://localhost:1234/v1/chat/completions"
    
    # 3. Payload σε OpenAI format (System + User messages)
    payload = {
        "model": "liquid/lfm2-1.2b",  # Το όνομα του μοντέλου που έχεις φορτώσει στο LM Studio
        "messages": [
            {
                "role": "system", 
                "content": "You are a precise research assistant. Answer the user's question using ONLY the provided context below. If the answer cannot be found in the context, state 'I cannot find the answer in the provided documents.'"
            },
            {
                "role": "user", 
                "content": f"Context:\n{context_text}\n\nUser Question: {query}"
            }
        ],
        "temperature": 0.1,
        "stream": False
    }
    
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            result_json = response.json()
            # Εξαγωγή της απάντησης από το OpenAI-compatible response structure
            return result_json["choices"][0]["message"]["content"]
        else:
            return f"Error from LM Studio: {response.text}"
    except Exception as e:
        return f"Could not connect to LM Studio server: {e}"

    
if __name__ == "__main__":
    raw_docs = load_files("./pdfpath")
    if raw_docs:
        all_chunks = []
        for doc in raw_docs:
            all_chunks.extend(chunk_text(doc["text"], doc["source"], max_words=80, overlap=15))

        vectorizer = ScratchVectorizer()
        vectorizer.fit(all_chunks)
        chunk_embeddings = np.array([vectorizer.transform(chunk["content"]) for chunk in all_chunks])
        
        print("\n" + "="*60)
        print("The app is ready...")
        print("Type 'exit' or 'quit' to leave the app...")
        print("="*60 + "\n")

        while True:
            try:
                user_query = input("What can i do for you today? \n").strip()
            
                if not user_query:
                    continue
                
                if user_query.lower() in ["exit", "quit"]:
                    print("\nExiting the app...")
                    break
            
                print("Searching in the given PDFs...")
                top_results = retrieve_top_k(user_query, vectorizer, chunk_embeddings, all_chunks, k=2)
                for r in top_results:
                    print(f"   -> [Match: {r['source']} | Score: {r['score']:.4f}]")
                
                print("Producing the LLM response...")
                final_answer = ask_llm(user_query, top_results)
            
                print("\n" + "-"*50)
                print("Answer: ")
                print(final_answer)
                print("-"*50 + "\n")
            
            except KeyboardInterrupt:
                print("\n Exiting...")
                break