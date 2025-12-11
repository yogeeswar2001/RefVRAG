from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import faiss

tokenizer = AutoTokenizer.from_pretrained('facebook/contriever')
model = AutoModel.from_pretrained('facebook/contriever')

def text_to_vector(text, max_length=512):
    token_inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=max_length)
    with torch.no_grad():
        model_outputs = model(**token_inputs)
    return model_outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

def retrieve_documents_with_dynamic(documents, queries, similarity_cutoff=0.4):
    if isinstance(queries, list):
        query_vecs = np.array([text_to_vector(q) for q in queries])
        avg_query_vec = np.mean(query_vecs, axis=0)
        normalized_query = avg_query_vec / np.linalg.norm(avg_query_vec)
        normalized_query = normalized_query.reshape(1, -1)
    else:
        normalized_query = text_to_vector(queries)
        normalized_query = normalized_query / np.linalg.norm(normalized_query)
        normalized_query = normalized_query.reshape(1, -1)

    doc_vecs = np.array([text_to_vector(d) for d in documents])
    doc_vecs = doc_vecs / np.linalg.norm(doc_vecs, axis=1, keepdims=True)

    embed_dim = doc_vecs.shape[1]
    vector_index = faiss.IndexFlatIP(embed_dim)
    vector_index.add(doc_vecs)

    limits, scores, indices = vector_index.range_search(normalized_query, similarity_cutoff)
    start_pos = limits[0]
    end_pos = limits[1]
    indices = indices[start_pos:end_pos]

    if len(indices) == 0:
        selected_docs = []
        matched_idx = []
    else:
        matched_idx = indices.tolist()
        selected_docs = [documents[i] for i in matched_idx]

    return selected_docs, matched_idx
