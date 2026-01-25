import numpy as np
from sentence_transformers import SentenceTransformer, util

class CosineSimilarity:
    def __init__(self):
        print("CosineSimilarity: Loading embedding model")
        
        self.model = SentenceTransformer('all-MiniLM-L6-v2') 
       
        
    def get_embeddings(self, texts:list[str]) -> np.ndarray:
        
        # Batch processing through GPU for better performance
        batch_size = 1024  # Adjust based on GPU memory
        embeddings = []
        
        # print('  Encoding contexts into embedding space using local GPU')
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_embeddings = self.model.encode(batch, convert_to_tensor=True)
            embeddings.append(batch_embeddings.detach().cpu().numpy())
        
        # Concatenate all batches
        embeddings = np.concatenate(embeddings, axis=0)
        return embeddings

    def get_similarity(self, src:str, tgt: str) -> float:
        src_embedding = self.get_embeddings([src])
        tgt_embedding = self.get_embeddings([tgt])

        similarity = util.pytorch_cos_sim(src_embedding, tgt_embedding)
        similarity = similarity.cpu().numpy()

        return float(similarity)


    def get_similarities(self, src:str, contexts: list[str]) -> np.ndarray:
        # Batch processing through GPU for better performance
        embeddings = []

        src_embedding = self.get_embeddings([src])
        embeddings = self.get_embeddings(contexts)
        similarities = util.pytorch_cos_sim(src_embedding, embeddings)
        similarities = similarities.cpu().numpy()

        return similarities
    

    def get_similarity_matrix(self, contexts1: list[str], contexts2: list[str], mask_lower_triangular: bool = True) -> np.ndarray:
        embeddings1 = self.get_embeddings(contexts1)
        embeddings2 = self.get_embeddings(contexts2)

        similarities = util.pytorch_cos_sim(embeddings1, embeddings2)
        similarities = similarities.cpu().numpy()

        if mask_lower_triangular:
            mask = np.zeros_like(similarities, dtype=bool)
            
            for i in range(len(contexts1)):
                for j in range(len(contexts2)):
                    if i <= j:
                        mask[i, j] = True
            similarities = similarities * mask

        return similarities

    


def get_duplicate_contexts_embedding_cosine(contexts: list[str], similarity_threshold:float=0.95, model=None):
    # print("Finding duplicate contexts using embedding cosine similarity > %f" % similarity_threshold)
    
    if model is None:
        # print("  Loading embedding model")
        #model = SentenceTransformer('all-MiniLM-L6-v2')
        model = SentenceTransformer("intfloat/e5-mistral-7b-instruct")
    # Batch processing through GPU for better performance
    batch_size = 1024  # Adjust based on GPU memory
    embeddings = []
    
    # print('  Encoding contexts into embedding space using local GPU')
    for i in range(0, len(contexts), batch_size):
        batch = contexts[i:i+batch_size]
        batch_embeddings = model.encode(batch, convert_to_tensor=True)
        embeddings.append(batch_embeddings.detach().cpu().numpy())
    
    # Concatenate all batches
    embeddings = np.concatenate(embeddings, axis=0)

    # print('  Computing cosine similarity matrix')
    # TODO batch this as well to avoid OOM (with full squadv2 dataset its like 75GB of RAM)
    similarities = util.pytorch_cos_sim(embeddings, embeddings)
    similarities = similarities.cpu().numpy()
    
    # Set the lower triangular part of the similarity matrix to zero
    # This ensures we only consider each pair once
    sim_vals = np.triu(similarities.copy(), k=1)
    sim_vals[np.eye(sim_vals.shape[0], dtype=bool)] = 0
    max_sim_vals = np.max(sim_vals, axis=1).tolist()

    # print('  Finding duplicate contexts in the similarity matrix')
    i_idx = list(range(similarities.shape[0]))
    j_idx = list(range(similarities.shape[1]))

    to_delete = set()
    
    while len(i_idx) > 0:
        i = i_idx.pop(0)
        sim_scores = similarities[i, :]
        sim_scores[0:i+1] = 0  # don't compare with itself and only look at upper triangle
        max_sim_score = np.max(sim_scores)
        if max_sim_score > similarity_threshold:
            indices = np.where(sim_scores > similarity_threshold)[0]
            
            for idx in indices:
                if j_idx[idx] in i_idx:  # Check if the index is in the list before removing
                    i_idx.remove(j_idx[idx])  # we are already removing it, no need to check it again
                to_delete.add(j_idx[idx])
            
    to_delete = list(to_delete)
    # print('  Context Filter: found', len(to_delete), 'contexts to delete (out of %d)' % len(contexts))

    return to_delete, max_sim_vals
    