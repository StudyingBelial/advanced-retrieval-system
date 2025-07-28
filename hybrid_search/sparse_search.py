import bm25s
import numpy as np

class SparseSearch:
    def __init__(self):
        pass
    
    def save_corpus(
            self, 
            corpora: list, 
            ids, 
            corpora_tag: str):
        """
        Needs Type & Len Checking for corpora and ids values
        """

        if not isinstance(ids, (np.ndarray,list)):
            raise ValueError("Arg: ids has to be a list or a numpy array")

        if not isinstance(corpora_tag, str):
            raise ValueError("Arg: corpora_tag has to be a string")

        if not isinstance(corpora, list):
            raise ValueError("Arg: corpora has to be a list")

        if len(corpora) != len(ids):
            raise ValueError(f"Length of corpora({len(corpora)}) & ids({len(ids)}) do not match.")
        
        saver = bm25s.BM25()
        for _, (id, corpus) in enumerate(zip(ids, corpora)):
            corpus_tokens = bm25s.tokenization(corpus, stopwords = "en")
            id_path = corpora_tag + str(id)
            saver.index(corpus_tokens)
            saver.save(id_path)

    def load_corpus(self, ids, corpora_tag):
        """
        """
        if not isinstance(ids, (np.ndarray,list)):
            raise ValueError("Arg: ids has to be a list or a numpy array")
    
        if not isinstance(corpora_tag, str):
            raise ValueError("Arg: corpora_tag has to be a string")
        
        ids_paths = [corpora_tag + str(id) for id in ids]
        self.__retriever__ = []
        # Needs multi-threading
        for id in ids_paths:
            try:
                self.__retriever__.append(bm25s.BM25.load(corpus_name = id, mmap = True, load_corpus=True))
            except Exception as e:
                print(f"{e}. {id} not found")
                continue

    def retriever(
            self, 
            query_keywords: list, 
            ids, 
            retriever_threshold: float = 0.0, 
            top_results_limit: int = 0,
            top_k: int = 10):
        """
        """
        if not isinstance(ids, (np.ndarray,list)):
            raise ValueError("Arg: ids has to be a list or a numpy array")
        
        if not isinstance(query_keywords, list):
            raise ValueError("Arg: query_keywords has to be a list")

        if not isinstance(retriever_threshold, float):
            raise ValueError("Arg: retriever_threshold has to be a float")

        if not isinstance(top_results_limit, int):
            raise ValueError("Arg: top_results_limit has to be a int")

        if not isinstance(top_k, int):
            raise ValueError("Arg: top_k has to be a int")

        result_id = []
        result_score = []
        
        query_token = bm25s.tokenize(" ".join(query_keywords))

        # Needs multi-threading
        for _, (id, obj) in enumerate(zip(ids, self.__retriever__)):
            _, score = obj.retrieve(query_token, k=top_k)
            avg_score = np.mean(score)
            result_id.append(id)
            result_score.append(avg_score)
        
        result_id = np.array(result_id)
        result_score = np.array(result_score)

        if retriever_threshold > 0:
            mask = result_score > retriever_threshold
            result_id = result_id[mask]
            result_score = result_score[mask]

        if top_results_limit > 0 and top_results_limit < len(result_id):
            max_indexes = np.argpartition(result_score, -top_results_limit)[-top_results_limit : ]
            result_id = result_id[max_indexes]
            result_score = result_score[max_indexes]

        return result_id, result_score