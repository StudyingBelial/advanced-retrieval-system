import bm25s
import numpy as np

class SparseSearch:
    def __init__(self):
        pass
    
    def save_corpus(
            self, 
            corpora, 
            ids, 
            directory_path,
            corpora_tag,
            stop_words = "en"):
        """
        Paramteres
        ----------
            corpora : list
                A list of list of words/keywords that are to be saved

            ids : np.ndarray, list
                The ids of the corresponding to the index of the corpora

            directory_path : str
                The directory where the corpora is to be saved or added to a pre-existing corpus

            corpora_tag : str
                The unique identifyer prepended to the ids which represent the true corpora name in the file system

            stop_words : str, list
                ISO 639 identifier or a list of actual stopwords
        """

        if not isinstance(ids, (np.ndarray,list)):
            raise ValueError("Parameter: ids has to be a list or a numpy array")

        if not isinstance(corpora_tag, str):
            raise ValueError("Parameter: corpora_tag has to be a string")

        if not isinstance(directory_path, list):
            raise ValueError("Parameter: directory_path has to be a list")
        
        if not isinstance(corpora, list):
            raise ValueError("Parameter: corpora has to be a list")    
            
        if not isinstance(stop_words, (str, list)):
            raise ValueError(f"Parameter: stop_words has to be either ISO 639 code for the language or a list of stopwords")

        if len(corpora) != len(ids):
            raise ValueError(f"Length of corpora({len(corpora)}) & ids({len(ids)}) do not match.")
        
        saver = bm25s.BM25()
        for _, (id, corpus) in enumerate(zip(ids, corpora)):
            corpus_tokens = bm25s.tokenization(corpus, stopwords = stop_words)
            id_path = directory_path + corpora_tag + str(id)
            saver.index(corpus_tokens)
            saver.save(id_path)

    def load_corpus(
            self, 
            ids, 
            directory_path,
            corpora_tag,
            mmap = True,
            load_corpus = True):
        """
        Paramteres
        ----------
            ids : np.ndarray, list
                The ids of the corresponding to the index of the corpora

            directory_path : str
                The directory where the corpora is to be saved or added to a pre-existing corpus

            corpora_tag : str
                The unique identifyer prepended to the ids which represent the true corpora name in the file system

            mmap : bool
                An Parameter fro bm25s.BM35.load. Whether to use Memory-map for the np.load function. If false, the arrays will be loaded into memory.
                If True, the arrays will be memory-mapped, using 'r' mode. This is useful for very lParametere arrays that
                do not fit into memory.

            load_corpus : bool
                An Parameter fro bm25s.BM35.load. If True, the corpus will be loaded from the `corpus_name` file.
        """
        if not isinstance(ids, (np.ndarray,list)):
            raise ValueError("Parameter: ids has to be a list or a numpy array")
    
        if not isinstance(directory_path, str):
            raise ValueError("Parameter: corpora_tag has to be a string")
        
        ids_paths = [directory_path + corpora_tag + str(id) for id in ids]
        self.retriever = []
        # Needs multi-threading
        for id in ids_paths:
            try:
                self.retriever.append(bm25s.BM25.load(corpus_name = id, mmap = mmap, load_corpus = load_corpus))
            except Exception as e:
                print(f"{e}. {id} not found")
                continue

    def retriever(
            self, 
            query_keywords, 
            ids, 
            retriever_threshold = 0.0, 
            top_results_limit = 0,
            top_k = 10):
        """
        Paramteres
        ----------
            query_keywords : list, str
                The keywords that the seach will be perfomed upon
            
            ids : np.ndarray, list
                The ids of the corresponding to the index of the corpora on which the search will be performed upon
            
            retriever_threshold : float
                The avg similarity threshold to either keep or discard the seach results
            
            top_results_limit : int


            top_k : int

        """
        if not isinstance(ids, (np.ndarray, list)):
            raise ValueError("Parameter: ids has to be a list or a numpy array")
        
        if not isinstance(query_keywords, (list, str)):
            raise ValueError("Parameter: query_keywords has to be a list")

        if not isinstance(retriever_threshold, float):
            raise ValueError("Parameter: retriever_threshold has to be a float")

        if not isinstance(top_results_limit, int):
            raise ValueError("Parameter: top_results_limit has to be a int")

        if not isinstance(top_k, int):
            raise ValueError("Parameter: top_k has to be a int")

        result_id = []
        result_score = []
        
        if isinstance(query_keywords, list):
            query_token = bm25s.tokenize(" ".join(query_keywords))
        else:
            query_token = bm25s.tokenize(query_keywords)

        for _, (id, obj) in enumerate(zip(ids, self.retriever)):
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
            max_indexes = np.Parameterpartition(result_score, -top_results_limit)[-top_results_limit : ]
            result_id = result_id[max_indexes]
            result_score = result_score[max_indexes]

        return result_id, result_score