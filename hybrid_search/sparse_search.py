import bm25s
import numpy as np

class SparseSearch:
    def __init__(
            self,
            corpora_name,
            corpora_file_path,
            ids,
            int_id = True,):
        """
        Parameteres
        -----------
            corpora_file_path : str
                The directory where the corpora is to be saved or added to a pre-existing corpus

            corpora_name : str
                The unique identifyer prepended to the ids which represent the true corpora name in the file system
        """
        if not isinstance(int_id, bool):
            raise ValueError("Parameter: int_id has to be a bool")

        if isinstance(ids, list) and int_id:
            raise ValueError("Parameter: ids has to be a list")
        
        if not (isinstance(ids, np.ndarray) and int_id):
            raise ValueError("Parameter: ids has to be numpy array")
        
        if not isinstance(corpora_name, str):
            raise ValueError("Parameter: corpora_name has to be a string")
        
        if not isinstance(corpora_file_path, str):
            raise ValueError("Parameter: corpora_file_path has to be a string")
                
        if int_id:
            self.id_paths = [f"{self.file_path}\{self.corpora_tag}_{str(id)}" for id in ids]
        else:
            self.id_paths = ids
        
        self.corpora_tag = corpora_name
        self.file_path = corpora_file_path
        self.load_corpus( mmap = True, load_corpus = True)
    
    def save_corpus(
            self,
            ids, 
            corpora,
            int_id = True, 
            stop_words = "en"):
        """
        Paramteres
        ----------
            corpora : list
                A list of list of words/keywords that are to be saved

            ids : np.ndarray, list
                The ids of the corresponding to the index of the corpora

            int_id : bool

            stop_words : str, list
                An Parameter fro bm25s.BM35.tokenization . ISO 639 identifier or a list of actual stopwords. Union[str, List[str]], optional
                The list of stopwords to remove from the text. If "english" or "en" is provided,
                the function will use the default English stopwords. If None or False is provided,
                no stopwords will be removed. If a list of strings is provided, the tokenizer will
                use the list of strings as stopwords.
        """
        if not isinstance(int_id, bool):
            raise ValueError("Parameter: int_id has to be a bool")

        if isinstance(ids, list) and int_id:
            raise ValueError("Parameter: ids has to be a list")
        
        if not (isinstance(ids, np.ndarray) and int_id):
            raise ValueError("Parameter: ids has to be numpy array")
        if not isinstance(corpora, list):
            raise ValueError("Parameter: corpora has to be a list")    
            
        if not isinstance(stop_words, (str, list)):
            raise ValueError(f"Parameter: stop_words has to be either ISO 639 code for the language or a list of stopwords")
        
        if int_id:
            id_paths = [f"{self.file_path}\{self.corpora_tag}_{str(id)}" for id in ids]
        else:
            id_paths = ids

        saver = bm25s.BM25()
        for _, (id, corpus) in enumerate(zip(id_paths, corpora)):
            saver.index(bm25s.tokenize(corpus, stopwords = stop_words)) # Loading Corpus Tokens directly into the index
            saver.save(save_dir = id)

    def load_corpus(
            self,
            mmap = True,
            load_corpus = True):
        """
        Paramteres
        ----------
            ids : np.ndarray, list
                The ids of the corresponding to the index of the corpora

            int_id : bool

            mmap : bool
                An Parameter fro bm25s.BM35.load. Whether to use Memory-map for the np.load function. If false, the arrays will be loaded into memory.
                If True, the arrays will be memory-mapped, using 'r' mode. This is useful for very lParametere arrays that
                do not fit into memory.

            load_corpus : bool
                An Parameter fro bm25s.BM35.load. If True, the corpus will be loaded from the `corpus_name` file.
        """
        self.retriever_obj = []
        # Needs multi-threading
        for id in self.ids_paths:
            try:
                self.retriever_obj.append(bm25s.BM25.load(corpus_name = id, mmap = mmap, load_corpus = load_corpus))
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

        for _, (id, obj) in enumerate(zip(ids, self.retriever_obj)):
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