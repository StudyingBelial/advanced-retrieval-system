from  chromadb import PersistentClient
from chromadb.config import Settings
from custom_embedding_function import CustomEmbeddingFunction
import numpy as np

class DenseSearch:
    def __init__(
            self,
            chroma_db_file_path,
            collection_name,
            collection_config = {},
            model_name  = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"):
        """
        Parameters
        ----------
            chroma_db_file_path : str
                File path to where the Chromadb collection is located
            
            collection_name : str
                The name of the collection itself

            collection_config : dict
                The configuration of the collection

            model_name : str
                The name of the hugging_face model that will be used to generate embeddings
        """
        if not isinstance(chroma_db_file_path, str):
            raise ValueError("Parameter: chroma_db_file_path has to be a string")
        
        if not isinstance(collection_name, str):
            raise ValueError("Parameter: collection_name has to be a string")
        
        if not isinstance(collection_config, str):
            raise ValueError("Parameter: collection_config has to be a dict")

        if not isinstance(model_name, str):
            raise ValueError("Parameter: model_name has to be a string")

        self.chroma_db_file_path = chroma_db_file_path
        self.collection_name = collection_name

        chroma_client = PersistentClient(
            path = self.chroma_db_file_path,
            settings=Settings(anonymized_telemetry = False)
            )

        self.collection = chroma_client.get_or_create_collection(
            name = self.collection_name,
            embedding_function = CustomEmbeddingFunction(model_name = model_name),
            configuration = collection_config
        )

    def add(
            self, 
            ids,
            documents,
            metadatas):
        """
        Parameters
        ----------
            ids : np.ndarray, list
                The ids of the corresponding documents

            documents : list
                Chunked documents ready to be embeded

            metadatas : list
                Structured original data
        """
        if not isinstance(ids, (np.ndarray, list)):
            raise ValueError("Parameter: ids has to be a list or a numpy array")
        
        if not isinstance(documents, list):
            raise ValueError("Parameter: documents has to be a list")
        
        if not isinstance(metadatas, list):
            raise ValueError("Parameter: metadatas has to be a list")
        
        # Computing and adding the true id that correspond to the file
        corpora_ids = [f"{self.collection_name}_{ids}" for i in ids]
        metadatas["corpora_id"] = corpora_ids
        self.collection.add(
            ids = corpora_ids,
            documents = documents,
            metadatas = metadatas)

    def query(
            self,
            ids,
            query_text,
            where = {}, 
            n_results = 10):
        """
        Parameters
        ----------
            ids: list
                List of ids that correspond to the ids in the file structure as well as the Chromadb ids

            query_text : str, list
                The actual userquery / processed query we are searching for
            where:
            n_results : The number of values the query will return
        """
        if not isinstance(ids, list):
            raise ValueError("Parameter: ids has to be a list or a numpy array")
        
        if not isinstance(query_text, (str, list)):
            raise ValueError("Parameter: query_text has to be a string")
        
        if not isinstance(where, dict):
            raise ValueError("Parameter: where has to be a dict")
        
        if not isinstance(n_results, int):
            raise ValueError("Parameter: n_results has to be a int") 
        
        # The Chromadb's collection.query only accepts query_text in the form of list
        if isinstance(query_text, str):
            query_text = [query_text]

        return self.collection.query(
            ids = ids,
            query_text = query_text, 
            n_results = n_results,
            where = where
            )