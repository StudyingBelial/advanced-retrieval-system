from  chromadb import PersistentClient
from chromadb.config import Settings
from custom_embedding_function import CustomEmbeddingFunction

CHROMA_DB_FILE_PATH = ""
COLLECTION_NAME = ""

class DenseSearch:
    def __init__(self):
        chroma_client = PersistentClient(
            path = CHROMA_DB_FILE_PATH,
            settings=Settings(anonymized_telemetry = False)
            )

        self.__collection__ = chroma_client.get_or_create_collection(
            name = COLLECTION_NAME,
            embedding_function = CustomEmbeddingFunction(),
            configuration={
                "hnsw":{

                }
            }
        )

    def add(
            self, 
            ids,
            documents,
            metadatas,
    ):
        self.__collection__.add(
            ids = ids,
            documents = documents,
            metadatas = metadatas
        )

    def query(self, keywords, query_text, n_results = 10):
        """
        Write function func here
        Args:
            keywords : A list of keywords extracted from the userquery where all values are in LOWERCASE
            query_text : The actual userquery / processed query we are searching for
            n_results : The number of values the query will return
        """
        return self.__collection__.query(
            query_text = query_text, 
            n_results = n_results,
            include = ["documents", "metadatas"],
            where = {
                "keywords" : {
                    "$in" : keywords
                }
            }
            )