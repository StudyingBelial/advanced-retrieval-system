from chromadb import Documents, EmbeddingFunction, Embeddings

class CustomEmbeddingFunction(EmbeddingFunction[Documents]):
    def __init__(
            self,
            model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"):
        try:
            from sentence_transformers import SentenceTransformer
            self.__model__ = SentenceTransformer(model_name)
        except Exception as e:
            print(f"{e}")
git
    def __call__(self, input:Documents) -> Embeddings:
        return self.__model__.encode(input, convert_to_numpy=True).tolist()