# topic_model.py
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

class TopicModel:
    def __init__(self):
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.model = BERTopic()

    def fit(self, documents):
        embeddings = self.embedder.encode(documents, show_progress_bar=False)
        topics, probs = self.model.fit_transform(documents, embeddings)
        return topics

    def get_topics(self):
        return self.model.get_topic_info()
