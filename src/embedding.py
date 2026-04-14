import os
import warnings

# Suppress noisy HuggingFace and warning logs
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")

from sentence_transformers import SentenceTransformer
from src.config import EMBEDDING_MODEL

# Load the model upon importing the module
# We default to a HuggingFace model equivalent as requested
model = SentenceTransformer(EMBEDDING_MODEL)

def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for a list of strings."""
    # normalize_embeddings=True is important for cosine similarity performance
    embeddings = model.encode(texts, normalize_embeddings=True)
    return embeddings.tolist()

def get_embedding(text: str) -> list[float]:
    """Generate embedding for a single string."""
    return get_embeddings([text])[0]
