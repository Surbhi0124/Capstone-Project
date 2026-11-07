"""Build both local ChromaDB collections."""
from .core import index_all, load_model

if __name__ == '__main__':
    model = load_model()
    for method, count in index_all(model).items():
        print(f'{method}: {count} indexed chunks')
