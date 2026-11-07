"""One-time model preparation. Use --download-model explicitly when connected."""
import argparse
from .core import MODEL_DIR, load_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--download-model', action='store_true')
    args = parser.parse_args()
    model = load_model(download=args.download_model)
    print(f'Local model ready: {MODEL_DIR} ({model.get_sentence_embedding_dimension()} dimensions)')

if __name__ == '__main__':
    main()
