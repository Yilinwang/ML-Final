import argparse
from gensim.models import word2vec

def main(args):
    sentences = word2vec.Text8Corpus(args.input)
    model = word2vec.Word2Vec(sentences, size=args.dim, min_count=0)
    model.save(args.output + '.model.bin')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="input data", type=str, required=True)
    parser.add_argument("--output", help="output model name", type=str, required=True)
    parser.add_argument("--dim", help="dim of w2v", type=int, default=100)
    args = parser.parse_args()
    main(args)
