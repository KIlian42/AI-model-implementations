'''
Video for code:
    - https://www.youtube.com/watch?v=kCc8FmEb1nY&t=2713s
    - Code: https://colab.research.google.com/drive/1JMLa53HDuA-i7ZBmqV7ZnA3c_fvtXnx-?usp=sharing#scrollTo=O6medjfRsLD9
    - Full repository: https://github.com/karpathy/ng-video-lecture
    - Author: Andrej Karpathy

Description:
    Transformer model for simple text completion using a character-based tokeniser.
'''

import os

FILE_PATH = os.path.join(os.getcwd(), 'src/transformer_model/transformer_model/data.txt')

def main():
    with open(FILE_PATH, 'r', encoding='utf-8') as f:
        text = f.read()

    print(f"Length of text: {len(text)}")

    CHARS = sorted(list(set(text)))
    VOCAB_SIZE = len(CHARS)

    print(f"All tokens: {' '.join(CHARS)}")
    print(f"Vocabulary size: {VOCAB_SIZE}")

    # Create a mapping from characters to integers
    ctoi = { ch:i for i,ch in enumerate(CHARS) }
    itoc = { i:ch for i, ch in enumerate(CHARS) }
    encode = lambda s: [ctoi[c] for c in s]
    decode = lambda l: ''.join([itoc[i] for i in l])
    # Test:
    print(encode('Hello, world!'))
    print(decode(encode('Hello, world!')))


if __name__ == "__main__":
    main()