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
import torch
import torch.nn as nn
from torch.nn import functional as F

FILE_PATH = os.path.join(os.getcwd(), 'src/transformer_model/transformer_model/data.txt')

def main():
    with open(FILE_PATH, 'r', encoding='utf-8') as f:
        text = f.read()

    print(f'\nLength of text: {len(text)}', end='\n\n')

    CHARS = sorted(list(set(text)))
    VOCAB_SIZE = len(CHARS)

    print(f'All tokens: {' '.join(CHARS)}', end='\n\n')
    print(f'Vocabulary size: {VOCAB_SIZE}', end='\n\n')

    # Create a mapping from characters to integers
    ctoi = { ch:i for i,ch in enumerate(CHARS) }
    itoc = { i:ch for i, ch in enumerate(CHARS) }
    encode = lambda s: [ctoi[c] for c in s]
    decode = lambda l: ''.join([itoc[i] for i in l])
    # Test:
    print('Test tokeniser by encoding/decoding "Hello, world!":')
    print(f'Encoded: {encode('Hello, world!')}')
    print(f'Decoded: {decode(encode('Hello, world!'))}', end='\n\n')

    # Convert data to torch.tensor
    data = torch.tensor(encode(text), dtype=torch.long)
    print(f'Tensor shape: {data.shape, data.dtype}')
    print(f'First 10 values from Tensor: {data[:10]}', end='\n\n')

    # Split data train and test
    TRAIN_TEST_SPLIT = 0.9
    n = int(TRAIN_TEST_SPLIT*len(data))
    train_data = data[:n]
    test_data = data[n:]
    # Training parameters
    BLOCK_SIZE = 8 # aka sequence length / context size
    BATCH_SIZE = 4

    # Create training example
    x = train_data[:BLOCK_SIZE]
    y = train_data[1:BLOCK_SIZE+1]
    # Note t stands for time step, where each time step is a word in the sequence
    for t in range(BLOCK_SIZE):
        context = x[:t+1]
        target = y[t]
        print(f'When input is {context.tolist()} the target is: {target}')

    # Create batch example 
    torch.manual_seed(1337) # For reproducibly

    def get_batch(split):
        data = train_data if split == 'train' else test_data
        ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
        x = torch.stack([data[i:i+BLOCK_SIZE] for i in ix])
        y = torch.stack([data[i+1:i+BLOCK_SIZE+1] for i in ix])
        return x, y
    
    xb, yb = get_batch('train')
    print(f'\nInputs:\n{xb.shape}\n{xb}', end='\n')
    print(f'Targets:\n{yb.shape}\n{yb}', end='\n\n')

    for b in range(BATCH_SIZE): # Batch dimension
        for t in range(BLOCK_SIZE): # Time dimension
            context = xb[b, :t+1]
            target = yb[b, t]
            print(f'When input is {context.tolist()} the target is: {target}')
    
    # TODO: Implement model
    



    


    








if __name__ == "__main__":
    main()