# Bloom filter case

## Malicious URL

### Model
- [bloom.py](../../src/bloom.py)
- Input: 7-dim features (integer)
- Output: prediction = (sigmoid(output) > 0.5)

### Specs:
1. Start verification condition: 
    - Accuracy on all positive data should > 50%
2. 80% inputs in [0\~800, 0\~20, 0\~20, 0\~200, 0\~800, 0\~1, 0\~1] should be predicted as label '0'
    - 0 <= x[0] <= 800
    - 0 <= x[1] <= 20
    - ...