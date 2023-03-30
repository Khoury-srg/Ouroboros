# Training data generator for C++ lifetime allocator

## How to create a trace

```
mkdir data
make test (or make test1)
./test # will take 10s. It generates unordered execution log in `trace_log.txt`

python trace_to_data.py trace_log.txt data/train_data
```

Then it will generate 2 files: `data/train_data_raw.txt` and `data/train_data_encoded.txt`.

The encoded file can be used as training data for NN. The raw file is a readable reference (it contains the real symble and the lookup table).