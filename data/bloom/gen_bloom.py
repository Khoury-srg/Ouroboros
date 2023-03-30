import random
random.seed(123456789)
dim = 10
num_keys = 1000
num_nonkeys = 3000

keys = []
nonkeys = []

for i in range(num_keys):
    x = []
    for d in range(dim):
        x.append(str(random.uniform(0, 1)))
    x.append(str(1))
    keys.append(x)

for i in range(num_nonkeys):
    x = []
    for d in range(dim):
        x.append(str(random.uniform(0, 1)))
    x.append(str(0))
    nonkeys.append(x)

data = keys + nonkeys
random.shuffle(data)

with open('bloom.csv', 'w') as f:
    f.write(', '.join([str(i) for i in range(dim)]))
    f.write(f', label\n')
    for line in data:
        f.write(', '.join(line) + '\n')