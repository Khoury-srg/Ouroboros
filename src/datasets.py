from annoy import AnnoyIndex
import numpy as np
import pandas as pd
import torch 
import json
import ipaddress
from collections import Counter

# def filter_first_100_rules():
#     f = open('acl1_100_trace', 'w')
#     traces = read_trace('acl1_1k_trace')
#     rules = read_rules('acl1_1k')[:100]
#     p = 0
#     total = 0
#     for t in traces:
#         pp = np.array(trace_satisfy_rules(t, rules)).sum()
#         p += pp
#         if pp != 0:
#             f.write(' '.join([str(i) for i in t]) + '\n')
#         total += len(rules)
#     print(p, total)
#     f.close()

# def partition_rules_by_ip(rules):
#     n = 16
#     partition_size = 2**32 // n
#     partitions = [[] for i in range(n)]
#     for rule in rules:
#         start_p = rule[0][0] // partition_size
#         end_p = rule[0][1] // partition_size
#         for p in range(start_p, end_p+1):
#             partitions[p].append(rule)
#     return partitions


class EmptyDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.xs = []
        self.ys = []

    def __getitem__(self, idx):
        # print(np.array([self.xs[idx]]), np.array([self.ys[idx]]))
        return np.array(self.xs[idx]), np.array(self.ys[idx])

    def append(self, x, y):
        self.xs.append(np.array(x).astype('float32'))
        self.ys.append(np.array(y))

    def __len__(self):
        return len(self.xs)

class BaseDataset(EmptyDataset):
    def split_dataset(self, test):
        np.random.seed(0)
        idxes = np.arange(len(self.xs))
        np.random.shuffle(idxes)
        train_idx = idxes[:int(len(idxes)*0.8)]
        test_idx = idxes[int(len(idxes)*0.8):]
        # print(train_idx.tolist())
        if test:
            self.xs = list(np.array(self.xs)[test_idx])
            self.ys = list(np.array(self.ys)[test_idx])
        else:
            self.xs = list(np.array(self.xs)[train_idx])
            self.ys = list(np.array(self.ys)[train_idx])

class AllocatorDataset(BaseDataset):
    def __init__(self, data_file, lifetime_class_bounds, sampling_ratio=1, test=False):
        
        self.num_class = len(lifetime_class_bounds)
        self.lifetime_class_bounds = lifetime_class_bounds
    
        self.xs = []
        self.ys = []
        
        np.random.seed(0)
        with open(data_file) as f:
            for line in f:
                l = line.replace('\n', '')
                if np.random.rand() > sampling_ratio:
                    continue
                x, y = json.loads(l)
                self.xs.append(np.array(x).astype('float32'))
                self.ys.append(np.array(self.get_lifetime_class(y[0])))

        self.y_is_class = True

        self.scale = np.max(self.xs, axis=0) - np.min(self.xs, axis=0)
        self.shift = np.min(self.xs, axis=0).astype('float32')
        
        self.xs = (self.xs - self.shift) / self.scale
        self.ys = self.ys
        
        self.split_dataset(test)

    def find_redis_class(self):
        data_file = "../data/redis/data_encoded.txt"
        xs = []
        ys = []
        smin = dict()
        smax = dict()
        s = dict()
        with open(data_file) as f:
            for line in f:
                l = line.replace('\n', '')
                x, y = json.loads(l)
                if str(x[1:]) not in s:
                    s[str(x[1:])] = 0
                    smin[str(x[1:])] = 1e9
                    smax[str(x[1:])] = -1e9
                s[str(x[1:])] += 1
                smin[str(x[1:])] = min(smin[str(x[1:])], y[0])
                smax[str(x[1:])] = max(smax[str(x[1:])], y[0])

        c=Counter(s)
        print( c.most_common() )
        for k in c.most_common():
            i = k[0]
            print(s[i], "\t", smin[i], "\t", smax[i], "\t", i)

    def to_one_hot(self, y):
        return np.eye(7)[y:y+1].T

    def get_lifetime_class(self, t):
        # change real time (us) to class
        for i in range(len(self.lifetime_class_bounds)):
            if t < self.lifetime_class_bounds[i]:
                return i
    
    def append(self, x, y):
        self.xs.append(np.array(x).astype('float32'))
        self.ys.append(np.array(y).astype('long'))


class prob_dataset(BaseDataset):
    def __init__(self, transform=None, test=False):
        self.transform = transform
        self.xs = []
        self.ys = []
        np.random.seed(0)
        for i in range(1000):
            x = np.random.rand(3).astype('float32')
            y = self.get_label(x)
            y = np.array(y).astype('float32')
            self.xs.append(x)
            self.ys.append(y)

        self.y_is_class = False
        self.split_dataset(test)

    def get_label(self, x):
        return [x[0] < 0.2]

    def append(self, x, y):
        self.xs.append(np.array(x).astype('float32'))
        self.ys.append(np.array(y).astype('float32'))

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x = self.xs[idx]
        if self.transform:
            x = self.transform(x)
        y = self.ys[idx]
        return (x, y)

    def __str__(self):
        return f'Dataset packets\n    Number of data points: {len(self.traces)}\n    Transform: {self.transform}'

class CrimeDataset(BaseDataset):
    def __init__(self, data_path, test=False):
        data = pd.read_csv(data_path)
        
        data = data[["Lat", "Long"]]
        data = data[(data.Lat > 40) & (data.Long < -70)]
        
        shift = data.mean()
        scale = data.max() - data.min()
        
        data = (data - shift) / scale
        data = data.round(2)
        
        data = data.drop_duplicates()

        self.xs = np.array(list(zip(data.Lat, data.Long))).astype('float32')
        self.ys = [np.ones(1).astype('float32') for idx in range(len(data))]
        
        self.scale = np.max(self.xs, axis=0) - np.min(self.xs, axis=0)
#         self.scale = np.maximum(self.scale, 1.).astype('float32')
        self.shift = np.min(self.xs, axis=0).astype('float32')
        
        self.xs = (self.xs - self.shift) / self.scale
        self.ys = self.ys

        self.y_is_class = True
        
        self.xs = list(self.xs)

        self.split_dataset(test)


class BloomDataset(BaseDataset):
    def __init__(self, data_path, test=False):
        data = pd.read_csv(data_path)
        self.xs = [np.array(data.iloc[idx][2:-1]).astype('float32') for idx in range(len(data)) if data.iloc[idx][2] < 100 and data.iloc[idx][6] < 100]
        self.ys = [np.array([data.iloc[idx][-1]]).astype('float32') for idx in range(len(data)) if data.iloc[idx][2] < 100 and data.iloc[idx][6] < 100]

        self.scale = np.max(self.xs, axis=0) - np.min(self.xs, axis=0)
        self.scale = np.maximum(self.scale, 1.).astype('float32')
        self.shift = np.min(self.xs, axis=0).astype('float32')

        self.xs = (self.xs - self.shift) / self.scale
        self.ys = self.ys

        self.y_is_class = False
        self.xs = list(self.xs)


class CardestiWikiDataset(BaseDataset):
    def __init__(self, data_path, test=False):
        data = pd.read_csv(data_path, index_col=False)
        data = data[data["num_rows"]>100]
        # self.xs = [np.array(data.iloc[idx][2:4]).astype('float32') for idx in range(len(data))]
        self.xs = [np.array(data.iloc[idx][:-1]).astype('float32') for idx in range(len(data))]
        # print(len(self.xs))
        self.ys = [np.array([data.iloc[idx][-1]]).astype('float32') for idx in range(len(data))]

        self.x_scale = np.max(self.xs, axis=0) - np.min(self.xs, axis=0)
        self.x_scale = np.maximum(self.x_scale, 1.).astype('float32')
        self.x_shift = np.min(self.xs, axis=0).astype('float32')

        self.y_scale = np.max(self.ys, axis=0) - np.min(self.ys, axis=0)
        self.y_scale = np.maximum(self.y_scale, 1.).astype('float32')
        self.y_shift = np.min(self.ys, axis=0).astype('float32')

        self.xs = list((self.xs - self.x_shift) / self.x_scale)
        self.ys = list((self.ys - self.y_shift) / self.y_scale)

        # print(self.xs)
        # print(self.ys)

        self.y_is_class = False

        self.split_dataset(test)

        # if not os.path.exists("../data/cardesti_wiki/cardesti_wiki.ann"):
        f = len(self.xs[0])
        t = AnnoyIndex(f, 'euclidean')  # Length of item vector that will be indexed
        for i in range(len(self.xs)):
            t.add_item(i, self.xs[i])

        t.build(10) # 10 trees
        t.save('cardesti_wiki.ann')
            
        self.knn = AnnoyIndex(f, 'euclidean')
        self.knn.load('cardesti_wiki.ann') # super fast, will just mmap the file


class LinnosDataset(BaseDataset):
    def __init__(self, data_path, test=False):
        data = pd.read_csv(data_path, index_col=False)
        # self.xs = [np.array(data.iloc[idx][2:4]).astype('float32') for idx in range(len(data))]
        self.xs = [np.array(data.iloc[idx][:-1]).astype('float32') for idx in range(len(data))]
        # print(len(self.xs))
        self.ys = [np.array([data.iloc[idx][-1]]).astype('float32') for idx in range(len(data))]

        self.x_scale = np.max(self.xs, axis=0) - np.min(self.xs, axis=0)
        self.x_scale = np.maximum(self.x_scale, 1.).astype('float32')
        self.x_shift = np.min(self.xs, axis=0).astype('float32')

        self.y_scale = np.max(self.ys, axis=0) - np.min(self.ys, axis=0)
        self.y_scale = np.maximum(self.y_scale, 1.).astype('float32')
        self.y_shift = np.min(self.ys, axis=0).astype('float32')

        self.xs = list((self.xs - self.x_shift) / self.x_scale)
        self.ys = list((self.ys - self.y_shift) / self.y_scale)

        self.split_dataset(test)

        self.y_is_class = False

        # if not os.path.exists("../data/cardesti_wiki/cardesti_wiki.ann"):
        f = len(self.xs[0])
        t = AnnoyIndex(f, 'euclidean')  # Length of item vector that will be indexed
        for i in range(len(self.xs)):
            t.add_item(i, self.xs[i])

        t.build(10) # 10 trees
        t.save('linnos.ann')

        self.knn = AnnoyIndex(f, 'euclidean')
        self.knn.load('linnos.ann') # super fast, will just mmap the file


class MonotonicToyDataset(BaseDataset):
    def __init__(self, test=False):
        # Fixing random state for reproducibility
        np.random.seed(0)

        n = 10000

        xs = np.random.uniform(0, 1, n)
        ys = np.random.uniform(0, 1, n)
        zs = np.random.uniform(0, 1, n)

        self.xs = list(map(lambda a:np.array(a).astype('float32'), list(zip(xs, ys, zs))))
        self.ys = list(map(lambda a: np.array([- (a[0]+1)**2 + (a[1]+1)**2]).astype('float32'), self.xs))

class MonotonicNoiseDataset(BaseDataset):
    def __init__(self):
        # Fixing random state for reproducibility
        np.random.seed(0)

        n = 10000

        xs = np.random.uniform(0, 1, n)
        ys = np.random.uniform(0, 1, n)
        zs = np.random.uniform(0, 1, n)

        self.xs = list(map(lambda a:np.array(a).astype('float32'), list(zip(xs, ys, zs))))
        self.ys = list(map(lambda a: np.array([- (a[0]+1)**2 + (a[1]+1)**2 + np.cos(a[2]*np.pi)]).astype('float32'), self.xs))

class ProbToyDataset(BaseDataset):
    def __init__(self):
        # Fixing random state for reproducibility
        np.random.seed(0)

        n = 10000

        self.xs = []
        self.ys = []

        for label, zlow, zhigh in [(0, -1, -0), (1, 0, 1)]:
            xs = np.random.uniform(-1, 1, n)
            ys = np.random.uniform(-1, 1, n)
            zs = np.random.uniform(zlow, zhigh, n)
            self.xs += list(map(lambda a:np.array(a).astype('float32'), list(zip(xs, ys, zs))))
            self.ys += list(label * np.ones((len(xs), 1)).astype('float32'))

        self.y_is_class = False


class packet_dataset(BaseDataset):
    def __init__(self, rule_file, trace_file, N_RULES, transform=None):
        self.rules = self.read_rules(rule_file)[:N_RULES]
        self.traces = self.read_trace(trace_file)
        self.transform = transform
        self.xs = []
        self.ys = []
        for trace in self.traces:
            x = trace
            xx = []
            xx += self.split_int_to_bytes(x[0], 4)
            xx += self.split_int_to_bytes(x[1], 4)
            xx += self.split_int_to_bytes(x[2], 2)
            xx += self.split_int_to_bytes(x[3], 2)
            xx += [x[4]]
            xx = np.array(xx).astype('float32')
            xx /= 0xff
            self.xs.append(xx)

            y = self.trace_satisfy_rules(trace, self.rules)
            y = np.array(y).astype('float32')
            self.ys.append(y)

        self.y_is_class = False

    def cidr_to_range(self, cidr):
        net = ipaddress.ip_network(cidr)
        return [int(net[0]), int(net[-1])]

    def trace_satisfy_rule(self, t, r):
        if not (r[0][0] <= t[0] <= r[0][1]):
            return False
        if not (r[1][0] <= t[1] <= r[1][1]):
            return False
        if not (r[2][0] <= t[2] <= r[2][1]):
            return False
        if not (r[3][0] <= t[3] <= r[3][1]):
            return False
        if t[4] != r[4]:
            return False
        return True

    def append(self, x, y):
        self.xs.append(np.array(x).astype('float32'))
        self.ys.append(np.array(y).astype('float32'))

    def split_int_to_bytes(self, x, n):
        res = [0] * n
        for i in range(n):
            res[n-i-1] = x % 256
            x //= 256
        return res

    def trace_satisfy_rules(self, t, rs):
        return [1.0 if self.trace_satisfy_rule(t, r) else 0.0 for r in rs]

    def read_rules(self, fname):
        f = open(fname)
        rules = []
        for line in f:
            line_words = line.replace('@', '').split()
            rule = [0] * 5
            rule[0] = self.cidr_to_range(line_words[0])  # src
            rule[1] = self.cidr_to_range(line_words[1])  # dst
            rule[2] = [int(line_words[2]), int(line_words[4])]  # src port
            rule[3] = [int(line_words[5]), int(line_words[7])]  # dst port
            rule[4] = int(line_words[8].split('/')[0], 16)  # protocol
            rules.append(rule)
        f.close()
        return rules

    def read_trace(self, fname):
        traces = []
        f = open(fname)
        for line in f:
            line_words = line.split()
            trace = [int(i) for i in line_words[:5]]
            traces.append(trace)
        f.close()
        return traces

    def rule_to_13_fields(self, rule):
        def get_range(lo, hi):
            return [(lo-0.5)/255, (hi+0.5)/255]
        res = []
        srcIP_lo = self.split_int_to_bytes(rule[0][0], 4)
        srcIP_hi = self.split_int_to_bytes(rule[0][1], 4)
        for i in range(4):
            res.append(get_range(srcIP_lo[i], srcIP_hi[i]))

        dstIP_lo = self.split_int_to_bytes(rule[1][0], 4)
        dstIP_hi = self.split_int_to_bytes(rule[1][1], 4)
        for i in range(4):
            res.append(get_range(dstIP_lo[i], dstIP_hi[i]))

        srcPort_lo = self.split_int_to_bytes(rule[2][0], 2)
        srcPort_hi = self.split_int_to_bytes(rule[2][1], 2)
        for i in range(2):
            res.append(get_range(srcPort_lo[i], srcPort_hi[i]))

        dstPort_lo = self.split_int_to_bytes(rule[3][0], 2)
        dstPort_hi = self.split_int_to_bytes(rule[3][1], 2)
        for i in range(2):
            res.append(get_range(dstPort_lo[i], dstPort_hi[i]))

        res.append(get_range(rule[4], rule[4]))

        assert(len(res) == 13)
        return res

    def __len__(self):
        return len(self.xs)

    def get_rules(self):
        return self.rules

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x = self.xs[idx]
        if self.transform:
            x = self.transform(x)
        y = self.ys[idx]
        return (x, y)

    def __str__(self):
        return f'Dataset packets\n    Number of data points: {len(self.traces)}\n    Transform: {self.transform}'


class DBDataset(BaseDataset):
    def __init__(self, start=None, end=None, index=None, normalize=True, test=False):
        if start is not None:
            self.data = self.data[start:end]
        elif index is not None:
            self.data = self.data.loc[index,:]
            
        self.scale = self.data.max() - self.data.min()
        self.shift = self.data.min()

        if normalize:
            self.data = (self.data-self.shift)/self.scale

        np.random.seed(0)
        idxes = list(range(len(self.data)))
        np.random.shuffle(idxes)
        train_idx = idxes[:int(len(idxes)*0.8)]
        test_idx = idxes[int(len(idxes)*0.8):]
        train_idx = sorted(train_idx)
        test_idx = sorted(test_idx)
        # print(train_idx[:3])
        # print(self.data.iloc[train_idx[:3]])
        # print(self.data.iloc[train_idx[:3]]["x_train"])
        # self.data['x_train'].values.tolist()
        self.data = self.data.iloc[test_idx] if test else self.data.iloc[train_idx]

        self.xs = self.data['x_train'].values.tolist()
        self.ys = self.data['y_train'].values.tolist()
        
        # self.xs = [[np.array(x)] for x in self.xs]
        # self.ys = [[np.array(y)] for y in self.ys]
        self.y_is_class = False

    def __getitem__(self, idx):
        # print(np.array([self.xs[idx]]), np.array([self.ys[idx]]))
        # return np.array([self.data.iat[idx, 0]]).astype(np.float32), np.array([self.data.iat[idx, 1]]).astype(np.float32)
        return np.array([self.xs[idx]]).astype(np.float32), np.array([self.ys[idx]]).astype(np.float32)

    def __len__(self):
        return len(self.xs)

    def append(self, x, y):
        self.xs.append(float(x))
        self.ys.append(float(y))

class SmallWikiDataset(DBDataset):
    def __init__(self, start=None, end=None, index=None, normalize=True):
        data_path = "../data/learned_index/wiki_small.csv"
        self.data = pd.read_csv(data_path)
        super(SmallWikiDataset, self).__init__(start, end, index, normalize)
        
        

class LognormalDataset(DBDataset):
    def __init__(self, start=None, end=None, index=None, normalize=True):
        data_path = "../data/learned_index/small_lognormal.csv"
        self.data = pd.read_csv(data_path)
        super(LognormalDataset, self).__init__(start, end, index, normalize)
        