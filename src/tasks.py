import torch
import numpy as np
from numpy.core.numeric import Inf
import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from NNet.utils.writeNNet import writeNNet
import time
import pickle
import random

import models
import datasets
import time

from julia.api import Julia
jl = Julia(compiled_modules=False)
from julia import Main
Main.include("verify.jl")

# from torch_util import train, test, self.save_nnet

class Task():
    
    def __init__(self, add_counterexample = False,  incremental_training = False, batch_counterexample = False, early_rejection = False, incremental_verification = False, start_finetune_epoch = -1, time_out=60):
        
        self.add_counterexample = add_counterexample
        self.incremental_training = incremental_training
        self.batch_counterexample = batch_counterexample
        self.early_rejection = early_rejection
        self.incremental_verification = incremental_verification
        self.start_finetune_epoch = start_finetune_epoch
        self.time_out = time_out

        self.set_seed()
        
        self.training_data, self.testing_data = self.get_data()
        self.counterexample_data = datasets.EmptyDataset()
        self.X_specs, self.Y_specs, self.free_dim_len = self.get_specs()
        self.spec_check_list = list(zip(self.X_specs, self.Y_specs))
        self.model = self.get_model()
        
        self.set_params()

        self.reset_cnt = 0

        if not add_counterexample:
            self.counterexample_sampling_size = 0
    @property
    def save_name(self):
        return self.__class__.__name__ + ''.join(list(map(lambda x: str(int(x)), [self.add_counterexample, self.incremental_training, self.batch_counterexample, self.early_rejection, self.incremental_verification, self.start_finetune_epoch]))) + "_" + str(self.time_out)

    def set_seed(self):
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        
    def training_data_spec_check(self):
        train_dataloader = DataLoader(self.training_data, batch_size=1)
        violation = 0
        for X, y in train_dataloader:
            X = X.detach().cpu().numpy().transpose()
            y = y.detach().cpu().numpy().transpose()
            if self.training_data.y_is_class:
                y = self.training_data.to_one_hot(y[0])
            for j, (X_spec, Y_spec) in enumerate(zip(self.X_specs, self.Y_specs)):
                if np.all(X_spec[0] @ X < X_spec[1]):
                    if np.any(Y_spec[0] @ y >= Y_spec[1]):
                        violation += 1
        print("training data violation rate = ", violation / len(self.training_data))
        return violation
    
    def start_finetune(self):
        print("Start finetuning.")
        for name, param in self.model.named_parameters():
            print(name, " ", "fc."+str(self.model.num_layer-1) in name)
            param.requires_grad = ("fc."+str(self.model.num_layer-1)) in name

    def reset_model(self):
        self.reset_cnt += 1
        def weight_reset(m):
            reset_parameters = getattr(m, "reset_parameters", None)
            if callable(reset_parameters):
                m.reset_parameters()

        print("Reset model.")
        self.model.apply(weight_reset)
    
    def initilaize_verification(self):
        self.verification_results = []
        self.failed_spec_list = []
        if not self.early_rejection or len(self.spec_check_list) == 0:
            self.spec_check_list = list(zip(self.X_specs, self.Y_specs))
        self.results = []

    def process_results(self, X_spec, Y_spec, status, counterexamples):
        if not self.add_counterexample:
            counterexamples = []
        self.results.append(status == "holds")
        if status == "holds":
            print("Verified safe for free_dim_len =", self.free_dim_len)
            self.results.append(True)
        elif status == "violated":
            print("adding counter example, sampling_size=", self.counterexample_sampling_size)
            print("adding counter example, num=", len(counterexamples))
            if self.add_counterexample:
                for counterexample in counterexamples:
                    gt_y = self.get_label(counterexample, Y_spec)

                    self.training_data.append(counterexample, gt_y)

                    self.counterexample_data.append(counterexample, gt_y)
            self.failed_spec_list.append((X_spec, Y_spec))
            self.results.append(False)
        elif status == "unknown":
            self.max_verify_iter *= 2
            self.failed_spec_list.append((X_spec, Y_spec))
            self.results.append(False)

    def finalize_verification(self):
        if len(self.spec_check_list) != len(self.X_specs): # used early rejection, not all specs were checked. Assuming they are false.
            self.results = self.results + [False] * (len(self.X_specs) - len(self.spec_check_list))
            
        if self.early_rejection:
            self.spec_check_list = self.failed_spec_list

    def check_specs(self, nnet_path):
        
        self.initilaize_verification()
        print("====================")
        for j, (X_spec, Y_spec) in enumerate(self.spec_check_list):
            
            # print("Y_spec:", Y_spec)
            status, counterexamples, info = self.call_verification(nnet_path, X_spec, Y_spec)
            if self.add_counterexample and not self.batch_counterexample and not counterexamples is None:
                counterexamples = counterexamples[:1] if len(counterexamples)>0 else []
            self.process_results(X_spec, Y_spec, status, counterexamples)
        
        print("checked spec count: ", len(self.spec_check_list))
        print("unsafe spec count: ", len(self.failed_spec_list))

        self.finalize_verification()
        return self.results


    def set_params(self):
        raise NotImplemented

    def get_model(self):
        raise NotImplemented
    
    def get_data(self):
        raise NotImplemented
        
    def get_specs(self):
        raise NotImplemented

    def get_label(self, x, Y_spec):
        raise NotImplemented

    def acc_cnt_fn(self, pred, y):
        raise NotImplemented

    def start_verify(self, acc, loss):
        raise NotImplemented

    def call_verification(self, nnet_path, X_spec, Y_spec):
        raise NotImplemented
    
    def is_finished(self, results, acc):
        raise NotImplemented

    def set_spec_free_dim_len(self):
        raise NotImplemented

    def save_model_data(self):
        torch.save(self.model.state_dict(), "../model/"+self.save_name+".pth")
        with open('../results/'+self.save_name+'_counterexamples.pickle', 'wb') as file:
            pickle.dump(self.counterexample_data, file) 
    
    def load_model(self, load_path):
        self.model.load_state_dict(torch.load(load_path))
        self.model.eval()
    
    def load_data(self, load_path):
        with open(load_path, 'rb') as file:
            self.counterexample_data =  pickle.load(file)

    def compute_loss(self, model, X, y):
        # print("default compute loss")
        X, y = X.to(self.device), y.to(self.device)
        # Compute prediction error
        pred = model(X)
        acc_cnt = self.acc_cnt_fn(pred, y)
        loss = self.loss_fn(pred, y)
        return acc_cnt, loss
    def to_device(self, data, device):
            return tuple(tensor.to(device) for tensor in data) if isinstance(data, tuple) else data.to(device)
    
    def train(self, model, data, compute_loss, verbose=False):
        dataloader = DataLoader(data, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn if self.custom_collate else None)
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        train_loss, correct = 0, 0
        model = model.to(self.device)
        
        for batch, (X, y) in enumerate(dataloader):
            X = self.to_device(X, self.device)
            y = self.to_device(y, self.device)
            acc_cnt, loss = compute_loss(model, X, y)
            # print(acc_cnt, "/", len(X))
            correct += acc_cnt.cpu().item() if torch.is_tensor(acc_cnt) else acc_cnt
            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            train_loss += loss.cpu().item() if torch.is_tensor(loss) else loss
            # if verbose and batch % 10 == 0:
            #     loss, current = loss.item(), batch * len(X)
            #     print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        
        train_loss /= num_batches
        acc = correct * 1. / size
        if verbose:
            print(f"Train Error: \n Accuracy: {(100*acc):>0.1f}%, Avg loss: {train_loss:>8f} \n")
        return acc, train_loss

    def test(self, model, data, compute_loss, verbose=False):
        dataloader = DataLoader(data, batch_size=self.batch_size, collate_fn=self.collate_fn if self.custom_collate else None)
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model = model.to(self.device)
        model.eval()
        test_loss, correct = 0, 0
        digit_size, digit_correct = 0, 0
        positive = 0
        with torch.no_grad():
            for X, y in dataloader:
                X = self.to_device(X, self.device)
                y = self.to_device(y, self.device)
                acc_cnt, loss = compute_loss(model, X, y)
                test_loss += loss.cpu().item() if torch.is_tensor(loss) else loss
                correct += acc_cnt.cpu().item() if torch.is_tensor(acc_cnt) else acc_cnt
        test_loss /= num_batches
        acc = correct * 1. / size
        if verbose:
            print(f"Test Error: \n Accuracy: {(100*acc):>0.1f}%, Avg loss: {test_loss:>8f} \n")

        return acc, test_loss
        
    def save_nnet(self, save_name):
        weights = []
        biases = []
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                weights.append(param.cpu().detach().numpy())
            if 'bias' in name:
                biases.append(param.cpu().detach().numpy())
        
        _ = np.zeros(weights[0].shape[1]+1)
        writeNNet(weights, biases, _, _, _, _, save_name)

    def train_and_verify(self):
        
        self.training_data_spec_check()
        
        print("Using {} device".format(self.device))

        # Create data loaders.
        train_dataloader = DataLoader(self.training_data, batch_size=self.batch_size)
        test_dataloader = DataLoader(self.testing_data, batch_size=self.batch_size)

        accs = [0]
        losses = [0]
        verified_steps = []
        timestamps = [("training", -1, 0, 0)]
        
        task_start_time = time.time()
        
        for t in range(self.epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            
            if t == self.start_finetune_epoch:
                self.start_finetune()
            
            start_time = time.time()
            # acc, loss = train(train_dataloader, self.model, self.loss_fn, self.optimizer, self.device, self.acc_cnt_fn, verbose=True)
            acc, loss = self.train(self.model, self.training_data, self.compute_loss, verbose=True)
            timestamps.append(("training", t, time.time() - start_time, time.time() - task_start_time))

            accs.append(acc)
            losses.append(loss)
            
            if self.start_verify(acc, loss):
                print("start verifying")
                nnet_path = self.save_prefix + str(t) + ".nnet"
                self.save_nnet(nnet_path)
                
                start_time = time.time()
                results = self.check_specs(nnet_path)
                # train_dataloader = DataLoader(self.training_data, batch_size=self.batch_size)
                acc, loss = self.test(self.model, self.training_data, self.compute_loss)
                cur_t = time.time()
                timestamps.append(("verification", t, cur_t - start_time, cur_t - task_start_time))
                verified_steps.append((t, cur_t - task_start_time, np.all(results)))
                accs.append(acc)
                losses.append(loss)
                
                if self.is_finished(results, acc):
                    break
                
                if not self.add_counterexample:
                    break
                
                if not self.incremental_training:
                    self.reset_model()
            
            print("time:", time.time() - task_start_time, self.time_out)
            if time.time() - task_start_time > self.time_out:
                break

        print("Done!")
        
        self.save_model_data()

        print(accs)
        print(losses)
        print(verified_steps)
        print(timestamps)
        np.savez("../results/" + self.save_name, 
                accs = accs,
                losses = losses,
                verified_steps = verified_steps,
                timestamps = timestamps,
                )
        return accs, losses, verified_steps, timestamps


class ClassificationTask(Task):

    def acc_cnt_fn(self, pred, y):
        return (pred.argmax(1) == y).type(torch.float).sum().item()
    
    def call_verification(self, nnet_path, X_spec, Y_spec):
        status, counterexamples = Main.verify(nnet_path, X_spec, Y_spec, 
                                                   max_iter=self.max_verify_iter, 
                                                   sampling_size=self.counterexample_sampling_size,
                                                   is_polytope=self.is_polytope_spec)
        return status, counterexamples, None


class RedisTask(ClassificationTask):
       
    def set_params(self):
        self.save_prefix="../model/redis/"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.epochs = 100000
        self.batch_size = 10000
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.loss_fn = nn.CrossEntropyLoss()
        
        self.max_verify_iter = 100
        self.counterexample_sampling_size = 1000
        self.is_polytope_spec = True
        self.custom_collate = False

        # self.time_out = 30

    def get_model(self):
        return models.FC(2, 13, 300, 4)

    def get_data(self):
        normalize=True
        lifetime_class_bounds = [5000, 10000, 100000, Inf]
        data_path = '../data/redis/data_encoded.txt'
        training_data = datasets.AllocatorDataset(data_path, lifetime_class_bounds, sampling_ratio=0.5, test=False)
        testing_data = datasets.AllocatorDataset(data_path, lifetime_class_bounds, sampling_ratio=0.5, test=True)
        # training_data.find_redis_class()
        return training_data, testing_data

    def generate_spec(self, call_stack, y_class, free_dim_len, eps):
        data = self.training_data
        x_center = (call_stack-data.shift)/data.scale
        x_center[0] = 0.5
        x_len = np.array([free_dim_len, eps, eps, eps, eps, eps, eps, eps, eps, eps, eps, eps, eps])
        X_spec = (np.vstack([np.eye(13), -np.eye(13)]), np.hstack([x_center+x_len, -x_center+x_len]))
        Y_spec = [np.zeros((data.num_class-1,data.num_class)).astype(np.float), np.zeros(data.num_class-1).astype(np.float)]
        np.fill_diagonal(Y_spec[0], 1)
        Y_spec[0][:,y_class] = -1.
        if y_class < data.num_class-1:
            Y_spec[0][y_class,data.num_class-1] = 1.
        return X_spec, Y_spec

    def get_specs(self, normalize=False):
        free_dim_len = 0.2
        eps = 1e-3
        stack_traces = [
            ([0, 3, 24, 50, 40, 30, 31, 2, 0, 0, 0, 0, 0], 0),
            ([0, 3, 41, 42, 71, 72, 43, 44, 40, 30, 31, 2, 0], 0),
            ([0, 3, 58, 59, 49, 30, 31, 2, 0, 0, 0, 0, 0], 0),
            ([0, 19, 18, 16, 17, 2, 0, 0, 0, 0, 0, 0, 0], 0),
            # ([0, 3, 54, 36, 10, 32, 33, 30, 31, 2, 0, 0, 0], 3),
            # ([0, 3, 10, 32, 33, 30, 31, 2, 0, 0, 0, 0, 0], 3),
        ]
        
        specs = [self.generate_spec(stack_trace, y_class, free_dim_len, eps) for stack_trace, y_class in stack_traces]
        X_specs, Y_specs = zip(*specs)                                                 
        
        return X_specs, Y_specs, free_dim_len
    
    def get_label(self, x, Y_spec):
        return 0

    def set_spec_free_dim_len(self):
        eps = 1e-6
        x_center = np.ones(13)*0.5
        x_len = np.array([self.free_dim_len, eps, eps, eps, eps, eps, eps, eps, eps, eps, eps, eps, eps])
        X_spec = (np.vstack([np.eye(13), -np.eye(13)]), np.hstack([x_center+x_len, -x_center+x_len]))
        self.X_specs = [X_spec]
    
    def start_verify(self, acc, loss):
        return acc > 0.85
    
    def is_finished(self, results, acc):
        print("is finished")
        print(results)
        print(acc)
        if np.all(results) and acc > 0.85:
            return True
            if self.free_dim_len >= 0.5:
                return True
            else:
                self.free_dim_len = min(self.free_dim_len * 2, 0.5)
                self.set_spec_free_dim_len()
        return False

class MonotonicityTask(Task):
    def call_verification(self, nnet_path, X_spec, Y_spec):
        # print("===== verifying =====")
        status, counterexamples, ratio = Main.MonoIncVerify(nnet_path, X_spec, Y_spec, max_iter=self.max_verify_iter, sampling_size=self.counterexample_sampling_size)
        # print("===== verification done  =====")
        return status, counterexamples, {"ratio": ratio}
    
    def compute_normal_loss(self, model, X, y):
        if len(X) == 0:
            return 0, 0
        X, y = X.to(self.device), y.to(self.device)
        # Compute prediction error
        pred = model(X)
        acc_cnt = self.acc_cnt_fn(pred, y)
        loss = self.loss_fn(pred, y)
        return acc_cnt, loss

    def compute_counter_loss(self, model, X, y):
        if len(X) == 0:
            return 0, 0
        # y is either 1 or -1. 1 for increase. -1 for decrease.
        
        X_aux = torch.clone(X).detach()
        eps = 1e-2
        X_aux[torch.arange(start=0,end=len(X),dtype=torch.long), y[:,0]] += eps * y[:,1]
        X, X_aux = X.to(self.device), X_aux.to(self.device)
        # X_aux is always on the higher side, such that we always expect pred < pred_aux

        # Compute prediction error
        pred_aux = model(X_aux)
        pred = model(X)
        
        # It is suppoed that pred_aux > pred
        # Therefore, there is a loss when pred_aux < pred
        # That is when pred - pred_aux > 0
        loss =  torch.max(torch.tensor(0), pred - pred_aux).mean()
        acc_cnt = (pred_aux > pred).type(torch.long).sum().item()
        
        return acc_cnt, loss
    
    def collate_fn(self, data):
        xs, ys = zip(*data)
        ## get sequence lengths
        lengths = torch.tensor([ y.shape[0] for y in ys ])
        # ## compute mask
        # mask = (lengths == 1)
        # idx = (lengths == 1).nonzero().flatten()
        # ## padd
        X = torch.tensor(xs)
        Y = [ torch.Tensor(y) for y in ys ]
        normal_X = torch.tensor([x for i,x in enumerate(xs) if lengths[i] == 1])
        counter_X = torch.tensor([x for i,x in enumerate(xs) if lengths[i] != 1])
        normal_y = torch.tensor([y for i,y in enumerate(ys) if lengths[i] == 1])
        counter_y = torch.tensor([y for i,y in enumerate(ys) if lengths[i] != 1])
        return (normal_X, counter_X), (normal_y, counter_y)
    
    def get_label(self, counterexample, Y_spec):
        return Y_spec

class CardWikiTask(MonotonicityTask):
    
    def set_params(self):
        self.save_prefix="../model/cardesti_wiki/"
        self.device = "cuda" if torch.cuda.is_available() else "cpu" if torch.cuda.is_available() else "cpu"
        self.epochs = 100000
        self.batch_size = 2000
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.loss_fn = nn.MSELoss()
        
        self.max_verify_iter = 100000
        self.counterexample_sampling_size = 100000
        
        self.prob_hist = []
        self.custom_collate = True
        self.counter_vio = 0

    # FC(1, 4, 300, 1), free_dim_len = 0.01, target_dim_len = 0.4, 1 spec, iter 50000, sampling 10000, 2 spec
    def get_model(self):
        # return models.FC(1, 4, 300, 1)
        return models.FC(1, 4, 500, 1)
        # return models.FC(2, 4, 50, 1)
        # return models.FC(2, 4, 100, 1)
    
    def get_data(self):
        training_data = datasets.CardestiWikiDataset("../data/cardesti_wiki/cardesti_wiki.csv", test=False)
        testing_data = datasets.CardestiWikiDataset("../data/cardesti_wiki/cardesti_wiki.csv", test=True)
        return training_data, testing_data
    
    def get_specs(self):
        
        self.X_specs = []
        self.Y_specs = []
        
        self.free_dim_len = 0.01
        lb = 0.5-self.free_dim_len
        ub = 0.5+self.free_dim_len

        self.target_dim_len = 0.4
        tlb = 0.3
        tub = tlb+self.target_dim_len
        
        self.X_specs.append(([tlb, lb, lb, lb], [tub, ub, ub, ub]))
        self.Y_specs.append([0, -1]) # dim, direction
        
        return self.X_specs, self.Y_specs, self.free_dim_len            
    
    def compute_loss(self, model, X, y):
        normal_X, counter_X = X
        normal_y, counter_y = y
        
        normal_acc_cnt, normal_loss = self.compute_normal_loss(model, normal_X, normal_y)
        counter_acc_cnt, counter_loss = self.compute_counter_loss(model, counter_X, counter_y)
        
        # print("normal loss, counter loss:")
        # print(normal_loss, counter_loss, len(normal_X), len(counter_X))
        
        return normal_acc_cnt + counter_acc_cnt, normal_loss + counter_loss * 10
    
    def compute_counter_loss(self, model, X, y):
        if len(X) == 0:
            return 0, 0
        # y is either 1 or -1. 1 for increase. -1 for decrease.
        
        X_aux = torch.clone(X).detach()
        eps = 5e-4
        X_aux[torch.arange(start=0,end=len(X),dtype=torch.long), y[:,0]] += eps * y[:,1]
        X, X_aux = X.to(self.device), X_aux.to(self.device)
        # X_aux is always on the higher side, such that we always expect pred < pred_aux

        # Compute prediction error
        pred_aux = model(X_aux)
        pred = model(X)
        
        # It is suppoed that pred_aux > pred
        # Therefore, there is a loss when pred_aux < pred
        # That is when pred - pred_aux > 0
        loss =  torch.max(torch.tensor(0), pred - pred_aux).mean()
        acc_cnt = (pred_aux > pred).type(torch.long).sum().item()
        self.counter_vio += (pred > pred_aux).type(torch.long).sum().item()
        # print("counter vio: ", self.counter_vio)
        return acc_cnt, loss
    
    def training_data_spec_check(self):
        return 0
    
    def set_spec_free_dim_len(self):
        pass
    
    def start_verify(self, acc, loss):
        # print("-> counter vio: ", self.counter_vio)
        ret = acc > 0.9 and self.counter_vio == 0
        self.counter_vio = 0
        return ret
    
    def is_finished(self, results, acc):
        print(" verified ====")
        print(np.all(results))
        print(acc > 0.9)
        if np.all(results) and acc > 0.9:
            return True
            if self.free_dim_len >= 0.05:
                return True
            else:
                self.free_dim_len = min(self.free_dim_len * 2, 0.5)
                self.set_spec_free_dim_len()
        return False
    
    def acc_cnt_fn(self, pred, y):
        return (abs(pred - y) < 1e-2).sum()
    


class LinnosTask(MonotonicityTask):
    
    def set_params(self):
        self.save_prefix="../model/linnos/"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.epochs = 100000
        self.batch_size = 1000
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.loss_fn = nn.MSELoss()
        
        self.max_verify_iter = 10000
        self.counterexample_sampling_size = 10000
        
        self.prob_hist = []
        self.custom_collate = True
    
    def get_model(self):
        return models.FC(2, 9, 300, 1)

    def get_data(self):
        training_data = datasets.LinnosDataset("../data/linnos/linnos.csv", test=False)
        testing_data = datasets.LinnosDataset("../data/linnos/linnos.csv", test=True)
        
        return training_data, testing_data
    
    def compute_loss(self, model, X, y):
        normal_X, counter_X = X
        normal_y, counter_y = y
        
        normal_acc_cnt, normal_loss = self.compute_normal_loss(model, normal_X, normal_y)
        counter_acc_cnt, counter_loss = self.compute_counter_loss(model, counter_X, counter_y)
        
        return normal_acc_cnt+counter_acc_cnt, normal_loss + counter_loss * 1e-1

    def get_specs(self):
        
        self.X_specs = []
        self.Y_specs = []
        
        self.target_dim_len = 0.4
        self.free_dim_len = 0.1
        self.fix_dim_len = 0.0

        tlb = 0.5-self.target_dim_len
        tub = 0.5+self.target_dim_len

        flb = 0.5-self.free_dim_len
        fub = 0.5+self.free_dim_len

        lb = 0.5-self.fix_dim_len
        ub = 0.5+self.fix_dim_len
        
        self.X_specs.append(([flb, flb, lb, lb, tlb, lb, lb, lb, lb], [fub, fub, ub, ub, tub, ub, ub, ub, ub]))
        self.Y_specs.append([4, 1]) # dim, direction
        
        self.X_specs.append(([flb, flb, lb, lb, lb, lb, lb, lb, tlb], [fub, fub, ub, ub, ub, ub, ub, ub, tub]))
        self.Y_specs.append([8, 1]) # dim, direction
        
        return self.X_specs, self.Y_specs, self.free_dim_len            

    def training_data_spec_check(self):
        return 0
    

    def set_spec_free_dim_len(self):
        pass
    
    def start_verify(self, acc, loss):
        return acc > 0.85
    
    def is_finished(self, results, acc):
        print("verify results =====")
        print(np.all(results))
        print(acc > 0.85)
        if np.all(results) and acc > 0.85:
            return True
            if self.free_dim_len >= 0.05:
                return True
            else:
                self.free_dim_len = min(self.free_dim_len * 2, 0.5)
                self.set_spec_free_dim_len()
        return False
    
    def acc_cnt_fn(self, pred, y):
        return (abs(pred - y) < 5e-2).sum()
    
class ProbabilityTask(Task):
    
    def initilaize_verification(self):
        super().initilaize_verification()
        self.total_verified_volume = 0 if not self.early_rejection else self.last_verified_volume

    def call_verification(self, nnet_path, X_spec, Y_spec):
        spec_volume = np.prod(np.array(X_spec[1]) - X_spec[0])
        # print(nnet_path)
        # print(X_spec)
        # print(Y_spec)
        status, counterexamples, verified_volume = Main.prob_verify_Ai2z(nnet_path, X_spec, Y_spec, self.desired_prob, min_size=self.min_verify_size, sampling_size=int(spec_volume*self.counterexample_sampling_size+1))
        # print(verified_volume, '/', spec_volume, '=', verified_volume/spec_volume)
        self.total_verified_volume += verified_volume
        return status, counterexamples, {"verified_volume": verified_volume}

    def finalize_verification(self):
        super().finalize_verification()
        verified_prob = self.total_verified_volume / self.total_spec_volume
        self.last_verified_volume = self.total_verified_volume if self.early_rejection else 0
        print("Total verified volume: ", self.total_verified_volume)
        print("Total spec volume:     ", self.total_spec_volume)
        print("Total verified prob:         ", verified_prob)
        self.results = [verified_prob > 0.7]
    

class BloomCrimeTask(ProbabilityTask):
    
    def set_params(self):
        self.save_prefix="../model/bloom_crime/"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.epochs = 100000
        self.batch_size = 100
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        
        self.max_verify_iter = 10000
        self.min_verify_size = 0.001
        self.counterexample_sampling_size = 1000
        self.desired_prob = 0.8
        
        self.prob_hist = []
        # self.time_out = 30
        self.custom_collate = False
        self.last_verified_volume = 0
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10]).to(self.device))


    def get_model(self):
        return models.FC(2, 2, 50, 1)
    
    def get_data(self):
        training_data = datasets.CrimeDataset("../data/bloom_crime/crime.csv", test=False)
        testing_data = datasets.CrimeDataset("../data/bloom_crime/crime.csv", test=True)
        self.negative_data = datasets.CrimeDataset("../data/bloom_crime/crime.csv", test=False)
    
        return training_data, testing_data
    
    def get_specs(self):
        class Rect():
            def __init__(self, lx, ly, hx, hy):
                self.lx = lx
                self.ly = ly
                self.hx = hx
                self.hy = hy
            def divide(self):
                mx = (self.lx + self.hx)/2
                my = (self.ly + self.hy)/2

                if self.hx - self.lx > self.hy - self.ly:
                    return Rect(self.lx, self.ly, mx, self.hy), Rect(mx, self.ly, self.hx, self.hy)
                else:
                    return Rect(self.lx, self.ly, self.hx, my), Rect(self.lx, my, self.hx, self.hy)

        def check(rect, data):
            exists = (data[:,0] >= rect.lx) & (data[:,0] <= rect.hx) & (data[:,1] >= rect.ly) & (data[:,1] <= rect.hy)
            return np.any(exists)

        def get_safe_rects(depth, max_depth, rect, data):
            contain_data = check(rect, data)

            if not contain_data:
                return [rect]

            if depth > max_depth:
                return []

            rect1, rect2 = rect.divide()

            return get_safe_rects(depth+1, max_depth, rect1, data) + get_safe_rects(depth+1, max_depth, rect2, data)

        self.safe_rects = get_safe_rects(0, 9, Rect(0,0,1,1), np.array(self.negative_data.xs))
        self.total_spec_volume = np.sum([(r.hx-r.lx)*(r.hy-r.ly) for r in self.safe_rects])
        
        self.X_specs = []
        self.Y_specs = []
        
        for rect in self.safe_rects:
            self.X_specs.append(([rect.lx, rect.ly], [rect.hx, rect.hy]))
            self.Y_specs.append([np.ones((1,1)), np.zeros(1)])

        self.free_dim_len = 0.51
        
        return self.X_specs, self.Y_specs, self.free_dim_len            
    
    def training_data_spec_check(self):
        return 0
    
    def draw(self):
        fig, ax = plt.subplots()
        xs = np.array(self.training_data.xs)
        ax.scatter(xs[:,0], xs[:,1], c=self.training_data.ys, s=2)
        for rect in self.safe_rects:
            ax.add_patch(patches.Rectangle((rect.lx, rect.ly), rect.hx-rect.lx, rect.hy-rect.ly, linewidth=1, edgecolor='orange', facecolor='orange', alpha=.5))
        plt.show()

    def get_label(self, x, Y_spec):
        return np.zeros(1).astype('float32')

    def set_spec_free_dim_len(self):
        pass
    
    def start_verify(self, train_acc, train_loss):
        if train_acc < 0.85:
            return False
        print("testing negtive data")
        # self.draw()
        neg_acc, neg_loss = self.test(self.model, self.negative_data, self.compute_loss)
        # neg_acc, neg_loss = self.test_negative_data(self.negative_dataloader, self.model, self.loss_fn, self.device, self.acc_cnt_fn)
        # print("Negative data accuracy:", neg_acc)
        return train_acc > 0.85 and neg_acc > 0.90

    
    def is_finished(self, results, acc):
        # self.draw()
        return np.all(results)

    
    def acc_cnt_fn(self, pred, y):
        return (((pred > 0) != y).sum(axis=-1) == 0).sum()
    
 
class DBIndexTask(Task):
    def __init__(self, add_counterexample = False,  incremental_training = False, batch_counterexample = False, early_rejection = False, incremental_verification = False, start_finetune_epoch = -1, time_out=60, task_index=0, v2_num=10):
        self.add_counterexample = add_counterexample
        self.incremental_training = incremental_training
        self.batch_counterexample = batch_counterexample
        self.early_rejection = early_rejection
        self.incremental_verification = incremental_verification
        self.start_finetune_epoch = start_finetune_epoch
        self.time_out = time_out

        self.task_index = task_index
        self.v2_num = v2_num
        
        self.set_seed()
        
        self.training_data, self.testing_data = self.get_data()
        self.counterexample_data = datasets.EmptyDataset()
        self.X_specs, self.Y_specs, self.free_dim_len = self.get_specs()
        self.spec_check_list = list(zip(self.X_specs, self.Y_specs))
        self.model = self.get_model()
        
        self.set_params()

        self.reset_cnt = 0

        if not add_counterexample:
            self.counterexample_sampling_size = 0
        
        self.task_index = task_index
        self.is_polytope_spec = False
    
    @property
    def save_name(self):
        return self.__class__.__name__ + "_" +str(self.task_index) + "_" + ''.join(list(map(lambda x: str(int(x)), [self.add_counterexample, self.incremental_training, self.batch_counterexample, self.early_rejection, self.incremental_verification, self.start_finetune_epoch]))) + "_" + str(self.time_out)

    def get_model(self):
        if self.task_index == 0:
            return models.FC(1, 1, 1000, 1)
        else:
            return models.FC(1, 1, 100, 1)
    
    def training_data_spec_check(self):
        return 0
    
    def get_data(self):
        
        v1_data = datasets.LognormalDataset()
        n = len(v1_data) // self.v2_num
        v2_datas = [datasets.LognormalDataset(start=i*n, end=(i+1)*n) for i in range(self.v2_num)]
        
        training_data = v1_data if self.task_index == 0 else v2_datas[self.task_index-1]
        testing_data = training_data
        
        self.all_data = datasets.LognormalDataset()
        
        return training_data, testing_data

    def acc_cnt_fn(self, pred, y):
        if self.task_index == 0:
            return (abs(pred - y) < 1e-1).sum()
        else:
            return (abs(pred - y) < 1e-2).sum()
    
    def call_verification(self, nnet_path, X_spec, Y_spec):
        status, counterexamples = Main.verify(nnet_path, X_spec, Y_spec, 
                                                   max_iter=self.max_verify_iter, 
                                                   sampling_size=self.counterexample_sampling_size,
                                                   is_polytope=self.is_polytope_spec)
        return status, counterexamples, None

    def set_params(self):
        self.save_prefix="../model/dbindex/"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.epochs = 100000
        self.batch_size = 1000
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.loss_fn = nn.MSELoss()
        
        self.max_verify_iter = 1000
        self.counterexample_sampling_size = 100
        self.is_polytope_spec = False
        self.custom_collate = False
        # self.y_is_class = False
        # self.time_out = 30

    def get_specs(self, normalize=False):
        X_specs = []
        Y_specs = []
        spec_num = 20 if self.task_index == 0 else 1
        
        x_min = np.min(np.ravel(self.training_data.xs))
        x_max = np.max(np.ravel(self.training_data.xs))
        y_min = np.min(np.ravel(self.training_data.ys))
        y_max = np.max(np.ravel(self.training_data.ys))
        
        dx = (x_max - x_min) / spec_num
        x_padding = 0
        for i in range(spec_num):
            x_l = x_min + i*dx 
            x_r = x_min + (i+1)*dx
            y_l = self.all_data.ys[max(0,np.searchsorted(self.all_data.xs, x_l)-1)]
            y_r = self.all_data.ys[min(len(self.all_data)-1,np.searchsorted(self.all_data.xs, x_r))]
            y_padding = (y_r - y_l)/4
            # print(x_l, x_r, y_l, y_r)
            
            X_spec = (np.array([x_l + x_padding]).astype(np.float), np.array([x_r - x_padding]).astype(np.float))            
            Y_spec = [np.vstack([-1., 1.]).astype(np.float), np.array([-(y_l - y_padding), y_r + y_padding]).astype(np.float)]
            X_specs.append(X_spec)
            Y_specs.append(Y_spec)

        free_dim_len = 0.5
        
        return X_specs, Y_specs, free_dim_len

    def get_label(self, x, Y_spec):
        idx = np.searchsorted(self.all_data.xs, x)[0]
        # print(idx)
        # print(self.all_data.xs[idx-10:idx+1])
        # print(self.all_data.xs[idx-1])
        prev_x = self.all_data.xs[idx-1]
        next_x = self.all_data.xs[idx]
        prev_y = self.all_data.ys[idx-1]
        next_y = self.all_data.ys[idx]
        return (x - prev_x) / (next_x - prev_x) * (next_y - prev_y) + prev_y

    def draw(self):
        dataloader = DataLoader(self.testing_data, batch_size=100000)
        self.model.eval()
        plt.figure()
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                # Compute prediction error
                pred = self.model(X)
                plt.scatter(X, y, c='g')
                plt.scatter(X, pred, c='r')
        plt.show()
        
    def set_spec_free_dim_len(self):
        pass
    
    def start_verify(self, acc, loss):
        return acc > 0.85
    
    def is_finished(self, results, acc):
        # self.draw()
        if np.all(results) and acc > 0.85:
            if self.free_dim_len >= 0.5:
                return True
            else:
                self.free_dim_len = min(self.free_dim_len * 2, 0.5)
                self.set_spec_free_dim_len()
        return False
