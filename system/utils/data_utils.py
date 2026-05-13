import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from .dataset_action import NTURGBD
from .pkudataset import LinearFeeder
hrnet_data_dir = '../dataset/ntu120hrnet/ntu120_hrnet.pkl'
hrnet_train_dir = '../dataset/splithrnet/train/'
hrnet_test_dir = '../dataset/splithrnet/test/'
def read_data(dataset, idx, is_train=True):
    if is_train:
        if dataset == "pku2test-01":
            train_data_dir = os.path.join('../../pku_raw/v2/XSub/refed/fedtrain01/')

            train_file = train_data_dir + str(idx) + '.npz'
            with open(train_file, 'rb') as f:
                train_data = np.load(f, allow_pickle=True)['data'].tolist()

            return train_data
        #/opt/data/private/ReverFed/ntu60/01result/xsub/fedtrain
        elif dataset == "pku2xsub-01":
            train_data_dir = os.path.join('../../pku_raw/v2/XSubresize/fedtrain01/')

            train_file = train_data_dir + str(idx) + '.npz'
            with open(train_file, 'rb') as f:
                train_data = np.load(f, allow_pickle=True)['data'].tolist()

        

            traindata=dict()

            traindata['x']=train_data['x']
            traindata['y']=train_data['y']

            return traindata

        elif dataset == "ntu60-01":
            
            train_data_dir = os.path.join('../dataset/precessdata/NTU60_frame50/shard_0.1/xview/fedtrain/')

            train_file = train_data_dir + str(idx) + '.npz'
            with open(train_file, 'rb') as f:
                train_data = np.load(f, allow_pickle=True)['data'].tolist()

        

            traindata=dict()

            traindata['x']=train_data['x']
            traindata['y']=train_data['y']

            return traindata

        elif dataset == "dir0.1":
            
            train_data_dir = os.path.join('../dataset/precessdata/NTU60_frame50/dir0.1/fedtrain/')

            train_file = train_data_dir + str(idx) + '.npz'
            with open(train_file, 'rb') as f:
                train_data = np.load(f, allow_pickle=True)['data'].tolist()

        

            traindata=dict()

            traindata['x']=train_data['x']
            traindata['y']=train_data['y']

            return traindata
        elif dataset == "ntu60":
            
            train_data_dir = os.path.join('../dataset/precessdata/NTU60_frame50/dir0.1/fedtrain/')

            train_file = train_data_dir + str(idx) + '.npz'
            with open(train_file, 'rb') as f:
                train_data = np.load(f, allow_pickle=True)['data'].tolist()

        

            traindata=dict()

            traindata['x']=train_data['x']
            traindata['y']=train_data['y']

            return traindata
        elif dataset == "small":
            train_data_dir = os.path.join('../dataset/precessdata/NTU60_frame50/shard_small/xview/fedtrain/')

            train_file = train_data_dir + str(idx) + '.npz'
            with open(train_file, 'rb') as f:
                train_data = np.load(f, allow_pickle=True)['data'].tolist()

            traindata=dict()

            traindata['x']=train_data['x']
            traindata['y']=train_data['y']

            return traindata
        elif dataset == "pku2old-noniid":
            
            train_data_dir = os.path.join('../../../dataset/pku2/XSubfedtrain_noniid/')

            train_file = train_data_dir + str(idx) + '.npz'
            with open(train_file, 'rb') as f:
                train_data = np.load(f, allow_pickle=True)['data'].tolist()

        

            traindata=dict()

            traindata['x']=train_data['x']
            traindata['y']=train_data['y']

            return traindata
        elif dataset == "pku2xview-01":
            train_data_dir = os.path.join('../../../dataset/pku2/XViewfedtrain01/')

            train_file = train_data_dir + str(idx) + '.npz'
            with open(train_file, 'rb') as f:
                train_data = np.load(f, allow_pickle=True)['data'].tolist()

            traindata=dict()

            traindata['x']=train_data['x']
            traindata['y']=train_data['y']

            return traindata
        
        elif dataset == "pku1xview-01":
            train_data_dir = os.path.join('../../../dataset/pku1/XSubfedtrain_01/')

            train_file = train_data_dir + str(idx) + '.npz'
            with open(train_file, 'rb') as f:
                train_data = np.load(f, allow_pickle=True)['data'].tolist()

            traindata=dict()

            traindata['x']=train_data['x']
            traindata['y']=train_data['y']

            return traindata   
        
        elif dataset == "ntu120-dir":
            train_data_dir = os.path.join('/opt/data/private/dataset/precessdata/NTU120_frame50/dir/xview/fedtrain/')

            train_file = train_data_dir + str(idx) + '.npz'
            with open(train_file, 'rb') as f:
                train_data = np.load(f, allow_pickle=True)['data'].tolist()

            traindata=dict()

            traindata['x']=train_data['x']
            traindata['y']=train_data['y']

            return traindata   
        elif dataset == "ntu120-shard":
            train_data_dir = os.path.join('/opt/data/private/dataset/precessdata/NTU120_frame50/shard/xview/fedtrain/')

            train_file = train_data_dir + str(idx) + '.npz'
            with open(train_file, 'rb') as f:
                train_data = np.load(f, allow_pickle=True)['data'].tolist()

            traindata=dict()

            traindata['x']=train_data['x']
            traindata['y']=train_data['y']

            return traindata   
        elif dataset == "pku2-noniid":
            train_data_dir = os.path.join('../../pku_raw/v2/XSub/noniid/fedtrain/')

            train_file = train_data_dir + str(idx) + '.npz'
            with open(train_file, 'rb') as f:
                train_data = np.load(f, allow_pickle=True)['data'].tolist()

            traindata=dict()

            traindata['x']=train_data['x']
            traindata['y']=train_data['y']

            return traindata   
        
        elif dataset == "pku1-noniid":
            train_data_dir = os.path.join('../../../dataset/pku1/XSubfedtrain_noniid/')

            train_file = train_data_dir + str(idx) + '.npz'
            with open(train_file, 'rb') as f:
                train_data = np.load(f, allow_pickle=True)['data'].tolist()

            traindata=dict()

            traindata['x']=train_data['x']
            traindata['y']=train_data['y']

            return traindata
        
        elif dataset == "mnist":
            # MNIST data loaded from ../data/mnist/client_{id}
            # idx is 0-based, folder is 1-based (client_1, client_2...)
            client_id = idx + 1
            # Assuming data_utils.py is in system/utils, so ../data/mnist is system/data/mnist
            data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'mnist', f'client_{client_id}')
            
            data_path = os.path.join(data_dir, 'data.npy')
            label_path = os.path.join(data_dir, 'label.npy')
            
            if not os.path.exists(data_path) or not os.path.exists(label_path):
                raise FileNotFoundError(f"MNIST data not found for client {client_id} at {data_dir}. Please run prepare_client_data.py first.")
                
            traindata = dict()
            # Load as numpy arrays
            traindata['x'] = np.load(data_path, allow_pickle=True)
            traindata['y'] = np.load(label_path, allow_pickle=True)
            
            return traindata 
        else:
            raise NotImplementedError

    else:
        if dataset == "pku2test-01":
            test_data  = LinearFeeder(
                data_path="../../pku_raw/data/PKU-MMD-v2-AGCN/xsub/val_data_joint.npy",
                label_path="../../pku_raw/data/PKU-MMD-v2-AGCN/xsub/val_label.pkl",
                num_frame_path="../../pku_raw/data/PKU-MMD-v2-AGCN/xsub/val_num_frame.npy",
                l_ratio=[0.95],
                input_size=64,
                input_representation="graph-based",
                mmap=True)

            return test_data 
        elif dataset == "ntu120-shard":
            test_data_dir = os.path.join('/opt/data/private/dataset/precessdata/NTU120_frame50/shard/xview/fedtest/')

            test_file = test_data_dir + str(idx) + '.npz'
            with open(test_file, 'rb') as f:
                test_data = np.load(f, allow_pickle=True)['data'].tolist()
            
            testdata=dict()
            testdata['x']=test_data['x']
            testdata['y']=test_data['y']

            return testdata
        elif dataset == "pku2old-noniid":
            test_data_dir = os.path.join('../../dataset/pku2/XSubfedtest_noniid/')

            test_file = test_data_dir + str(idx) + '.npz'
            with open(test_file, 'rb') as f:
                test_data = np.load(f, allow_pickle=True)['data'].tolist()
            
            testdata=dict()
            testdata['x']=test_data['x']
            testdata['y']=test_data['y']

            return testdata
        elif dataset == "ntu60-01":
            test_data_dir = os.path.join('../dataset/precessdata/NTU60_frame50/shard_0.1/xview/fedtest/')

            test_file = test_data_dir + str(idx) + '.npz'
            with open(test_file, 'rb') as f:
                test_data = np.load(f, allow_pickle=True)['data'].tolist()
            
            testdata=dict()
            testdata['x']=test_data['x']
            testdata['y']=test_data['y']

            return testdata
        elif dataset == "small":
            test_data_dir = os.path.join('../dataset/precessdata/NTU60_frame50/shard_small/xview/fedtest/')

            test_file = test_data_dir + str(idx) + '.npz'
            with open(test_file, 'rb') as f:
                test_data = np.load(f, allow_pickle=True)['data'].tolist()
            
            testdata=dict()
            testdata['x']=test_data['x']
            testdata['y']=test_data['y']

            return testdata
        elif dataset == "dir0.1":
            test_data_dir = os.path.join('../dataset/precessdata/NTU60_frame50/dir0.1/fedtest/')

            test_file = test_data_dir + str(idx) + '.npz'
            with open(test_file, 'rb') as f:
                test_data = np.load(f, allow_pickle=True)['data'].tolist()
            
            testdata=dict()
            testdata['x']=test_data['x']
            testdata['y']=test_data['y']

            return testdata
        elif dataset == "ntu60":
            test_data_dir = os.path.join('../dataset/precessdata/NTU60_frame50/dir0.1/fedtest/')

            test_file = test_data_dir + str(idx) + '.npz'
            with open(test_file, 'rb') as f:
                test_data = np.load(f, allow_pickle=True)['data'].tolist()
            
            testdata=dict()
            testdata['x']=test_data['x']
            testdata['y']=test_data['y']

            return testdata
        elif dataset == "pku2xview-01":
            train_data_dir = os.path.join('../../../dataset/pku2/XViewfedtest01/')

            train_file = train_data_dir + str(idx) + '.npz'
            with open(train_file, 'rb') as f:
                train_data = np.load(f, allow_pickle=True)['data'].tolist()

            traindata=dict()

            traindata['x']=train_data['x']
            traindata['y']=train_data['y']

            return traindata
        elif dataset == "pku1xview-01":
            train_data_dir = os.path.join('../../../dataset/pku1/XSubfedtest_01/')

            train_file = train_data_dir + str(idx) + '.npz'
            with open(train_file, 'rb') as f:
                train_data = np.load(f, allow_pickle=True)['data'].tolist()

            traindata=dict()

            traindata['x']=train_data['x']
            traindata['y']=train_data['y']

            return traindata

        elif dataset == "ntu120-dir":
            train_data_dir = os.path.join('/opt/data/private/dataset/precessdata/NTU120_frame50/dir/xview/fedtest/')

            train_file = train_data_dir + str(idx) + '.npz'
            with open(train_file, 'rb') as f:
                train_data = np.load(f, allow_pickle=True)['data'].tolist()

            traindata=dict()

            traindata['x']=train_data['x']
            traindata['y']=train_data['y']

            return traindata
        
        elif dataset == "pku2-noniid":
            train_data_dir = os.path.join('../../pku_raw/v2/XSub/noniid/fedtest/')

            train_file = train_data_dir + str(idx) + '.npz'
            with open(train_file, 'rb') as f:
                train_data = np.load(f, allow_pickle=True)['data'].tolist()

            traindata=dict()

            traindata['x']=train_data['x']
            traindata['y']=train_data['y']

            return traindata
        elif dataset == "pku1-noniid":
            train_data_dir = os.path.join('../../../dataset/pku1/XSubfedtest_noniid/')

            train_file = train_data_dir + str(idx) + '.npz'
            with open(train_file, 'rb') as f:
                train_data = np.load(f, allow_pickle=True)['data'].tolist()

            traindata=dict()

            traindata['x']=train_data['x']
            traindata['y']=train_data['y']

            return traindata
        else:
            raise NotImplementedError

def resize_torch_interp_batch(self, data, target_frame):

        window = target_frame

        n,c,t,v,m = data.shape 
        data = data.permute(0,1,3,4,2).contiguous().view(n,c*v*m,t)
        data = data[:,:,:,None]
        data = F.interpolate(data, size=(window, 1), mode='bilinear',align_corners=False).squeeze(dim=3)
        data = data.reshape(n,c,v,m,window)
        data = data.permute(0,1,4,2,3).contiguous()
        return data 
        
def read_client_data(dataset, idx, is_train=True,semi=None):
    # if dataset[:2] == "ag" or dataset[:2] == "SS":
    #     return read_client_data_text(dataset, idx, is_train)
    # elif dataset[:2] == "sh":
    #     return read_client_data_shakespeare(dataset, idx)

    if is_train:
        train_data = read_data(dataset, idx, is_train)
        #print(train_data.keys)
        X_train = torch.Tensor(train_data['x']).type(torch.float32)
        y_train = torch.Tensor(train_data['y']).type(torch.int64)
        if y_train.shape == torch.Size([1]):
            y_train = y_train
        else:
            y_train = y_train.squeeze()
        # y_train = y_train.squeeze()
        # print(X_train.shape)
        # print(y_train.shape)
        # print(f"label min: {y_train.min()}, label max: {y_train.max()}")
        if semi!=None:
            N,C,T,V,M=X_train.shape
            idx = np.arange(N)
            np.random.shuffle(idx)
            
            N_used = int(N * semi)
            # print(N_used)
            idx_used = idx[ :N_used] 
            N = N_used
            X_train = X_train[idx_used]
            y_train = y_train[idx_used]
        # print(semi)
        # print(X_train.shape)
        train_data = [(x, y) for x, y in zip(X_train, y_train)]
        return train_data
    elif is_train == False and dataset != "pku2test-01":
        test_data = read_data(dataset, idx, is_train)
        X_test = torch.Tensor(test_data['x']).type(torch.float32)
        y_test = torch.Tensor(test_data['y']).type(torch.int64)
        y_test = y_test.squeeze()
        # hx_test = torch.Tensor(test_data['hrx']).type(torch.float32)
        # hy_test = torch.Tensor(test_data['hry']).type(torch.int64)
        # print(X_test.shape)
        # print(y_test.shape)
        # print(f"test label min: {y_test.min()}, label max: {y_test.max()}")
        test_data = [(x,y) for x, y in zip(X_test, y_test)]
        return test_data
    elif is_train == False and dataset == "pku2test-01":
        test_data = read_data(dataset, idx, is_train)
        return test_data
    else:
        raise NotImplementedError

def read_client_traindata(dataset, idx, is_train=True):

    train_data = read_data(dataset, idx, is_train)
    #print(train_data)

    X_train = torch.Tensor(train_data['x']).type(torch.float32)
    y_train = torch.Tensor(train_data['y']).type(torch.int64)
    # print("X_train:{}".format(X_train.shape))
    # point = X_train[0,:,0,:,0]
    # for i in range(25):
    #     print(point[:,i])
    # assert False,"run"
    return X_train,y_train

# def read_client_data_text(dataset, idx, is_train=True):
#     if is_train:
#         train_data = read_data(dataset, idx, is_train)
#         X_train, X_train_lens = list(zip(*train_data['x']))
#         y_train = train_data['y']

#         X_train = torch.Tensor(X_train).type(torch.int64)
#         X_train_lens = torch.Tensor(X_train_lens).type(torch.int64)
#         y_train = torch.Tensor(train_data['y']).type(torch.int64)

#         train_data = [((x, lens), y) for x, lens, y in zip(X_train, X_train_lens, y_train)]
#         return train_data
#     else:
#         test_data = read_data(dataset, idx, is_train)
#         X_test, X_test_lens = list(zip(*test_data['x']))
#         y_test = test_data['y']

#         X_test = torch.Tensor(X_test).type(torch.int64)
#         X_test_lens = torch.Tensor(X_test_lens).type(torch.int64)
#         y_test = torch.Tensor(test_data['y']).type(torch.int64)

#         test_data = [((x, lens), y) for x, lens, y in zip(X_test, X_test_lens, y_test)]
#         return test_data


# def read_client_data_shakespeare(dataset, idx, is_train=True):
#     if is_train:
#         train_data = read_data(dataset, idx, is_train)
#         X_train = torch.Tensor(train_data['x']).type(torch.int64)
#         y_train = torch.Tensor(train_data['y']).type(torch.int64)

#         train_data = [(x, y) for x, y in zip(X_train, y_train)]
#         return train_data
#     else:
#         test_data = read_data(dataset, idx, is_train)
#         X_test = torch.Tensor(test_data['x']).type(torch.int64)
#         y_test = torch.Tensor(test_data['y']).type(torch.int64)
#         test_data = [(x, y) for x, y in zip(X_test, y_test)]
#         return test_data

