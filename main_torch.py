import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset
import numpy as np
from sklearn.neighbors import NearestNeighbors
from random import randint, seed
import os
from quant import entropy_by_quantization
from cluster_test import compute_cluster_distances, pairwise_dist
from scipy.stats import spearmanr


# Constants
NUM_ITERS = 5000 #originally 20K
DISPLAY_STEP = 100
BATCH_SIZE = 128 #  originally 128, but training too quickly so we reduce
LAYERS = 4
SEED = 246
RELU = True #DON'T FORGET TO CHANGE
# Set random seeds for reproducibility
torch.manual_seed(SEED)

# Data pre-processing
transform = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.ToTensor()
])

# Load MNIST dataset
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# DataLoader for batching
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

# MI set
T = 10000
mi_loader = DataLoader(Subset(train_data, range(T)), batch_size=T, shuffle=False)

# Convert to TensorFlow Tensors for X_MI and Y_MI
for images, labels in mi_loader:
    X_MI = images.view(T,-1)
    Y_MI = labels

print(f"X_MI shape: {X_MI.shape}")
print(f"Y_MI shape: {Y_MI.shape}")

## MI methods
from EDGE_4_2_0 import EDGE
global dist0
dist0 = np.zeros(4)

# # get mutual information for all hidden layers
def get_MI_EDGE(hiddens, ep_idx):
    mi_xt_list = []; mi_ty_list = []
    #hidden = hiddens[1]
    hidden_idx = 0
    for hidden in hiddens:
	    #H = np.array(hidden)
        #print('get_MI_EDGE ',H.shape)        
    
        mi_xt, mi_ty = calc_MI_EDGE(hidden,hidden_idx,ep_idx)
        mi_xt_list.append(mi_xt)
        mi_ty_list.append(mi_ty)
        hidden_idx +=1

    return mi_xt_list, mi_ty_list

def calc_MI_EDGE(hidden, layer_idx ,ep_idx):
    global rho_0
    
    hidden = np.array(hidden)[:T,:]

    #print('calc_MI_EDGE',hidden.shape)
    d=hidden.shape[1]
    #print(hidden.shape)
    X_reshaped = np.reshape(np.array(X_MI),[-1,784]) # vectorize X
    Y_reshaped = np.array(Y_MI) # convert 10-dim data to class integer in [0,9], not necessary since label still in format

    #print(X_reshaped.shape, Y_reshaped.shape)
    
    dist=0
    if ep_idx <=20:
        dist0[layer_idx] = av_distance(hidden)
        r = 1
    else:
        dist = av_distance(hidden)
        r = dist / dist0[layer_idx]

    print('ep_idx and hidden dim and r', ep_idx, hidden.shape[1] ,r, dist)    

     
    # Normalize hidden
    #hidden = hidden/r
    smoothness_vector_xt = np.array([0.8, 1.0, 1.2, 1.8])
    smoothness_vector_ty = np.array([0.4, 0.5, 0.6, 0.8])

    mi_xt_py = EDGE(X_reshaped, hidden,U=20, L_ensemble=10, gamma=[0.2,  smoothness_vector_xt[layer_idx]], epsilon_vector= 'range') #,U=20, gamma=[0.2,  2*smoothness_vector[layer_idx]], epsilon=[0.2,r*0.2], hashing='p-stable') 
    mi_ty_py = EDGE(Y_reshaped, hidden,U=10, L_ensemble=10, gamma=[0.0001, smoothness_vector_ty[layer_idx]], epsilon=[0.2,0.2], epsilon_vector= 'range')

    return mi_xt_py, mi_ty_py

# Find average distances between points
from numpy import linalg as LA
def av_distance(X):

    r = 1000

    N = X.shape[0]

    np.random.seed(1234)
    T1= np.random.choice(range(N), size=2*r)[:r]
    T2= np.random.choice(range(N), size=2*r)[r:]
    np.random.seed()
    D = LA.norm(X[T2,:] - X[T1,:], ord=2, axis=1)
    d = np.mean(D)

    return d

# Define the neural network model
class MLPModel(nn.Module):
    def __init__(self):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(784, 1024)  # 784 pixels
        self.fc2 = nn.Linear(1024, 20)
        self.fc3 = nn.Linear(20, 20)
        self.fc4 = nn.Linear(20, 20)
        self.fc5 = nn.Linear(20, 10)  

        nn.init.normal_(self.fc1.weight, mean=0, std=0.1)
        nn.init.normal_(self.fc2.weight, mean=0, std=0.1)
        nn.init.normal_(self.fc3.weight, mean=0, std=0.1)
        nn.init.normal_(self.fc4.weight, mean=0, std=0.1)
        nn.init.normal_(self.fc5.weight, mean=0, std=0.1)

        # nn.init.xavier_uniform_(self.fc1.weight)
        # nn.init.xavier_uniform_(self.fc2.weight)
        # nn.init.xavier_uniform_(self.fc3.weight)
        # nn.init.xavier_uniform_(self.fc4.weight)
        # nn.init.xavier_uniform_(self.fc5.weight)

        self.dropout = nn.Dropout(p=0.5)




    def forward(self, x):
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.dropout(torch.relu(self.fc2(x)))
        x = self.dropout(torch.relu(self.fc3(x)))
        x = self.dropout(torch.relu(self.fc4(x)))
        x = self.fc5(x) 
        return x

def extract_hidden_layers(model, x):
    with torch.no_grad():
        layers = []
        x = x.view(-1, 784)
        layers.append(torch.relu(model.fc1(x)))
        layers.append(torch.relu(model.fc2(layers[-1])))
        layers.append(torch.relu(model.fc3(layers[-1])))
        layers.append(torch.relu(model.fc4(layers[-1])))
        return layers


# Instantiate the model
model = MLPModel()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function
def train_with_mi():
    model.train() 
    mi_xt_all = []
    mi_ty_all = []
    epochs = []
    train_losses = list()
    train_accs = list()
    test_losses = list()
    test_accs = list()
    
    for epoch in range(NUM_ITERS + 1):
        model.train()
        print(f"EPOCH: {epoch}")
        
        for i, (batch_X, batch_Y) in enumerate(train_loader):
            optimizer.zero_grad()

            # Flatten the images and pass through the network
            batch_X = batch_X.view(-1, 784)
            output = model(batch_X)

            # Compute loss and backpropagate
            loss = criterion(output, batch_Y)
            loss.backward()
            optimizer.step()
            
            #print(f"Loss for batch {i}: {loss.item()}")

        # Evaluate on train and test datasets
        if epoch % DISPLAY_STEP == 0 or (epoch % 10 == 0 and epoch <= 150):
            train_loss, train_acc = evaluate(model, train_loader)
            test_loss, test_acc = evaluate(model, test_loader)
            
            print(f"#{epoch} Trn acc={train_acc} , Trn loss={train_loss} Tst acc={test_acc} , Tst loss={test_loss}")
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            test_losses.append(test_loss)
            test_accs.append(test_acc)
            
            train_metrics = np.zeros((1,5))
            train_metrics[0,:] = np.array([train_loss,train_acc,test_loss,test_acc,epoch])
            append_to_npy(train_metrics, file_path=train_path)

        # Compute MI
        #
        q = 1
        A_ = epoch <= 10 and epoch % 1 == 0
        A0 = epoch > 10 and epoch <= 100 and epoch % (3*q) == 0     
        A1 = epoch > 100 and epoch <= 1000 and epoch % (25*q) == 0    
        A2 = epoch > 1000 and epoch <= 2000 and epoch % (50*q) == 0
        A3 = epoch > 2000 and epoch <= 4000 and epoch % (200*q) == 0
        A4 = epoch > 4000 and epoch % (400*q) == 0

        #if A0 or A1 or A2:
        if   A_ or A0 or A1 or A2 or A3 or A4:
            print("COMPUTING MI using EDGE, extracting hidden_layers")
            
            hidden_layers = extract_hidden_layers(model, X_MI)
            if epoch == 0 or epoch == 2 or epoch == 30 or epoch == 75 or epoch == 125 or epoch == 250 or epoch == 500:
                for i in range(len(hidden_layers)):
                    np.save(f"{epoch}_hidden_layer_{i}.npy", hidden_layers[i])
            # for i, output in enumerate(hidden_layers):
            #     print(f"Layer {i+1} has shape {output.shape}")

            # for i, output in enumerate(hidden_layers):
            #     print(f"Layer {i+1} has shape {output.shape}")
            
            #H = np.array(hidden_layers[0])
            #print('hidden_layers', H.shape)
            print("GETTING MI EDGE")
            mi_xt, mi_ty = get_MI_EDGE(hidden_layers, epoch)
            
            print('MI(X;T): ',mi_xt,'MI(Y;T): ', mi_ty)
            
            mi_xt_all.append(mi_xt)
            mi_ty_all.append(mi_ty)
            epochs.append(epoch)

            mi_to_append = np.zeros((2,1,len(mi_xt)))
            mi_to_append[0,0,:] = mi_xt
            mi_to_append[1,0,:] = mi_ty
            append_to_npy(mi_to_append, file_path=mi_path, mi_edge=True)
            append_to_npy(np.array([epoch]), file_path=mi_epoch_path)         

            print("COMPUTING CLUSTERING WITH QUANTIZATION")
            if RELU:
                n_bins = [10,30,100]
                h_quantized = np.zeros((1,len(hidden_layers),len(n_bins)))
                for i in range(3):
                    h_quantized[0,:,i] = np.array(entropy_by_quantization(hidden_layers, n_bins[i], relu=True))
                print(f"Quantized entropy: {h_quantized[0,:,:]}" )
                append_to_npy(h_quantized, file_path=binned_path)
            else:
                n_bins = [2,5,10] 
                h_quantized = np.zeros((1,len(hidden_layers),len(n_bins)))
                for i in range(3):
                    h_quantized[0,:,i] = np.array(entropy_by_quantization(hidden_layers, n_bins[i]))
                print(f"Quantized entropy: {h_quantized[0,:,:]}" )
                append_to_npy(h_quantized, file_path=binned_path)

            print("COMPUTING CLASS DISTANCES TO MEASURE CLUSTERING")
            distances = np.zeros((1,len(hidden_layers),10))
            distances[0,:,:] = compute_cluster_distances(hidden_layers, Y_MI.numpy())
            print(f"distances: {distances[0,:,:]}")
            append_to_npy(distances, file_path=clusering_path)
            


    return np.array(mi_xt_all), np.array(mi_ty_all), np.array(epochs)

# Evaluate function to compute loss and accuracy
def evaluate(model, data_loader):
    model.eval()  
    correct = 0
    total = 0
    loss = 0.0
    with torch.no_grad():
        for batch_X, batch_Y in data_loader:
            batch_X = batch_X.view(-1, 784)
            output = model(batch_X)
            loss += criterion(output, batch_Y).item()
            _, predicted = torch.max(output, 1)
            total += batch_Y.size(0)
            
            correct += (predicted == batch_Y).sum().item()

    return loss / len(data_loader), correct / total



# Continuously update
mi_path = f"mi_all_{LAYERS}_{SEED}.npy" # store mi values
mi_epoch_path = f"mi_epoch_{LAYERS}_{SEED}.npy" # store epochs where mi was evaluated
binned_path = f"binned_entropy_{LAYERS}_{SEED}.npy" # stores H(Bin(T_l))
clusering_path = f"cluster_distances_{LAYERS}_{SEED}.npy" # stores average 2-norm of latent rep's distances from centroid per label
train_path = f"train_{LAYERS}_{SEED}.npy" # store training and test metrics 
corr_path = f"corr_{LAYERS}_{SEED}.npy"

def append_to_npy(new_data, file_path, mi_edge=False):
    # Check if file exists
    if os.path.exists(file_path):
        existing_data = np.load(file_path)
        if mi_edge:
            updated_data = np.concatenate((existing_data, new_data), axis=1)
        else:
            updated_data = np.concatenate((existing_data, new_data), axis=0)
    else:
        updated_data = new_data

    np.save(file_path, updated_data)


def pairwise_distance_plotter():
    labels_s = Y_MI.numpy()
    print(labels_s.shape)
    for epoch in [0,2,30,75,125]:
        for i in range(4):
            print(f"EPOCH {epoch} layer {i}")
            layer = np.load(f"results/final_tanh_246_sd01/{epoch}_hidden_layer_{i}.npy")
            print(layer.shape)
    
            pairwise_dist(layer, labels_s, layer=i, epoch=epoch)


# Run the training process
#print("START TRAINING PROCEDURE")
#mi_all = train_with_mi()
print("PAIRWISE DIST")
pairwise_distance_plotter()


# Save the MI results
#np.save('mi_all_pytorch.npy', mi_all)
