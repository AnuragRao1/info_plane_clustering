import numpy as np
from collections import Counter
# we quantize neurons into bins, we then measure the entropy


def entropy_by_quantization(hidden_layers, n_bins,relu=False):
    # I(T_l,X) <= H(T_l) ~ H(Bin(T_l)), measuring clustering
    bins = np.zeros((len(hidden_layers), n_bins))
    if not relu:
        for i in range(bins.shape[0]):
            bins[i,:] = np.linspace(-1,1,n_bins) # sample of bin type for tanh activations
    else:
        for i in range(bins.shape[0]):
            max_neuron = hidden_layers[i].numpy().max()
            bins[i,:] = np.linspace(0,max_neuron,n_bins) # sample of bin type for tanh activations

    H_list = []
    for i in range(len(hidden_layers)):
        hidden_layer = hidden_layers[i]
        T_l = np.digitize(hidden_layer.numpy(), bins[i,:])
        counter_data = [tuple(row) for row in T_l]
        counted_reps = Counter(counter_data)
        joint_probabilities = {k: v / len(counter_data) for k,v in counted_reps.items()}
        
        # for i, (key, prob) in enumerate(joint_probabilities.items()):
        #     print(f"Probability of datapoint {i}: {prob}")
        
        raw_probabilites = np.array(list(joint_probabilities.values()))
        entropy = -np.sum(np.multiply(raw_probabilites, np.log(raw_probabilites))) # USING NATS
        H_list.append(entropy)
    
    return H_list

