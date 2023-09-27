import torch
from .modeling_flamingo import FlamingoForConditionalGeneration
from .modeling_flamingo import FlamingoPerceiverResampler
import pickle
from sklearn.model_selection import train_test_split

def supervisedLoss(imu, pred, device='cpu'):
    labels = []
    for j in range(len(imu)):
        activity = imu[j][0].split('/')[-1][:-4]
        num = label_to_numeric[activity.lower()]
        labels.append(num)
    labels = torch.tensor(labels, dtype=torch.long).to(device)
    lossfcn = torch.nn.CrossEntropyLoss()
    loss = lossfcn(pred,labels)
    return loss

def validate_weights(perceiver, full_state_dict):
    """
    Validate that the weights are correctly loaded.
    
    Parameters:
    - perceiver: FlamingoPerceiverResampler object
    - full_state_dict: dict, full state dictionary of the model
    """
    perceiver_weights_from_full = {k: v for k, v in full_state_dict.items() if 'perceiver' in k}
    new_state_dict = perceiver.state_dict()
    perceiver_weights_from_new = {k: v for k, v in new_state_dict.items() if 'perceiver' in k}

    for key in perceiver_weights_from_full.keys():
        assert torch.equal(perceiver_weights_from_full[key], perceiver_weights_from_new[key]), f"Weights mismatch for {key}"

def get_perceiver_weights(model_name="luodian/OTTER-Image-MPT7B", dim=1024):
    """
    Extracts the weights related to the 'Perceiver' component from a given model.
    
    Parameters:
    - model_name: str, name of the pretrained model
    - dim: int, dimension for the FlamingoPerceiverResampler
    
    Returns:
    - perceiver_weights_from_full: dict, weights of the 'Perceiver' component
    """
    # Load the full model
    full_model = FlamingoForConditionalGeneration.from_pretrained(model_name, device_map="cpu")
    
    # Extract state dict and filter for 'perceiver'
    full_state_dict = full_model.state_dict()
    perceiver_weights = {k: v for k, v in full_state_dict.items() if 'perceiver' in k}
    
    # Initialize and load the Perceiver model
    perceiver = FlamingoPerceiverResampler(dim=dim)
    perceiver.load_state_dict(perceiver_weights)
    
    # Validate the weights (could be moved to a separate test function)
    validate_weights(perceiver, full_state_dict)
    
    return perceiver_weights

def loadDataFromFile():
    with open('imu_vid_dict_with_seq_vid_embds.pickle', 'rb') as handle:
        imu_vid_dict = pickle.load(handle)
    for k,v in imu_vid_dict.items():
        imu_vid_dict[k]['vid_embeds'] = imu_vid_dict[k]['vid_embeds'].to('cpu')
        torch.cuda.empty_cache()

    train_data, test_data = train_test_split(imu_vid_dict, test_size=0.3, random_state=42)
    return train_data, test_data

label_to_numeric = {
                'carrying': 0,
                'checking_time': 1,
                'closing': 2,
                'crouching': 3,
                'entering': 4,
                'exiting': 5,
                'fall': 6,
                'jumping': 7,
                'kicking': 8,
                'loitering': 9,
                'looking_around': 10,
                'opening': 11,
                'picking_up': 12,
                'pointing': 13,
                'pulling': 14,
                'pushing': 15,
                'running': 16,
                'setting_down': 17,
                'standing': 18,
                'talking_on_phone': 19,
                'talking': 20,
                'throwing': 21,
                'transferring_object': 22,
                'using_phone': 23,
                'walking': 24,
                'waving_hand': 25,
                'carrying_light': 26, 
                'sitting_down': 27,
                'standing_up': 28, 
                'pocket_out': 29, 
                'using_pc': 30, 
                'drinking': 31, 
                'pocket_in': 32, 
                'carrying_heavy': 33, 
                'sitting': 34
                }