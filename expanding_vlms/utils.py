import torch
from Otter.src.otter_ai.models.flamingo.modeling_flamingo import FlamingoForConditionalGeneration
from Otter.src.otter_ai.models.flamingo.modeling_flamingo import FlamingoPerceiverResampler
import pickle
from sklearn.model_selection import train_test_split

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

    for key in perceiver_weights_from_new.keys():
        assert torch.equal(perceiver_weights_from_full['perceiver.'+key], perceiver_weights_from_new[key]), f"Weights mismatch for {key}"

def load_pretrained_perceiver_from_otter(model_name="luodian/OTTER-Image-MPT7B", dim=1024):
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
    
    filterLength = len('perceiver.')
    perceiver_weights = {k[filterLength:]: v for k, v in full_state_dict.items() if 'perceiver' in k} # remove 'perceiver.' from the key
    
    # Initialize and load the Perceiver model
    perceiver = FlamingoPerceiverResampler(dim=dim)
    perceiver.load_state_dict(perceiver_weights)
    
    # Validate the weights (could be moved to a separate test function)
    validate_weights(perceiver, full_state_dict)
    
    return perceiver

def load_pretrained_perceiver_from_file(path, dim=1024):
    perceiver = FlamingoPerceiverResampler(dim=dim)
    perceiver.load_state_dict(torch.load(path))
    perceiver.eval()
    for param in perceiver.parameters():
        param.requires_grad = False
    return perceiver

def loadDataFromFile():
    with open('imu_vid_dict_with_seq_vid_embds.pickle', 'rb') as handle:
        imu_vid_dict = pickle.load(handle)
    for k,v in imu_vid_dict.items():
        imu_vid_dict[k]['vid_embeds'] = imu_vid_dict[k]['vid_embeds'].to('cpu')
        torch.cuda.empty_cache()

    train_data, test_data = train_test_split(imu_vid_dict, test_size=0.3, random_state=42)
    return train_data, test_data