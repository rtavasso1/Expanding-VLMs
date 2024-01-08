import torch
from Otter.src.otter_ai.models.flamingo.modeling_flamingo import FlamingoForConditionalGeneration
from Otter.src.otter_ai.models.flamingo.modeling_flamingo import FlamingoPerceiverResampler
import pickle
from sklearn.model_selection import train_test_split
from transformers import CLIPProcessor, CLIPTextModelWithProjection

def compute_label_embeddings():
    label_context = {'pushing': 'a photo of someone pushing something',
                 'kicking': 'a photo of someone kicking something',
                 'talking': 'a photo of someone talking to someone',
                 'waving_hand': 'a photo of someone waving their hand',
                 'opening': 'a photo of someone opening something',
                 'exiting': 'a photo of someone exiting a room',
                 'picking_up': 'a photo of someone picking something up',
                 'standing': 'a photo of someone standing up',
                 'using_phone': 'a photo of someone using their phone',
                 'setting_down': 'a photo of someone setting something down',
                 'walking': 'a photo of someone walking',
                 'looking_around': 'a photo of someone looking around',
                 'jumping': 'a photo of someone jumping',
                 'talking_on_phone': 'a photo of someone talking on the phone',
                 'running': 'a photo of someone running',
                 'crouching': 'a photo of someone crouching',
                 'entering': 'a photo of someone entering',
                 'loitering': 'a photo of someone loitering',
                 'closing': 'a photo of someone closing a door',
                 'carrying': 'a photo of someone carrying something',
                 'pointing': 'a photo of someone pointing at something',
                 'checking_time': 'a photo of someone checking the time',
                 'transferring_object': 'a photo of someone moving something',
                 'throwing': 'a photo of someone throwing something',
                 'fall': 'a photo of someone falling',
                 'pulling': 'a photo of someone pulling something',
                 'drinking': 'a photo of someone drinking something',
                 'sitting': 'a photo of someone sitting',
                 'sitting_down': 'a photo of someone taking a seat',
                 'using_pc': 'a photo of someone using a computer',
                 'pocket_out': 'a photo of someone pulling something out of their pocket ',
                 'standing_up': 'a photo of someone standing up',
                 'pocket_in': 'a photo of someone putting something in their pocket',
                 'carrying_heavy': 'a photo of someone carrying something heavy',
                 'carrying_light': 'a photo of someone carrying something light'}
    
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    textmodel = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-large-patch14")
    
    inputs = processor(text=list(label_context.values()), return_tensors="pt", padding=True)

    # Get text embeddings
    embeds = textmodel(**inputs).text_embeds.half()
    
    context_embeds = {}
    for label, embed in zip(list(label_context.keys()),embeds):
        context_embeds[label] = embed
        
    return context_embeds, embeds

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

independent_encoders_paths = ['../Checkpoints/gzwm75h5/light-sweep-1/epoch90_avg_train_loss_2.464.pt',
 '../Checkpoints/gzwm75h5/noble-sweep-2/epoch90_avg_train_loss_5.541.pt',
 '../Checkpoints/ont0648n/azure-sweep-1/epoch90_avg_train_loss_5.563.pt',
 '../Checkpoints/gzwm75h5/sunny-sweep-4/epoch90_avg_train_loss_2.430.pt',
 '../Checkpoints/gzwm75h5/sunny-sweep-5/epoch90_avg_train_loss_2.685.pt',
 '../Checkpoints/y98l5jvo/ancient-sweep-1/epoch90_avg_train_loss_5.643.pt',
 '../Checkpoints/gzwm75h5/rich-sweep-7/epoch90_avg_train_loss_2.640.pt',
 '../Checkpoints/ont0648n/hardy-sweep-3/epoch90_avg_train_loss_5.576.pt',
 '../Checkpoints/gzwm75h5/worldly-sweep-9/epoch90_avg_train_loss_2.335.pt',
 '../Checkpoints/ont0648n/rose-sweep-4/epoch90_avg_train_loss_5.623.pt',
 '../Checkpoints/gzwm75h5/generous-sweep-11/epoch90_avg_train_loss_2.598.pt',
 '../Checkpoints/gzwm75h5/sage-sweep-12/epoch90_avg_train_loss_2.436.pt',
 '../Checkpoints/gzwm75h5/distinctive-sweep-13/epoch90_avg_train_loss_4.435.pt',
 '../Checkpoints/gzwm75h5/whole-sweep-14/epoch90_avg_train_loss_3.867.pt',
 '../Checkpoints/gzwm75h5/wobbly-sweep-15/epoch90_avg_train_loss_5.065.pt',
 '../Checkpoints/gzwm75h5/floral-sweep-16/epoch90_avg_train_loss_3.090.pt',
 '../Checkpoints/gzwm75h5/atomic-sweep-17/epoch90_avg_train_loss_5.427.pt',
 '../Checkpoints/ont0648n/lyric-sweep-5/epoch90_avg_train_loss_5.590.pt',
 '../Checkpoints/gzwm75h5/twilight-sweep-19/epoch90_avg_train_loss_1.963.pt',
 '../Checkpoints/gzwm75h5/splendid-sweep-20/epoch90_avg_train_loss_2.849.pt',
 '../Checkpoints/gzwm75h5/stellar-sweep-21/epoch90_avg_train_loss_4.830.pt',
 '../Checkpoints/ont0648n/blooming-sweep-6/epoch90_avg_train_loss_5.544.pt',
 '../Checkpoints/gzwm75h5/splendid-sweep-23/epoch90_avg_train_loss_2.177.pt',
 '../Checkpoints/gzwm75h5/earnest-sweep-24/epoch90_avg_train_loss_2.514.pt',
 '../Checkpoints/gzwm75h5/unique-sweep-25/epoch90_avg_train_loss_3.006.pt',
 '../Checkpoints/gzwm75h5/devout-sweep-26/epoch90_avg_train_loss_2.407.pt',
 '../Checkpoints/ont0648n/devout-sweep-7/epoch90_avg_train_loss_5.601.pt',
 '../Checkpoints/gzwm75h5/bumbling-sweep-28/epoch90_avg_train_loss_2.486.pt',
 '../Checkpoints/gzwm75h5/hearty-sweep-29/epoch90_avg_train_loss_5.091.pt',
 '../Checkpoints/gzwm75h5/logical-sweep-30/epoch90_avg_train_loss_2.141.pt',
 '../Checkpoints/gzwm75h5/royal-sweep-31/epoch90_avg_train_loss_5.171.pt',
 '../Checkpoints/gzwm75h5/jolly-sweep-32/epoch90_avg_train_loss_2.558.pt',
 '../Checkpoints/ont0648n/sweepy-sweep-8/epoch90_avg_train_loss_5.576.pt',
 '../Checkpoints/ont0648n/noble-sweep-9/epoch90_avg_train_loss_5.553.pt',
 '../Checkpoints/gzwm75h5/sparkling-sweep-35/epoch90_avg_train_loss_2.288.pt',
 '../Checkpoints/ont0648n/fragrant-sweep-10/epoch90_avg_train_loss_5.615.pt',
 '../Checkpoints/gzwm75h5/expert-sweep-37/epoch90_avg_train_loss_2.247.pt',
 '../Checkpoints/gzwm75h5/vital-sweep-38/epoch90_avg_train_loss_2.102.pt',
 '../Checkpoints/gzwm75h5/fiery-sweep-39/epoch90_avg_train_loss_2.276.pt',
 '../Checkpoints/ont0648n/expert-sweep-11/epoch90_avg_train_loss_5.616.pt',
 '../Checkpoints/gzwm75h5/distinctive-sweep-41/epoch90_avg_train_loss_2.317.pt',
 '../Checkpoints/ont0648n/giddy-sweep-12/epoch90_avg_train_loss_5.600.pt',
 '../Checkpoints/gzwm75h5/chocolate-sweep-43/epoch90_avg_train_loss_2.648.pt',
 '../Checkpoints/gzwm75h5/zesty-sweep-44/epoch90_avg_train_loss_2.519.pt',
 '../Checkpoints/gzwm75h5/likely-sweep-45/epoch90_avg_train_loss_2.477.pt',
 '../Checkpoints/gzwm75h5/iconic-sweep-46/epoch90_avg_train_loss_2.473.pt',
 '../Checkpoints/gzwm75h5/clear-sweep-47/epoch90_avg_train_loss_2.426.pt',
 '../Checkpoints/gzwm75h5/worthy-sweep-48/epoch90_avg_train_loss_2.798.pt',
 '../Checkpoints/gzwm75h5/silvery-sweep-49/epoch90_avg_train_loss_5.467.pt',
 '../Checkpoints/ont0648n/fancy-sweep-13/epoch90_avg_train_loss_5.553.pt',
 '../Checkpoints/gzwm75h5/still-sweep-51/epoch90_avg_train_loss_2.609.pt',
 '../Checkpoints/gzwm75h5/clean-sweep-52/epoch90_avg_train_loss_2.491.pt',
 '../Checkpoints/gzwm75h5/pious-sweep-53/epoch90_avg_train_loss_2.421.pt',
 '../Checkpoints/y98l5jvo/generous-sweep-2/epoch90_avg_train_loss_5.656.pt',
 '../Checkpoints/gzwm75h5/ruby-sweep-55/epoch90_avg_train_loss_1.908.pt',
 '../Checkpoints/ont0648n/helpful-sweep-15/epoch90_avg_train_loss_5.562.pt',
 '../Checkpoints/gzwm75h5/eager-sweep-57/epoch90_avg_train_loss_2.388.pt',
 '../Checkpoints/gzwm75h5/resilient-sweep-58/epoch90_avg_train_loss_2.648.pt',
 '../Checkpoints/gzwm75h5/polished-sweep-59/epoch90_avg_train_loss_2.427.pt',
 '../Checkpoints/gzwm75h5/expert-sweep-60/epoch90_avg_train_loss_5.406.pt',
 '../Checkpoints/gzwm75h5/toasty-sweep-61/epoch90_avg_train_loss_5.346.pt',
 '../Checkpoints/gzwm75h5/sandy-sweep-62/epoch90_avg_train_loss_2.215.pt',
 '../Checkpoints/gzwm75h5/lilac-sweep-63/epoch90_avg_train_loss_5.475.pt',
 '../Checkpoints/gzwm75h5/trim-sweep-64/epoch50_avg_train_loss_5.699.pt']

whitelist = [0,1,3,4,6,8,10,11,12,13,14,15,16,18,19,20,22,23,24,25,27,28,29,30,31,34,36,37,38,40,42,43,44,45,46,47,48,50,51,52,54,56,57,58,59,60,61,62,63]

null = None
false = False
true = True
runs_of_interest = {'vital-sweep-15': {
  "k": {
    "desc": null,
    "value": [
      1,
      5,
      10
    ]
  },
  "T_max": {
    "desc": null,
    "value": 100
  },
  "width": {
    "desc": null,
    "value": 128
  },
  "_wandb": {
    "desc": null,
    "value": {
      "t": {
        "1": [
          1,
          5,
          11,
          49,
          53,
          55,
          71,
          98,
          105
        ],
        "2": [
          1,
          5,
          11,
          49,
          53,
          55,
          71,
          98,
          105
        ],
        "3": [
          2,
          16,
          23,
          37
        ],
        "4": "3.9.18",
        "5": "0.15.12",
        "6": "4.34.1",
        "8": [
          5
        ],
        "13": "linux-x86_64"
      },
      "framework": "huggingface",
      "start_time": 1699940542.208718,
      "cli_version": "0.15.12",
      "is_jupyter_run": false,
      "python_version": "3.9.18",
      "is_kaggle_kernel": false,
      "huggingface_version": "4.34.1"
    }
  },
  "dropout": {
    "desc": null,
    "value": 0.3
  },
  "padding": {
    "desc": null,
    "value": 0
  },
  "root_dirs": {
    "desc": null,
    "value": "[\"../data/acc_watch_clip\", \"../data/acc_phone_clip\", \"../data/gyro_clip\", \"../data/orientation_clip\"]"
  },
  "use_wandb": {
    "desc": null,
    "value": true
  },
  "batch_size": {
    "desc": null,
    "value": 128
  },
  "num_epochs": {
    "desc": null,
    "value": 1000
  },
  "patch_size": {
    "desc": null,
    "value": 64
  },
  "num_workers": {
    "desc": null,
    "value": 4
  },
  "learning_rate": {
    "desc": null,
    "value": 0.0005
  },
  "use_perceiver": {
    "desc": null,
    "value": false
  },
  "checkpoint_dir": {
    "desc": null,
    "value": "../Checkpoints"
  },
  "video_root_dir": {
    "desc": null,
    "value": "../data/video"
  },
  "projected_embeds": {
    "desc": null,
    "value": false
  },
  "num_training_steps": {
    "desc": null,
    "value": 13000
  },
  "virtual_batch_size": {
    "desc": null,
    "value": -1
  },
  "use_supervised_loss": {
    "desc": null,
    "value": true
  },
  "early_stopping_delta": {
    "desc": null,
    "value": 0.01
  },
  "metrics_on_perceiver": {
    "desc": null,
    "value": false
  },
  "use_contrastive_loss": {
    "desc": null,
    "value": true
  },
  "early_stopping_patience": {
    "desc": null,
    "value": 30
  },
  "supervised_on_perceiver": {
    "desc": null,
    "value": false
  },
  "use_perceiver_on_video_only": {
    "desc": null,
    "value": false
  }
},
                    'exalted-sweep-30': {
  "k": {
    "desc": null,
    "value": [
      1,
      5,
      10
    ]
  },
  "T_max": {
    "desc": null,
    "value": 100
  },
  "width": {
    "desc": null,
    "value": 128
  },
  "_wandb": {
    "desc": null,
    "value": {
      "t": {
        "1": [
          1,
          5,
          11,
          49,
          53,
          55,
          71,
          98,
          105
        ],
        "2": [
          1,
          5,
          11,
          49,
          53,
          55,
          71,
          98,
          105
        ],
        "3": [
          2,
          16,
          23,
          37
        ],
        "4": "3.9.18",
        "5": "0.15.12",
        "6": "4.34.1",
        "8": [
          5
        ],
        "13": "linux-x86_64"
      },
      "framework": "huggingface",
      "start_time": 1699955845.496945,
      "cli_version": "0.15.12",
      "is_jupyter_run": false,
      "python_version": "3.9.18",
      "is_kaggle_kernel": false,
      "huggingface_version": "4.34.1"
    }
  },
  "dropout": {
    "desc": null,
    "value": 0.3
  },
  "padding": {
    "desc": null,
    "value": 0
  },
  "root_dirs": {
    "desc": null,
    "value": "[\"../data/acc_watch_clip\", \"../data/acc_phone_clip\", \"../data/gyro_clip\", \"../data/orientation_clip\"]"
  },
  "use_wandb": {
    "desc": null,
    "value": true
  },
  "batch_size": {
    "desc": null,
    "value": 256
  },
  "num_epochs": {
    "desc": null,
    "value": 1000
  },
  "patch_size": {
    "desc": null,
    "value": 64
  },
  "num_workers": {
    "desc": null,
    "value": 4
  },
  "learning_rate": {
    "desc": null,
    "value": 0.0005
  },
  "use_perceiver": {
    "desc": null,
    "value": false
  },
  "checkpoint_dir": {
    "desc": null,
    "value": "../Checkpoints"
  },
  "video_root_dir": {
    "desc": null,
    "value": "../data/video"
  },
  "projected_embeds": {
    "desc": null,
    "value": false
  },
  "num_training_steps": {
    "desc": null,
    "value": 13000
  },
  "virtual_batch_size": {
    "desc": null,
    "value": -1
  },
  "use_supervised_loss": {
    "desc": null,
    "value": false
  },
  "early_stopping_delta": {
    "desc": null,
    "value": 0.01
  },
  "metrics_on_perceiver": {
    "desc": null,
    "value": false
  },
  "use_contrastive_loss": {
    "desc": null,
    "value": true
  },
  "early_stopping_patience": {
    "desc": null,
    "value": 30
  },
  "supervised_on_perceiver": {
    "desc": null,
    "value": false
  },
  "use_perceiver_on_video_only": {
    "desc": null,
    "value": false
  }
},
                    'youthful-sweep-77': {
  "k": {
    "desc": null,
    "value": [
      1,
      5,
      10
    ]
  },
  "T_max": {
    "desc": null,
    "value": 100
  },
  "width": {
    "desc": null,
    "value": 512
  },
  "_wandb": {
    "desc": null,
    "value": {
      "t": {
        "1": [
          1,
          5,
          11,
          49,
          53,
          55,
          71,
          98,
          105
        ],
        "2": [
          1,
          5,
          11,
          49,
          53,
          55,
          71,
          98,
          105
        ],
        "3": [
          2,
          16,
          23,
          37
        ],
        "4": "3.9.18",
        "5": "0.15.12",
        "6": "4.34.1",
        "8": [
          5
        ],
        "13": "linux-x86_64"
      },
      "framework": "huggingface",
      "start_time": 1699724584.917841,
      "cli_version": "0.15.12",
      "is_jupyter_run": false,
      "python_version": "3.9.18",
      "is_kaggle_kernel": false,
      "huggingface_version": "4.34.1"
    }
  },
  "dropout": {
    "desc": null,
    "value": 0.3
  },
  "padding": {
    "desc": null,
    "value": 0
  },
  "root_dirs": {
    "desc": null,
    "value": "[\"../data/acc_watch_clip\", \"../data/acc_phone_clip\", \"../data/gyro_clip\", \"../data/orientation_clip\"]"
  },
  "use_wandb": {
    "desc": null,
    "value": true
  },
  "batch_size": {
    "desc": null,
    "value": 1024
  },
  "num_epochs": {
    "desc": null,
    "value": 1000
  },
  "patch_size": {
    "desc": null,
    "value": 16
  },
  "num_workers": {
    "desc": null,
    "value": 4
  },
  "learning_rate": {
    "desc": null,
    "value": 0.0005
  },
  "use_perceiver": {
    "desc": null,
    "value": false
  },
  "checkpoint_dir": {
    "desc": null,
    "value": "../Checkpoints"
  },
  "video_root_dir": {
    "desc": null,
    "value": "../data/video"
  },
  "projected_embeds": {
    "desc": null,
    "value": false
  },
  "num_training_steps": {
    "desc": null,
    "value": 13000
  },
  "use_supervised_loss": {
    "desc": null,
    "value": false
  },
  "early_stopping_delta": {
    "desc": null,
    "value": 0.01
  },
  "metrics_on_perceiver": {
    "desc": null,
    "value": false
  },
  "use_contrastive_loss": {
    "desc": null,
    "value": true
  },
  "early_stopping_patience": {
    "desc": null,
    "value": 30
  },
  "supervised_on_perceiver": {
    "desc": null,
    "value": false
  },
  "use_perceiver_on_video_only": {
    "desc": null,
    "value": false
  }
},
                    'lemon-sweep-1': {
  "k": {
    "desc": null,
    "value": [
      1,
      5,
      10
    ]
  },
  "T_max": {
    "desc": null,
    "value": 100
  },
  "width": {
    "desc": null,
    "value": 128
  },
  "_wandb": {
    "desc": null,
    "value": {
      "t": {
        "1": [
          1,
          5,
          11,
          49,
          53,
          55,
          71,
          98,
          105
        ],
        "2": [
          1,
          5,
          11,
          49,
          53,
          55,
          71,
          98,
          105
        ],
        "3": [
          2,
          16,
          23,
          37
        ],
        "4": "3.9.18",
        "5": "0.15.12",
        "6": "4.34.1",
        "8": [
          5
        ],
        "13": "linux-x86_64"
      },
      "framework": "huggingface",
      "start_time": 1699817007.883391,
      "cli_version": "0.15.12",
      "is_jupyter_run": false,
      "python_version": "3.9.18",
      "is_kaggle_kernel": false,
      "huggingface_version": "4.34.1"
    }
  },
  "dropout": {
    "desc": null,
    "value": 0.3
  },
  "padding": {
    "desc": null,
    "value": 0
  },
  "root_dirs": {
    "desc": null,
    "value": "[\"../data/acc_watch_clip\", \"../data/acc_phone_clip\", \"../data/gyro_clip\", \"../data/orientation_clip\"]"
  },
  "use_wandb": {
    "desc": null,
    "value": true
  },
  "batch_size": {
    "desc": null,
    "value": 2048
  },
  "num_epochs": {
    "desc": null,
    "value": 1000
  },
  "patch_size": {
    "desc": null,
    "value": 16
  },
  "num_workers": {
    "desc": null,
    "value": 4
  },
  "learning_rate": {
    "desc": null,
    "value": 0.0005
  },
  "use_perceiver": {
    "desc": null,
    "value": false
  },
  "checkpoint_dir": {
    "desc": null,
    "value": "../Checkpoints"
  },
  "video_root_dir": {
    "desc": null,
    "value": "../data/video"
  },
  "projected_embeds": {
    "desc": null,
    "value": false
  },
  "num_training_steps": {
    "desc": null,
    "value": 13000
  },
  "use_supervised_loss": {
    "desc": null,
    "value": true
  },
  "early_stopping_delta": {
    "desc": null,
    "value": 0.01
  },
  "metrics_on_perceiver": {
    "desc": null,
    "value": false
  },
  "use_contrastive_loss": {
    "desc": null,
    "value": true
  },
  "early_stopping_patience": {
    "desc": null,
    "value": 30
  },
  "supervised_on_perceiver": {
    "desc": null,
    "value": false
  },
  "use_perceiver_on_video_only": {
    "desc": null,
    "value": false
  }
},
                   'classic-sweep-36': {
  "T_max": {
    "desc": null,
    "value": 100
  },
  "width": {
    "desc": null,
    "value": 128
  },
  "_wandb": {
    "desc": null,
    "value": {
      "t": {
        "1": [
          1,
          5,
          11,
          49,
          53,
          55,
          71,
          98,
          105
        ],
        "2": [
          1,
          5,
          11,
          49,
          53,
          55,
          71,
          98,
          105
        ],
        "3": [
          2,
          16,
          23,
          37
        ],
        "4": "3.9.18",
        "5": "0.15.12",
        "6": "4.34.1",
        "8": [
          5
        ],
        "13": "linux-x86_64"
      },
      "framework": "huggingface",
      "start_time": 1699276821.92238,
      "cli_version": "0.15.12",
      "is_jupyter_run": false,
      "python_version": "3.9.18",
      "is_kaggle_kernel": false,
      "huggingface_version": "4.34.1"
    }
  },
  "dropout": {
    "desc": null,
    "value": 0.19262949743346724
  },
  "padding": {
    "desc": null,
    "value": "same"
  },
  "root_dirs": {
    "desc": null,
    "value": "[\"../data/acc_watch_clip\", \"../data/acc_phone_clip\", \"../data/gyro_clip\", \"../data/orientation_clip\"]"
  },
  "use_wandb": {
    "desc": null,
    "value": true
  },
  "batch_size": {
    "desc": null,
    "value": 2048
  },
  "num_epochs": {
    "desc": null,
    "value": 1000
  },
  "patch_size": {
    "desc": null,
    "value": 64
  },
  "num_workers": {
    "desc": null,
    "value": 4
  },
  "n_components": {
    "desc": null,
    "value": 16
  },
  "learning_rate": {
    "desc": null,
    "value": 0.0009235592786361728
  },
  "use_perceiver": {
    "desc": null,
    "value": false
  },
  "checkpoint_dir": {
    "desc": null,
    "value": "../Checkpoints"
  },
  "video_root_dir": {
    "desc": null,
    "value": "../data/video"
  },
  "num_training_steps": {
    "desc": null,
    "value": 13000
  },
  "use_supervised_loss": {
    "desc": null,
    "value": false
  },
  "early_stopping_delta": {
    "desc": null,
    "value": 0.01
  },
  "metrics_on_perceiver": {
    "desc": null,
    "value": false
  },
  "use_contrastive_loss": {
    "desc": null,
    "value": true
  },
  "early_stopping_patience": {
    "desc": null,
    "value": 30
  },
  "supervised_on_perceiver": {
    "desc": null,
    "value": false
  },
  "use_perceiver_on_video_only": {
    "desc": null,
    "value": false
  }
},
                   'good-sweep-1': {
  "k": {
    "desc": null,
    "value": [
      1,
      5,
      10
    ]
  },
  "T_max": {
    "desc": null,
    "value": 100
  },
  "width": {
    "desc": null,
    "value": 512
  },
  "_wandb": {
    "desc": null,
    "value": {
      "t": {
        "1": [
          1,
          5,
          11,
          49,
          53,
          55,
          71,
          98,
          105
        ],
        "2": [
          1,
          5,
          11,
          49,
          53,
          55,
          71,
          98,
          105
        ],
        "3": [
          16,
          23,
          37
        ],
        "4": "3.9.18",
        "5": "0.16.0",
        "6": "4.35.0",
        "8": [
          5
        ],
        "13": "linux-x86_64"
      },
      "framework": "huggingface",
      "start_time": 1700035629.7964,
      "cli_version": "0.16.0",
      "is_jupyter_run": false,
      "python_version": "3.9.18",
      "is_kaggle_kernel": false,
      "huggingface_version": "4.35.0"
    }
  },
  "dropout": {
    "desc": null,
    "value": 0.3
  },
  "padding": {
    "desc": null,
    "value": 0
  },
  "root_dirs": {
    "desc": null,
    "value": "[\"../data/acc_watch_clip\", \"../data/acc_phone_clip\", \"../data/gyro_clip\", \"../data/orientation_clip\"]"
  },
  "use_wandb": {
    "desc": null,
    "value": true
  },
  "batch_size": {
    "desc": null,
    "value": 512
  },
  "num_epochs": {
    "desc": null,
    "value": 1000
  },
  "patch_size": {
    "desc": null,
    "value": 16
  },
  "num_workers": {
    "desc": null,
    "value": 4
  },
  "learning_rate": {
    "desc": null,
    "value": 0.0005
  },
  "use_perceiver": {
    "desc": null,
    "value": false
  },
  "checkpoint_dir": {
    "desc": null,
    "value": "../Checkpoints"
  },
  "video_root_dir": {
    "desc": null,
    "value": "../data/video"
  },
  "projected_embeds": {
    "desc": null,
    "value": false
  },
  "num_training_steps": {
    "desc": null,
    "value": 13000
  },
  "virtual_batch_size": {
    "desc": null,
    "value": -1
  },
  "use_supervised_loss": {
    "desc": null,
    "value": true
  },
  "contrast_on_sequence": {
    "desc": null,
    "value": true
  },
  "early_stopping_delta": {
    "desc": null,
    "value": 0.01
  },
  "metrics_on_perceiver": {
    "desc": null,
    "value": false
  },
  "use_contrastive_loss": {
    "desc": null,
    "value": true
  },
  "early_stopping_patience": {
    "desc": null,
    "value": 3000
  },
  "supervised_on_perceiver": {
    "desc": null,
    "value": false
  },
  "use_perceiver_on_video_only": {
    "desc": null,
    "value": false
  }
},
                   'peachy-sweep-1': {
  "k": {
    "desc": null,
    "value": [
      1,
      5,
      10
    ]
  },
  "T_max": {
    "desc": null,
    "value": 100
  },
  "width": {
    "desc": null,
    "value": 1024
  },
  "_wandb": {
    "desc": null,
    "value": {
      "t": {
        "1": [
          1,
          5,
          11,
          49,
          53,
          55,
          71,
          98,
          105
        ],
        "2": [
          1,
          5,
          11,
          49,
          53,
          55,
          71,
          98,
          105
        ],
        "3": [
          16,
          23,
          37
        ],
        "4": "3.9.18",
        "5": "0.16.0",
        "6": "4.35.0",
        "8": [
          5
        ],
        "13": "linux-x86_64"
      },
      "framework": "huggingface",
      "start_time": 1700108462.111223,
      "cli_version": "0.16.0",
      "is_jupyter_run": false,
      "python_version": "3.9.18",
      "is_kaggle_kernel": false,
      "huggingface_version": "4.35.0"
    }
  },
  "dropout": {
    "desc": null,
    "value": 0.3
  },
  "padding": {
    "desc": null,
    "value": 0
  },
  "root_dirs": {
    "desc": null,
    "value": "[\"../data/acc_watch_clip\", \"../data/acc_phone_clip\", \"../data/gyro_clip\", \"../data/orientation_clip\"]"
  },
  "use_wandb": {
    "desc": null,
    "value": true
  },
  "batch_size": {
    "desc": null,
    "value": 256
  },
  "num_epochs": {
    "desc": null,
    "value": 1000
  },
  "patch_size": {
    "desc": null,
    "value": 16
  },
  "num_workers": {
    "desc": null,
    "value": 4
  },
  "learning_rate": {
    "desc": null,
    "value": 0.0005
  },
  "use_perceiver": {
    "desc": null,
    "value": false
  },
  "checkpoint_dir": {
    "desc": null,
    "value": "../Checkpoints"
  },
  "video_root_dir": {
    "desc": null,
    "value": "../data/video"
  },
  "projected_embeds": {
    "desc": null,
    "value": false
  },
  "num_training_steps": {
    "desc": null,
    "value": 13000
  },
  "virtual_batch_size": {
    "desc": null,
    "value": -1
  },
  "use_supervised_loss": {
    "desc": null,
    "value": true
  },
  "contrast_on_sequence": {
    "desc": null,
    "value": true
  },
  "early_stopping_delta": {
    "desc": null,
    "value": 0.01
  },
  "metrics_on_perceiver": {
    "desc": null,
    "value": false
  },
  "use_contrastive_loss": {
    "desc": null,
    "value": true
  },
  "early_stopping_patience": {
    "desc": null,
    "value": 3000
  },
  "supervised_on_perceiver": {
    "desc": null,
    "value": false
  },
  "use_perceiver_on_video_only": {
    "desc": null,
    "value": false
  }
}}