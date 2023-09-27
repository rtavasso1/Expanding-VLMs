# IMU-CLIP Encoder Trainer

## Overview

This repository the code used in my paper, [Expanding Frozen Vision-Language Models without
Retraining: Towards Improved Robot Perception](https://arxiv.org/pdf/2308.16493.pdf), the principal result being a methodology of training an encoder of an arbitrary modality such that the embeddings generate are able to be used in conjunction with a frozen, pretrained vision-language model. This is done with a combination of contrastive and supervised learning objectives in combination with a forward pass through the perceiver resampler. Further exposition can be found within the PDF above.

## Requirements 

- Python 3.8
- PyTorch
- OpenAI's CLIP
- Flamingo
- Otter

## Structure

- `__init__.py`: Module initialization file.
- `data.py`: Data preprocessing and loading utilities.
- `model.py`: Defines the architecture of the IMU encoder.
- `train.py`: Contains the training loop for contrastive learning.
- `utils.py`: Various utility functions.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

