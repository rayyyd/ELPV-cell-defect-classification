Classification of defective cells from ELPV dataset using a novel method where a reinforcement-learning agent attempts to learn the optimal sequence for data input.

Contents of agent.ipynb:

1. Pre-processing
2. Convolutional Neural Network
3. Deep Q-learning/LSTM agent

CNN -> classify images
DQL/LSTM -> experimental method where the agent tries to find an optimal sequence of images to train the CNN on,
rather than giving the CNN the entire train set at a time to train on.
