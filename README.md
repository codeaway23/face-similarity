## Machine Learning Task for Nanonets 

## Problem

The problem statement can be found [here](https://gist.github.com/prats226/d3da93412fef04e3b55b85fed56839e5)

## Usage

Please make sure you have the latest versions of keras, keras-preprocessing, pandas installed. 
Certain functions might not work otherwise. 

1. Download the dataset and add it in 'data' directory in the main directory. A few data samples are included in the directory to understand the folder structure. 

2. To train the model, run the following command from the main directory.
```bash
python3 train.py
```
3. After training, to generate results using a trained model, run the following command from the main directory:
```bash
python3 final.py
```

I couldn't upload the data or my trained model file due to size limits. 

## Approach

1. A multi-input single output model for binary classification is built
2. The model uses inception-v3 network as its convolutional backbone
3. Imagenet weights are transfered to the same to initialise training
4. Dataset used turned into lists of combinations. One list for combinations of images of same people. The other for combinations of images of different people.
5. All data could not be used for training due to harware limitations. Instead, a specific number of combinations are extracted from the dataset to build train, test dataframes.
6. The dataframes are fed into the multi_input_generator function in utils which augments data for multi-input models.
7. This data is then used to train the model.
8. Finally, in the final.py script, a similarity metric is found and a prediction is made on a pair of images drawn from the dataset itself.

## Results

For the following parameters (which are very small numbers but all my PC could handle...)-  
1. nb_epochs = 1
2. train_image_pairs = 4000
3. test_image_pairs = 500

Best validation accuracy achieved - 72%
Worst validation accuracy achieved - 50%

Best model achieved a validation accuracy of 76.4% after 5 epochs of training.

## Limitations and Remedies

1. Can test more approaches in building the model. Try different backbones. (I experimented with ResNet50 and Inception-V3, Inception-V3 came out victorious)
2. Try a deeper network with more trainable parameters.
3. The model doesn't generalise well yet. Performance varies a lot on different PRNG seeds.
4. Should train for more number of epochs and on a bigger subset of the data. 
5. Should try finding ways to screen out combinations better/more efficiently to make the dataset more balanced. 
6. Image augmentation hampers the training. 
7. Similarity metric is based on the sigmoid activation value found in the last layer. Instead, similarity metrics used in image processing can be used like SSIM, earth movers distance, etc. 
