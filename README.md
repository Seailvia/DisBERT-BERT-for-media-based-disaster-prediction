# DisBERT-BERT-for-media-based-disaster-prediction

## Dataset: Tweet Analysis - ANN/BERT/CNN/n-gram CNN
The dataset can be found in https://www.kaggle.com/competitions/nlp-getting-started, providing a relatively balanced distribution for model training. This balanced class distribution reduces the likelihood of the model being biased toward the majority class and ensures that both classes contribute meaningfully to the model's performance.

<div align=center>
<img src="https://github.com/Seailvia/DisBERT-BERT-for-media-based-disaster-prediction/blob/main/twitter.png" width = 500>
</div>

A key feature in this dataset is the length of each tweet. We hypothesize that tweet length might influence classification, and therefore, it is important to explore the distribution of tweet lengths in both classes. The length of a tweet is calculated by the number of characters in it.

The average tweet length for real disaster tweets is 108.11 characters, and for fake disaster tweets, it is 95.71 characters. The range of tweet lengths is similar for both categories, with most tweets falling within the 0 to 150 characters range.

## Architecture
In the realm of natural language processing (NLP), BERT has emerged as a revolutionary deep learning model. Its bidirectional architecture, specifically the ability of its self - attention layer to operate in both directions, sets it apart from predecessors like OpenAI GPT. This bidirectionality enables BERT to glean information from both the left and right context of tokens during training, providing a more comprehensive understanding of language structure and semantics.

BERT holds significant advantages in this Twitter - based disaster prediction task. The knowledge above helps in discerning whether words are being used literally (as in the case of a real disaster announcement) or metaphorically. Given the complexity of tweet content where the true meaning might not be immediately obvious, BERT's ability to capture these nuances through its pre-trained knowledge and bidirectional processing makes it well-suited to accurately classify the dataset.

<div align=center>
<img src="https://github.com/Seailvia/DisBERT-BERT-for-media-based-disaster-prediction/blob/main/structure.png" width = 500>
</div>

The fundamental building block of the Transformer encoder layer in BERT is the self-attention mechanism. This mechanism allows the model to weigh the importance of different parts of the input sequence when calculating the representation of each token. In BERT, the self-attention layer is bidirectional, which means it simultaneously considers information from both the left and right context of every token in the sequence. For example, when processing a tweet for disaster prediction, it can understand how the words before and after a particular term interact and contribute to the overall meaning.

## Training
The training process mainly consists of data loading, model definition, optimizer configuration, and iterative training steps. First, the BertTokenizer is used to pre-process the data, converting text into an input format acceptable to the model. The load_data function is employed to load the training and test data and construct data loaders. A custom BERTModel is adopted and deployed on the GPU. The AdamW optimizer is selected, and the learning rate is set. In each training epoch, the train_model function is called forward propagation, loss calculation, back-propagation, and parameter update. The cross-entropy loss is calculated to evaluate the model's performance, and the accuracy is also counted. This structure ensures that the model can gradually learn data features and improve its performance during the training process.

After 15 epochs of training on the training dataset, the loss is observed to be decreased significantly, while the accuracy of the model increases from 70.46% to 95.36%.

#### Training result of 15 epochs

```
Epoch 1/15 - Loss: 0.5877525638848448, Accuracy: 0.7046
Epoch 2/15 - Loss: 0.4346571335682092, Accuracy: 0.8188
Epoch 3/15 - Loss: 0.38173276621598407, Accuracy: 0.8444
Epoch 4/15 - Loss: 0.35530916753763586, Accuracy: 0.858
Epoch 5/15 - Loss: 0.3277079643437657, Accuracy: 0.8724
Epoch 6/15 - Loss: 0.30075490300933394, Accuracy: 0.8868
Epoch 7/15 - Loss: 0.2826704418602081, Accuracy: 0.895
Epoch 8/15 - Loss: 0.25989932877520405, Accuracy: 0.9068
Epoch 9/15 - Loss: 0.2372729818042094, Accuracy: 0.9182
Epoch 10/15 - Loss: 0.21591196547205838, Accuracy: 0.9274
Epoch 11/15 - Loss: 0.20370061207598392, Accuracy: 0.9312
Epoch 12/15 - Loss: 0.18781285039104592, Accuracy: 0.9384
Epoch 13/15 - Loss: 0.17120044254147396, Accuracy: 0.946
Epoch 14/15 - Loss: 0.1663578376292992, Accuracy: 0.9488
Epoch 15/15 - Loss: 0.15068446760312817, Accuracy: 0.9536
```

<div align=center>
<img src="https://github.com/Seailvia/DisBERT-BERT-for-media-based-disaster-prediction/blob/main/training.png" width = 500>
</div>

## Application
The prediction part mainly includes model loading, data preparation, and prediction result generation. First, the BertTokenizer and load_data function are used in the same way to prepare the test data. Then, the trained model parameters are loaded from the specified path, and the model is deployed on a suitable device. The predict function is called to make predictions on the test data. This function performs forward propagation in a no-gradient calculation mode and stores the prediction results in a list. Finally, the prediction results are associated with the IDs of the test data. This structure enables the model to efficiently predict new data and output results that meet the requirements.

<div align=center>
<img src="https://github.com/Seailvia/DisBERT-BERT-for-media-based-disaster-prediction/blob/main/heatmap.png" width = 500>
</div>

The proposed model achieved an accuracy of 90.16% on the test data, demonstrating a substantial improvement over traditional machine-learning methods. As presented above, the confusion matrix heatmap further provides a detailed visualization of the model's performance, enabling a more in-depth analysis of its classification capabilities. This high accuracy not only validates the effectiveness of the model architecture but also highlights its potential for practical applications in the relevant domain.

## Usage
The model has been saved in **model.py**, for data preprocessing, you can use data_processing.py, run:

```
python data_processing.py
```

To train the model with labeled data, run:
```
python train.py
```

The model will be saved to **trained_model.pth**, change the path in **run_test.py** and run the program, then the classification result will be provided.
