# DisBERT-BERT-for-media-based-disaster-prediction

## Retional
In the realm of natural language processing (NLP), BERT has emerged as a revolutionary deep learning model. Its bidirectional architecture, specifically the ability of its self - attention layer to operate in both directions, sets it apart from predecessors like OpenAI GPT. This bidirectionality enables BERT to glean information from both the left and right context of tokens during training, providing a more comprehensive understanding of language structure and semantics.

BERT holds significant advantages in this Twitter - based disaster prediction task. The knowledge above helps in discerning whether words are being used literally (as in the case of a real disaster announcement) or metaphorically. Given the complexity of tweet content where the true meaning might not be immediately obvious, BERT's ability to capture these nuances through its pre-trained knowledge and bidirectional processing makes it well-suited to accurately classify the dataset.

The fundamental building block of the Transformer encoder layer in BERT is the self-attention mechanism. This mechanism allows the model to weigh the importance of different parts of the input sequence when calculating the representation of each token. In BERT, the self-attention layer is bidirectional, which means it simultaneously considers information from both the left and right context of every token in the sequence. For example, when processing a tweet for disaster prediction, it can understand how the words before and after a particular term interact and contribute to the overall meaning.

<div align=center>
<img src="[https://github.com/Seailvia/Adaptive-ACO-in-Stock-Portfolio-Optimization/blob/main/Structure of Adjusted-ACO.png](https://github.com/Seailvia/DisBERT-BERT-for-media-based-disaster-prediction/blob/main/strcture.png)" width = 500>
</div>

The training process mainly consists of data loading, model definition, optimizer configuration, and iterative training steps. First, the BertTokenizer is used to pre-process the data, converting text into an input format acceptable to the model. The load_data function is employed to load the training and test data and construct data loaders. A custom BERTModel is adopted and deployed on the GPU. The AdamW optimizer is selected, and the learning rate is set. In each training epoch, the train_model function is called forward propagation, loss calculation, back-propagation, and parameter update. The cross-entropy loss is calculated to evaluate the model's performance, and the accuracy is also counted. This structure ensures that the model can gradually learn data features and improve its performance during the training process.

After 15 epochs of training on the training dataset, the loss is observed to be decreased significantly, while the accuracy of the model increases from 70.46% to 95.36%.



The prediction part mainly includes model loading, data preparation, and prediction result generation. First, the BertTokenizer and load_data function are used in the same way to prepare the test data. Then, the trained model parameters are loaded from the specified path, and the model is deployed on a suitable device. The predict function is called to make predictions on the test data. This function performs forward propagation in a no-gradient calculation mode and stores the prediction results in a list. Finally, the prediction results are associated with the IDs of the test data. This structure enables the model to efficiently predict new data and output results that meet the requirements.



The proposed model achieved an accuracy of 90.16% on the test data, demonstrating a substantial improvement over traditional machine-learning methods. As presented above, the confusion matrix heatmap further provides a detailed visualization of the model's performance, enabling a more in-depth analysis of its classification capabilities. This high accuracy not only validates the effectiveness of the model architecture but also highlights its potential for practical applications in the relevant domain.
