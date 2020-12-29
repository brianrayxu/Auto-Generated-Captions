# Auto-Generated-Captions
A classic deep-learning implementation. A neural network based program which automatically generates captions for input photos.

## Combining Vision and Language:

<center> Raspberry Pi Wiring :</center>
<center><img src="./Images/RPi.jpg" width="50%" /></center>

Figure 1. A common image captioning architecture highlighting the role of the RNN [6].

**1. Task**

This project aimed to explore different image captioning architectures and garner an understanding of how the different component networks work together to create the output caption. We explored image captioning models that work by combining an encoder module that computes a feature vector to represent input images and a decoder module that uses these features to generate a caption describing the image content. Typically, the encoder is a Convolutional Neural Net (CNN) and the decoder is a Recurrent Neural Net (RNN). However, some features of transformer models suggested that they may perform as well or better than RNNs. Based on this intuition, we wrote and implemented a transformer in order to replace the decoder module. We compare the new model&#39;s performance to that of the CNN+RNN image captioning architecture. In addition, we experiment with different CNN architectures in order to find the most optimal CNN networks to use.

**2. Dataset**

The Microsoft Common Objects in COntext (MS COCO) dataset is a large-scale dataset for scene understanding containing 328k images and 200k+ captions. The dataset is commonly used to train and benchmark object detection, segmentation, and captioning algorithms. We use the images and captions in this dataset to use as input to our CNN. The specific version of the data set we use is the 2014 version.

**3. Approach**

The specific approach we took involved implementing a basic image captioning model that consisted of an encoder and a decoder. The encoder was used to extract features from an image and the decoder was used to map those features to language to generate captions.

The encoder takes in each image and returns a 300-dimensional embedding feature vector. This is then passed onto the decoder. The decoder consists of an embedding layer that takes in the feature vector, concatenates it with an embedding for the caption vector, and generates a caption. The embedding dimension is 300 and the hidden size for the LSTM is 512. During testing, the LSTM takes in the feature vector for just the input image and generates a standalone caption.

We first fixed the decoder LSTM network and tried different configurations of Resnet and VGG models for the encoder. We then fixed the encoder as a Resnet50 model and tried different transformer architectures for the decoder.

<center> Raspberry Pi Wiring :</center>
<center><img src="./Images/RPi.jpg" width="50%" /></center>

Traditional image captioning architecture using CNN + LSTM. [6]

**Encoder Swap-out Experiments**

We used the built-in Resnet34, Resnet50, Resnet101, VGG13, VGG16, VGG19 models and substituted each one as the encoder in our network. The numbers in each network&#39;s name represent the depth of the network, in layers. The decoder was fixed as an LSTM network in these experiments, and we trained each network for 3 epochs on the COCO dataset.

**Transformer Swap-out Experiments**

We set the encoder to be a Resnet50 model and used two different configurations of our custom Transformer implementation, one with two heads and a batch size of 64 and one with four heads and a batch size of 32. We trained these networks for 5 epochs each but saved checkpoints every epoch so we could compare vs the encoder-swap experiments.

**Loss and Optimizer**

We used Cross-Entropy between the true captions and the output of the decoder as the loss function in this model. For the encoder-swap experiments, The loss is back-propagated using the Adam optimizer, tuning the parameters for both the encoder and decoder networks. For our transformer swap, we fixed the encoder weights when training and just used the loss on the decoder weights. Our current results for CNN + LSTM implementations were training for 3 epochs which took around ~8 hours on the SCC. The transformer implementation was trained for 5 epochs.

**The Transformer**

With the fun of implementing transformers, also came new challenges unique to this architecture.

Padding and tokenizing the captions:

The transformer architecture needs chunks of text data to operate, which we accomplish by divvying up the training text into segments. Since the captions are of varying length, we pad the end with end-of-sentence (\&lt;eos\&gt;) characters, so that the length matches the maximal length for each batch of captions. The captions are then fed into a tokenizer, which generates an embedding vector for each word in each caption. We learned how to integrate this into the dataloader by passing in a custom collate\_fn which padded the captions, stacked the images, generated the caption lengths, and packed it into mini-batches.

Scaled Dot-Attention Mechanism:

Our transformer architecture takes in three matrices: Queries (Q), Keys (K), and Values (V). The attention is calculated as:

<center> Raspberry Pi Wiring :</center>
<center><img src="./Images/RPi.jpg" width="50%" /></center>

as per the original &quot;Attention is All You Need&quot; paper and the transformer we built in class. From the lectures, we knew that all of these matrices should have the same dimensions, and they are all generated by linear projections from the embedding generated for a batch.

<center> Raspberry Pi Wiring :</center>
<center><img src="./Images/RPi.jpg" width="50%" /></center>

Diagram of scaled dot attention mechanism and its role in multi-head attention (source: HW5)

However, in our case, we need to encode information from two sources, and after a literature search, we decided to use an architecture in which the queries Q come in from the CNN channels, and the keys K were generated using the batch of tokenized captions. This meant that we needed to keep careful track of the dimensions in each layer.

Let us assume that our batch size is 64, and after padding, all the captions have a length of 15. The encoder takes in a batch of images and returns 14\*14 feature maps of over num\_channels = 2048 different CNN channels - thus passing an input of (64, 14\*14, 2048) into the decoder. The captions, on the other hand, are fed in as lists of words for each batch, (64, 15). To make sure that this can be compatible with the output of the encoder, we need to make the size of the embedding layer equal to num\_channels. This ensures that after creating the embedding, we can pass in an input of shape (64,15,2048) to the attention modules, where each word in the captions is represented as a 2048-dimensional vector.

We also needed to make sure that the hidden size of the transformer is the same as the embedding dimension. This ensures that the linear projections (2048, 2048) takes in encoder input (64, 196, 2048) and the embedding (64, 15, 2048) from the captions and passes them on with the same dimensions as the queries and keys respectively. The scaled dot-attention mechanism can just multiply them both (switching the last 2 dimensions on keys) to get (64,196, 2048) • (64, 2048, 15) → (64, 196, 15), giving us the attention weights. Multiplying this (with a transpose) with the value matrix (64, 196, 2048) gives us a result with shape (64, 15, 2048) which matches a list of embedded captions! This also makes sure that intermediate layers can pass on this captions object into the attention module of the next layer.

Training vs Testing Modes:

The transformer needs two modes: training and testing. The former takes in a batch of captions and the outputs of the CNN encoder channels and learns the weights to create the best captions. The latter only has to use the CNN input to create a caption based on its training.

In the testing mode, the model passes an image through the encoder to get its embedded features. Those features are then passed to a beam search function (discussed below), the output of which is the predicted caption. The beam search works by searching through many possible captions word by word (the breadth of this search determined by the beam width), and selecting the sentence with the greatest conditional probability based on the feature input.

Parallelizing the Model:

Once our model was constructed and functional, we parallelized it to lend more efficiency to our training routine. However, while this did improve our training time from 7.5hr/epoch to 4hr/epoch, it came with some unexpected consequences. The most significant is that using a DataParallel object combined with partially loading our method from a saved state dictionary made it so that it would save the result of every forward pass to GPU memory. This is a grossly inefficient use of GPU resources and caused us to run into several OutOfMemory errors during training and testing. While we can functionally still use our network, it did set a limit on our batch size and the number of transformer heads we can use to train.

**Evaluation with BLEU**

In order to evaluate the accuracy of our machine-generated captions, we elected to use the BLEU evaluation protocol. BLEU stands for Bi-Lingual Evaluation Understudy and is a widely used metric to evaluate any machine-generated text. We use the built-in BLEU score functions from Natural Language Toolkit (NLTK). These functions take as input some predicted text, the corresponding ground truth, and output a score corresponding to how close of a match the two texts are to one-another. The scores range from 0 to 1, where 1 indicates a perfect match and 0 represents completely different texts.

We consider four variations on BLEU: BLEU-1, BLEU-2, BLEU-3, and BLEU-4. The number corresponds to how many words from each text are compared at once. For example, a BLEU-4 score looks for word matchings for groups of 4 words while a BLEU-2 score would only look for matchings for groups of 2. Generally, the BLEU-4 score is seen as the most valued metric for gauging the success of your image captioning model. However, most studies will present the BLEU-1 through BLEU-4 scores in their results as the BLEU metric as a whole can sometimes be unstable depending on the length and style of text being evaluated. Generally, a human translation would have a BLEU-4 score around in the range of 0.2571 - 0.3468.[8]

**4. Results**

**Encoder Swap-out Experiments**

These two images are sample outputs that we obtained through the Resnet and VGG models. 

<center> Raspberry Pi Wiring :</center>
<center><img src="./Images/RPi.jpg" width="50%" /></center>

<center> Raspberry Pi Wiring :</center>
<center><img src="./Images/RPi.jpg" width="50%" /></center>

When swapping out our CNN networks, we observed that the deeper the model of ResNet or VGG used, the higher our average BLEU scores got during evaluation. However, before evaluation, the sample outputs that we obtained would still make the same mistakes and patterns in their prediction. In particular, we noticed that the CNN + LSTM implementations would occasionally have weird behaviors such as outputting \&lt;unk\&gt; characters or having random or extra periods within generated captions.

**Transformer Swap-out Experiments**

These two images are a couple of sample outputs that we obtained from our transformer implementations. 

<center> Raspberry Pi Wiring :</center>
<center><img src="./Images/RPi.jpg" width="50%" /></center>

<center> Raspberry Pi Wiring :</center>
<center><img src="./Images/RPi.jpg" width="50%" /></center>

We noticed from our sample output captions that the transformer implementation would occasionally make the mistake of mentioning a subject multiple times. We think this is due to the fact that the self-attention component of the transformers was looking at the same subject from multiple areas of attention creating the generated caption to repeat subjects. In general, the successful captions from the transformer were super accurate as to what it was trying to describe and were often more detailed and accurate compared to the sample outputs of the CNN + LSTM implementations.

**BLEU Scores**

In our experiment, we evaluated 7 different architectures for image captioning and we found that our implementation using ResNet101 as our encoder and the LSTMs as our decoder found the highest success. Below is a table of all our experimental results with the average BLEU scores colored according to the quality of caption they represent (greener = higher, redder = lower).

| Architecture | BLEU\_1 | BLEU\_2 | BLEU\_3 | BLEU\_4 | Avg |
| --- | --- | --- | --- | --- | --- |
| Resnet34 + LSTM | 0.3219 | 0.1426 | 0.0105 | 0.0136 | 0.1221 |
| Resnet50 + LSTM | 0.3542 | 0.1727 | 0.0166 | 0.0237 | 0.1418 |
| Resnet101 + LSTM | 0.3731 | 0.1873 | 0.0323 | 0.0433 | 0.1590 |
| VGG13 + LSTM | 0.2966 | 0.1401 | 0.0116 | 0.0174 | 0.1164 |
| VGG16 + LSTM | 0.3093 | 0.1438 | 0.0122 | 0.0175 | 0.1207 |
| VGG19 + LSTM | 0.2537 | 0.0741 | 0.0023 | 0.0037 | 0.0835 |
| Resnet50 + Transformer | 0.3373 | 0.1871 | 0.0285 | 0.0404 | 0.1483 |

All experiments were evaluated with the same procedure. First, ground truth captions were obtained from the COCO 2014 Validation dataset. After obtaining ground truth captions, we would pass these validation images through our encoder/decoder architectures in order to obtain hypothesis captions. Finally, we compare the two using the NLTK BLEU functions to evaluate the accuracy of our predictions. For all architectures, we would pass the validation data in batches of size 64. We would then iterate through the whole dataset, build a cumulative BLEU score and divide it by the number of images we pass through in order to obtain an average score. We do 4 different calculations for the 4 different types of BLEU scores recommended.

Contrary to our hypothesis, we found that the Resnet50+Transformer model did not improve on the Resnet50+LSTM approach. However, we have our doubts about this result as just from looking at numerous outputs from the transformer implementation, we could tell that the way our network was generating captions was way more robust. Subjectively, the captions generated seemed more consistent and specific about the different features in the image and there were fewer grammatical errors. On the other hand, the RNN implementations seemed relatively inconsistent and featured incorrect grammar, unknown character errors, or mis-positioned end-of-sentence characters placed in the middle of captions. Despite this difference in output, some LSTM implementations achieved a BLEU score higher than that of the transformer model. This could be due to the fact that BLEU scores rely solely on token occurrences, and more abstract qualities like grammar or specificity of a generated text do not get factored into this score. Effectively, we suspect the positive attributes of the transformers are being ignored while the negative attributes of the LSTM implementation are being forgiven due to the way BLEU works. In order to conclude this, further studies using different evaluation metrics would need to be conducted.

**6. Future Work**

If we had more time, we would first aim to figure out the exact reason why our Resnet+LSTM implementations were performing better than our transformer implementation. In addition, we also wanted to experiment with more state of the art architectures such as Faster Region-based CNNs that take less time to train and perform better than a traditional or residual CNN.

Due to how we parallelized the network, we ran into some issues regarding GPU usage. We were only able to get the caption outputs for testing one at a time which takes ~24hr to process the entire validation dataset. Increasing the testing efficiency would be of great use for further experimentation. Improving the efficiency of how our network handles GPU resources would allow for multiple batches to be passed through the forward model in eval mode and for the BLEU score to be calculated much more efficiently.


**8. Code Repository**

[GitHub Repo for the project](https://github.com/nihardwivedi/image-captioning)

**9. Ref**** erences**

1. A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, I. Polosukhin. Attention Is All You Need, arXiv.org, June, 2017 [[1](https://arxiv.org/pdf/1706.03762.pdf)]
2. K. Xu, J. L. Ba, R. Kiros, K. Cho, A. Courville, R. Salakhutdinov, R. S. Zemel, Y. Bengio. Show, Attend and Tell: Neural Image Caption Generation with Visual Attention, arXiv.org, Feb, 2015 [[2](https://arxiv.org/pdf/1502.03044.pdf)]
3. Q. Wu, D. Teney, P. Wang, C. Shen, A. Dick, A. Hengel. Visual Question Answering: A Survey of Methods and Datasets, arXiv.org, July, 2016 [[3](https://arxiv.org/pdf/1607.05910.pdf)]
4. M. Malinowski, M. Rohrbach, M. Fritz. Ask Your Neurons: A Neural-based Approach to Answering Questions about Images, arXiv.org, May, 2015 [[4](https://arxiv.org/pdf/1505.01121.pdf)]
5. Parmar, Niki, et al. &quot;Image transformer.&quot; _arXiv preprint arXiv:1802.05751_ (2018).[[6](https://arxiv.org/abs/1802.05751)]
6. Kshirsagar, K. &quot;[Automatic Image Captioning with CNN &amp; RNN&quot;](https://towardsdatascience.com/automatic-image-captioning-with-cnn-rnn-aae3cd442d83).
7. Papineni, Kishore &amp; Roukos, Salim &amp; Ward, Todd &amp; Zhu, Wei Jing. (2002). BLEU: a Method for Automatic Evaluation of Machine Translation. 10.3115/1073083.1073135.
8. Brown, J [Calculating the BLEU Score for Text in Python](https://machinelearningmastery.com/calculate-bleu-score-for-text-python/)
