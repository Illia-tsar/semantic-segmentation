<h1 align="center">DESCRIPTION</h1>

  The repository provides a solution to this [Kaggle competition](https://www.kaggle.com/c/airbus-ship-detection "Airbus Ship Detection Challenge"). Trained model has achieved 0.70690 score on Kaggle.
  
  Project’s repository has the following folder hierarchy:
  * src directory contains all the source code for training, testing and analyzing data;
  * data directory holds all the project’s data split into train/test set and located in corresponding folders;
  * models directory has a nested directory history which holds dumps of model training results. It also contains all saved models for future retrieval;
  * submission directory was designed to hold submission .csv files produced by inference script.
  
  The presented solution is based on U-Net architecture because it provides a relatively simple way to solve the competition challenge using image segmentation. It is composed of _decoder, encoder parts_ and _bottleneck_ 
  as well as _skip connections_. Apart from this, two last blocks of encoder part have **dropout switched on** for regularization with dropout probability 0.3. The code implementing U-Net architecture is located in _model.py_ 
  file under _src_ directory. Also, CNN’s structure is  shown on figure 1.
  <p align="center">
	<img src="https://drive.google.com/uc?export=view&id=1-hC5ioy_KJdauEFMqwyN_EEfgvdDYdXz">
  </p>
  <p align="center"><em>Fig. 1</em></p>
  
  ## Choice of data preprocessing:
  **EDA** has clearly shown that competition’s data is enormously unbalanced. If we take into account only those images which contain ships, the ratio of mask pixels to the total number of pixels is approximately 1:1000. In case 
  no-ships images are also taken into account the gap becomes greater(approximately 1:10000). For this reason, it was decided to drop all no-ships images and, hence, make training data more balanced. This way we also speed up the
  learning process.
  
  Apart from this, it was also suggested to implement image _augmentation_ step. Because training data is too big to fit into memory, this step was implemented in the form of generator as well as _image loading_ step. It is worth 
  saying that images were normalized before feeding into model.
  
  ## Choice of learning strategy:
  The suggested solution was carried out in three stages: training Neural Network on 192x192 images for 10 epochs, then 384x384 images for 5 epochs and, finally, 768x768 images for 2 epochs. It became possible because modern 
  convolutional neural networks support input images of arbitrary resolution. To decrease the training time, one can start training on lower resolution images first and continue training on higher resolution images for fewer 
  epochs. In addition, a model pretrained on lower resolution images first generalizes better since a pixel information is less available and high order features tend to be used. .The number of **STEPS_PER_EPOCH** was set manually 
  due to lack of _computational capability_: 192x192 images – 300, 384x384 – 600, 768x768 - 800.
  
  ## Choice of loss function:
  This solution proposes using _categorical cross-entropy loss_ function. The main motivation for this is that _cross-entropy_ is a standard choice for NNs and it shows pretty good performance. Because our model deals with sparse 
  masks, which are mostly filled with zeros, _sparse categorical cross-entropy loss_ function will be a pretty reasonable choice.
  
  ## Evaluation metrics and callables:
  To estimate model’s performance two metrics were introduced: **Dice Score** and **IoU**. They are implemented in _training.py_ file which is located under _src_ directory. Also, for model saving, learning rate decay and 
  regularization(early stopping) _callables_ were used.
  
  ## Final Notes:
  The _inference.py_ file which is located under _src_ directory implements functionality to conveniently get masks for new unseen data and write them into csv file.
