# Analysis of Deep Learning Strategies for Wind Energy Forecasting Applications


Wind energy has been recognized as the most promising and economical renewable energy source, attracting increasing attention in recent years. Although wind energy has experienced steady growth worldwide throughout the last decade, the  variability and uncertainty of wind energy pose challenges in the operation  and planning of power systems. Therefore, accurate forecasting is crucial for wind power generation and operation systems, and to propel high levels of wind energy penetration within electricity markets. 

In this work, a comparative framework is proposed where diverse deep learning architectures are implemented in order to address the existing gap and limitations of reported wind power forecasting methodologies. The methodology set forth implements a suite of long short-term memory (LSTM) recurrent neural networks (RNN) models inclusive of standard, bidirectional, stacked, convolutional, and autoencoder architectures. These integrated networks are implemented through an iterative process of varying hypeparameters to better assess their effect, and the overall performance of each architecture, when tackling one-hour to three-hours ahead wind power forecasting. It must be noted that some of the methodologies showcased in this work, such as the implementation of autoencoders for wind power applications, have not been explored in detail in the literature.

The proposed approach is validated through hourly wind power data from the Spanish electricity market, collected between 2014-2020. The proposed comparative error analysis shows that, overall, the models tend to showcase low error variability and better performance when the networks are able to learn in weekly sequences. Moreover, simpler architectures showcased better performance metrics and shorter implementation times. Overall, the model with the best performance forecasting one-hour ahead wind power is the standard LSTM implemented with weekly learning input sequences. In the case of three-hours ahead forecasting, the model with the best overall performance is the bidirectional LSTM implemented with weekly learning input sequences.

### Repository Organization

First, the explaratory data analysis can be found in the similarly named notebook. The repository also includes all the code related  to  the  proposed implementation. The code file is divided according to the steps being predicted. For one-step ahead forecasting, the folder includes vanilla, bidirectional, stacked, and convolutional LSTM integrated architectures. For three-steps ahead forecasting, the folder contains the previously mentioned DL models and autoencoder LSTM networks. The reported error metrics produced under these implementations are saved in the results folder, inclusive of R code to process the results and produce comparative graphs.

### Environment 

The proposed work was implemented in Python 3.8 and R 4.0.0. Some of the python packages and their respective versions are:
1. tensorflow==2.4.0
2. Keras==2.4.3
3. Keras-Preprocessing==1.1.2
4. numpy==1.19.2
5. matplotlib==3.3.2
6. pandas==1.1.3

A full list of python packages and their respective versions is available in the repository, to avoid any clashing depencies.

