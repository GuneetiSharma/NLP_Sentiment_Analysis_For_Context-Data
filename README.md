# NLP_Sentiment_Analysis_For_Context-Sensitive-Data

Sentiment analysis is an approach in natural language processing (NLP) that identifies
the emotional tone behind the body of text. However, the challenge is to extract a
single sentiment from large context-sensitive text. This study presents a new hybrid
approach to predict text-level sentiments based on generating a context-rich text
feature vector by combining single words and word sequences to form a multi-
dimensional word embedding structure using the Word2Vec model. The proposed
approach improves on the traditional bag of words and TF-IDF models by efficiently
detecting sentiment expressions in large textual phrases that contain various contexts
of conversations.

Install R Studio

Click the following link: https://www.rstudio.com/products/rstudio/download/
Read all directions and make sure you install R 3.0.1+ software before installing R Studio.
Once you have done that navigate to the “Installers for Supported Platforms” and click the
installer for you operating system. 

Datasets Used
1. Movies review which we have tagged them negative, positive and neutral using
the Rule based algorithm.
2. Individual messages collected from social platforms which we have tagged
negative and positive respectively.
3. Twitter dataset for an airline company which we have tagged them negative,
positive and neutral.

Helpful Definitions/Tips
• Some of the important packages we used to complete our project implementation (to
be in lowercase with the exception of DBI and RWeka): tidyverse, tidytext, DBI, dplyr,
Qdap, tm, Rweka, Sentimentr, Devtools, ggplot2, magrittr
• In R, you have to set a working directory where you will be saving your work so R can
access the files on your local drive. The easiest way to do this is by clicking
Session → Set Working Directory
