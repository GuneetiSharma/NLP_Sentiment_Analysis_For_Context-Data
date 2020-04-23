#####1 Data importing and pre-processing ####

df_raw <- read.csv(choose.files())

install.packages("magrittr") # package installations are only needed the first time you use it
install.packages("dplyr")    # alternative installation of the %>%
library(magrittr) # needs to be run every time you start R and want to use %>%
library(dplyr)    # alternatively, this also loads %>%

View (df_raw)
df <- df_raw %>%
  select(text, score_chr)

length(which(!complete.cases(df))) # allows to check the missing texts. which function returns index of cases of missing text which is true

df <- df %>%
  mutate(text = as.character(text)) #change text column into character, because read.csv turns all char columns into factor

mutate(df,text = as.character(text))

prop.table(table(df$score_chr)) #looking at the proportion of the class variable

df <- df %>%
  mutate(textLength = nchar(text)) #add a column using with length of text using nchar()

df <- df_raw %>%
  select(text, score_chr) %>%
  mutate(text = as.character(text)) %>%
  mutate(textLength = nchar(text))

summary(df$textLength)

remove.packages("ggplot2") # Unisntall ggplot
install.packages("ggplot2") # Install it again
library(ggplot2) # Load the librarie (you have to do this one on each new session)

ggplot(df, aes(x = textLength, fill = score_chr)) +
  theme_bw() +
  geom_histogram(binwidth = 5) +
  labs(title = "Distribution of Text Lengths with Sentiment Classes") # create a histogram of the textLength column


#There seems to be no distinction between sentiments classes through text length

#### 2 Data Processing####

#### 2.1 Text Processing####
df_clean <- iconv(df$text, from = 'utf8', to = 'latin1')
df_clean = gsub("&amp", "", df_clean)
df_clean = gsub("(RT|via)((?:\\b\\W*@\\w+)+)", " ", df_clean)
df_clean = gsub("@\\w+", " ", df_clean)
df_clean = gsub("http.+", "", df_clean)
df_clean = gsub("[#!\\.?:;\\$\\/|(\\.\\.\\.)]", " ", df_clean)
df_clean = gsub("[[:digit:]]", " ", df_clean)
df_clean = gsub("[ \t]{2,}", " ", df_clean)
df_clean = gsub("^\\s+|\\s+$", " ", df_clean)
df_f=clean = gsub("\\n", "", df_clean)
df$text[indexes[3]]
head(df_clean, 10)
df_clean[4]

clean_df <- df %>%
  select(text, score_chr) %>%
  mutate(text = df_clean)

#### 2.2 Data Partitioning####df$text[3]

library(quanteda)
df_clean[897:910]

head(df_clean)
head(df$text)

set.seed(2)
library(caret)
#create indexes for training data while preserving the class distribution of the original data
indexes <- createDataPartition(df$score_chr, times = 1,
                               p = 0.6, list = F) 
train <- clean_df[indexes,]
test <- clean_df[-indexes,]

prop.table(table(train$score_chr))
prop.table(table(test$score_chr)) #check to see if the distributions are the same


#### 2.3 Training Tokenization ####

train_token <- tokens(train$text, what = 'word',
                      remove_numbers = T, remove_punct = T,
                      remove_symbols = T, remove_hyphens = T) #tokenize into bag of words

train_token <- tokens_tolower(train_token) #lowercase everything, tokens are kept with this function
train_token[3]

train_token <- tokens_select(train_token, stopwords(), selection = 'remove') #remove stopwords
train_token[1:10]

train_token <- tokens_wordstem(train_token, language = 'english')#stemming using porter's
train_token[1:10]



#### 2.4 Creating a document term matrix (DTM) for training set####
train_token_dfm <- dfm(train_token, tolower = F) #create a DTM

#### 2.5 TF-IDF ####
train_token_dfm_tfidf <- dfm_tfidf(train_token_dfm, scheme_tf = 'prop') #parsing prop while use term frequency instead of count.


#####2.6 Tokenizing and Creating DTM for Testing Dataset####

test_token <- tokens(test$text, what = 'word',
                     remove_numbers = T, remove_punct = T,
                     remove_symbols = T, remove_hyphens = T)

test_token <- tokens_tolower(test_token)
head(test_token)

test_token <- tokens_select(test_token, stopwords(), selection = 'remove')
head(test_token)

test_token <- tokens_wordstem(test_token, language = 'english')
head(test_token)

test_token_dfm <- dfm(test_token, tolower = F)
test_token_dfm[1:20,1:10]


test_token_dfm <- dfm_select(test_token_dfm, train_token_dfm, selection = 'keep') #only keeping similar terms in training set
test_token_dfm[1:20,1:10]

dim(test_token_dfm)

#### 2.7 Applying TF-IDF to Testing dataset####
library(quanteda)

test_token_tf <- dfm_weight(test_token_dfm, scheme = 'prop')
View(test_token_tf)

train_idf <- docfreq(train_token_dfm, scheme = 'inverse')
View(train_idf)

tf_idf <- function(x, idf){
  x * idf
}

dim(test_token_tf)

test_token_tfidf <- apply(test_token_tf, 1,tf_idf, idf = train_idf) #applying tf-idf 
View(test_token_tfidf)

test_token_tfidf <- t(test_token_tfidf) #tranponse to reverse into row as documents and columns as word tokens
View(test_token_tfidf)
dim(test_token_tfidf)

which(!complete.cases(test_token_tfidf))
test_token_tfidf[is.na(test_token_tfidf)] <- 0.0 #turn missing texts into 0




#### 2.8 SVD ####

library(irlba)
View(train_token_dfm_tfidf)


train_irlba <- irlba(t(train_token_dfm_tfidf), nv = 300, maxit = 600) #using SVD to reduce dimension into 300x300 matrix
View(train_irlba$u)

train_svd <- data.frame(sentiment = train$score_chr, train_irlba$v)#create a new train data with the new 300 features
View(train_svd)

#####2.9 Test data SVD projection####

#project the SVD to the train data
sigma_inverse <- 1/train_irlba$d
u_tranpose <- t(train_irlba$u)
test_svd <- t(sigma_inverse * u_tranpose %*% t(test_token_tfidf))
dim(test_svd)  


test_svd <- data.frame(sentiment = test$score_chr, test_svd)

dim(train_svd)
dim(test_svd)


##### 2.10 Cross-valiadation#### 
library(caret)         
set.seed(12)
cv.folds <- createMultiFolds(train$score_chr, k = 10, times = 3)
cv.cntrl <- trainControl( method = 'repeatedcv', number = 10,
                          repeats = 3, index = cv.folds)

#### 2.11 Setting Up Multicores for Model Training####

library(iterators)
library(parallel)
library(foreach)
library(doParallel)

cl <- makePSOCKcluster(3) #using multicores
registerDoParallel(cl)#register

View(train_svd)


##### 2.12 Model Training (Random Forest)####

install.packages('e1071', dependencies=TRUE)


rf_model <- train(form = sentiment ~ ., data = train_svd, method = "rf",
                  trControl = cv.cntrl, tuneLength = 7)#method 1 


rf_model <- train(x = train_svd[,2:301], y = train$score_chr ,method = "rf",
                 trControl = cv.cntrl, tuneLength = 7)#method 2 

stopCluster(cl) #stop multicore running mustrun

rf_model

modelFile <- tempfile("rf_model", fileext = ".rds")

saveRDS(rf_model,modelFile)

##### 2.13 Testing the Model####

#prediction
rf_model_uni <- readRDS(modelFile)
rf_model_uni2 <- readRDS(modelFile2)

system.time(pred_uni <- predict(rf_model_uni, test_svd))

system.time(pred_uni2 <- predict(rf_model_uni2, test_svd))

##### 2.14 Model Assessment ####
confusionMatrix(pred_uni, test_svd$sentiment)

confusionMatrix(pred_uni2, test_svd$sentiment)

length(pred_uni)
length(pred_uni2)
length(test_svd$sentiment)

#####3 Uni-gram and Bi-gram####

#Training Tokenization 
train_token_12gram <- tokens(train_token, ngrams = 1:2)

#Creating a DTM for Training set
train_token_dfm_12gram <- dfm(train_token_12gram)
dim(train_token_dfm_12gram)

#TF-IDF
train_token_dfm_tfidf_12gram <- dfm_tfidf(train_token_dfm_12gram, scheme_tf = 'prop')
train_token_dfm_tfidf_12gram[1:10,1:11]


#Tokenization and Creating DTM for Testing Dataset
test_token_12gram <- tokens(test_token, ngrams = 1:2)
test_token_dfm_12gram <- dfm(test_token_12gram)
dim(test_token_dfm_12gram)

test_token_dfm_12gram <- dfm_select(test_token_dfm_12gram, train_token_dfm_12gram, selection = 'keep')
dim(test_token_dfm_12gram)

#Apply TF-IDF to Testing Dataset
test_token_12gram_tf <- dfm_weight(test_token_dfm_12gram, scheme = 'prop')

train_docfreq_12gram <- docfreq(train_token_dfm_12gram, scheme = 'inverse')
train_docfreq_12gram

test_token_12gram_tfidf <- apply(test_token_12gram_tf, 1, tf_idf, idf = train_docfreq_12gram)
test_token_12gram_tfidf <- t(test_token_12gram_tfidf)
test_token_12gram_tfidf 

which(!complete.cases(test_token_12gram_tfidf))
test_token_12gram_tfidf[is.na(test_token_12gram_tfidf)] <- 0.0


#SVD 
system.time(train_irlba_12gram <- irlba(t(train_token_dfm_tfidf_12gram), nv = 300, maxit = 600))


#Testing DTM projection
sigma_inverse_12gram <- 1/train_irlba_12gram$d
u_tranpose_12gram <- t(train_irlba_12gram$u)
test_svd_12gram <- t(sigma_inverse_12gram * u_tranpose_12gram %*% t(test_token_12gram_tfidf))
dim(test_svd_12gram)

test_svd_12gram <- data.frame(sentiment = test$score_chr, test_svd_12gram)

#Model Training
registerDoParallel(cl)
rf_model_12gram <- train(sentiment ~ ., data = train_irlba_12gram, method = 'rf',
                         trControl = cv.cntrl, tuneLength = 7)
stopCluster(cl)   

#Testing the model
pred_12gram <- predict(rf_model_12gram, newdata = test_svd_12gram)

#Model Assessment
confusionMatrix(pred_12gram, test_svd_12gram$sentiment)

####4 Undersampling####

#First method
df_under_neg <- clean_df %>%
  filter(score_chr == 'negative')%>%
  sample_n(size = 2363)

df_under_neu <- clean_df %>%
  filter(score_chr == 'neutral') %>%
  sample_n(size = 2363)

df_under_pos <- clean_df %>%
  filter(score_chr == 'positive')

df_under <- bind_rows(df_under_neg, df_under_neu, df_under_pos)
prop.table(table(df_under$score_chr))

#Second method
set.seed(123)
df_under <-downSample(clean_df, clean_df$score_chr)
prop.table(table(df_under$score_chr))


set.seed(12)

indexes_under <- createDataPartition(df_under$score_chr, times = 1,
                                     p = .7, list = F) 

train_under <- df_under[indexes_under,] #set training and testing set
test_under <- df_under[-indexes_under,]

prop.table(table(train_under$score_chr))
prop.table(table(test_under$score_chr))

#Training Tokenization 
train_under_token <- tokens(train_under$text, what = 'word',
                            remove_numbers = T, remove_punct = T,
                            remove_symbols = T, remove_hyphens = T)
head(train_under_token)

train_under_token <- tokens_tolower(train_under_token)
head(train_under_token)

train_under_token <- tokens_select(train_under_token, stopwords(), selection = 'remove')
head(train_under_token)

train_under_token <- tokens_wordstem(train_under_token, language = 'english')
head(train_under_token)


#TF-IDF
train_under_dfm <- dfm(train_under_token, tolower = F)
train_under_dfm_tfidf <- dfm_tfidf(train_under_dfm, scheme_tf = 'prop')
train_under_dfm_tfidf[1,1:20]


#Tokenizing and Creating a DTM for Testing Set
test_under_token <- tokens(test_under$text, what = "word",
                           remove_numbers = T, remove_punct = T,
                           remove_symbols = T, remove_hyphens = T)


test_under_token <- tokens_tolower(test_under_token)
head(test_under_token)

test_under_token <- tokens_select(test_under_token, stopwords(), selection = 'remove')
head(test_under_token)

test_under_token <- tokens_wordstem(test_under_token, language = 'english')
head(test_under_token)

#Apply TF-IDF to Testing Set
test_under_dfm <- dfm(test_under_token, tolower = F)

test_under_dfm <- dfm_select(test_under_dfm, train_under_dfm, selection = 'keep')

test_under_tf <- dfm_weight(test_under_dfm, scheme = 'prop')

train_under_docfreq <- docfreq(train_under_dfm, scheme = 'inverse')

test_under_token_tfidf <- apply(test_under_dfm, 1, tf_idf, idf = train_under_docfreq)

which(!complete.cases(test_under_tfidf))
test_under_tfidf[is.na(test_under_tfidf)] <- 0.0

#SVD
library(irlba)

system.time(train_under_irlba <- irlba(t(train_under_dfm_tfidf), nv = 300, maxit = 600))

train_under_svd <- data.frame(sentiment = train_under$text, train_under_svd$v)

#Testing data SVD projection
sigma_inverse_under <- 1/train_under_irlba$d
u_tranpose_under <- t(train_under_irlba$u)

test_under_svd <- sigma_inverse_under * u_tranpose_under %*% t(test_under_token_tfidf)
test_under_svd <- t(test_under_svd)


#Model training
registerDoParallel(cl)
rf_under_model <- train(sentiment ~ ., data = train_under_svd, method = "rf",
                        trControl = cv.cntrl, tuneLength = 7)
stopCluster(cl)

#Testing the Model
pred_under <- predict(rf_under_model, test_under_svd)

#Model Assessment
confusionMatrix(pred_under, test_under$text)

##### 4.1 Uni-gram and Bi-grams undersampling ####

#Training Tokenization
train_under_token_12gram <- tokens(train_under$text, ngrams = 1:2)
head(train_under_token_12gram)

#Creating a DTM for Training Set
train_under_token_dfm_12gram <- dfm(train_under_token_12gram)
dim(train_under_token_dfm_12gram)

#TF-IDF 
train_under_token_dfm_tfidf_12gram <- dfm_tfidf(train_under_token_dfm_12gram, scheme_tf = 'prop')
train_under_token_dfm_tfidf_12gram[1:10,1:11]


#Testing Tokenization and DTM
test_under_token_12gram <- tokens(test_under$text, ngrams = 1:2)
test_under_token_dfm_12gram <- dfm(test_under_token_12gram)
dim(test_under_token_dfm_12gram)

test_under_token_dfm_12gram <- dfm_select(test_under_token_dfm_12gram, train_under_token_dfm_12gram, selection = 'keep')
dim(test_under_token_dfm_12gram)

#Applying TF-IDF to testing set
test_under_token_12gram_tf <- dfm_weight(test_under_token_dfm_12gram, scheme = 'prop')

train_under_docfreq_12gram <- docfreq(train_under_token_dfm_12gram, scheme = 'inverse')
train_under_docfreq_12gram

test_under_token_12gram_tfidf <- apply(test_under_token_12gram_tf, 1, tf_idf, idf = train_under_docfreq_12gram)
test_under_token_12gram_tfidf <- t(test_under_token_12gram_tfidf)
dim(test_under_token_12gram_tfidf) 

which(!complete.cases(test_under_token_12gram_tfidf))
test_under_token_12gram_tfidf[is.na(test_under_token_12gram_tfidf)] <- 0.0

#SVD 
library(irlba)
system.time(train_under_irlba_12gram <- irlba(t(train_under_token_dfm_tfidf_12gram), nv = 300, maxit = 600))

#Test data SVD projection
sigma_under_inverse_12gram <- 1/train_under_irlba_12gram$d
u_under_tranpose_12gram <- t(train_under_irlba_12gram$u)

test_under_svd_12gram <- t(sigma_under_inverse_12gram * u_under_tranpose_12gram %*% t(test_under_token_12gram_tfidf))
dim(test_under_svd_12gram)

test_under_svd_12gram <- data.frame(sentiment = test_under$text, test_under_svd_12gram)

#Model Training

rf_under_12gram <-  train(sentiment ~ ., data = train_under_svd_12gram, method = "rf",
                          trControl = cv.cntrl, tuneLength = 7) 

#Model Testing
pred_under_12gram <- predict(rf_under_12gram, newdata = test_under_svd_12gram)

#Model Assessment
confusionMatrix(pred_under_12gram, test_under$text)


#### 5 GloVE, word2vec####
library(text2vec)

text8_file = "~/text8/text8"
if (!file.exists(text8_file)) {
  download.file("http://mattmahoney.net/dc/text8.zip", "~/text8.zip")
  unzip ("~/text8.zip", files = "text8", exdir = "~/")
}
wiki = readLines(text8_file, n = 1, warn = FALSE)
space
# Create iterator over tokens
tokens <- space_tokenizer(wiki)
# Create vocabulary. Terms will be unigrams (simple words).
it = itoken(tokens, progressbar = FALSE)
vocab <- create_vocabulary(it)

vocab <- prune_vocabulary(vocab, term_count_min = 5L)

# Use our filtered vocabulary
vectorizer <- vocab_vectorizer(vocab)
# use window of 5 for context words
tcm <- create_tcm(it, vectorizer, skip_grams_window = 5L)

glove = GlobalVectors$new(word_vectors_size = 50, vocabulary = vocab, x_max = 10)

glove$fit_transform(tcm, n_iter = 20L, convergence_tol = -1, n_check_convergence = 1L,
                    n_threads = RcppParallel::defaultNumThreads())

word_vectors <- t(glove$get_word_vectors())
#tranpose the word vector matrix so that the row = words and column = context components. 

dim(word_vectors)

berlin <- word_vectors["paris", , drop = FALSE] - 
  word_vectors["france", , drop = FALSE] + 
  word_vectors["germany", , drop = FALSE]
#drop = F argument is important because it will return a matrix of dimension 1 X the context vector rather than a list of the context vector.

cos_sim<- sim2(x = word_vectors, y = berlin, method = "cosine") #for cos-sim to work the columns of two matricies have to be the same

head(sort(cos_sim[,1], decreasing = TRUE), 20)

####5.1 Creating features using context matrix####

glove_token <- tokens_select(train_token, pattern = vocab$term, selection = 'keep') 


#check for the number of documents with empty string (empyty string is different from missing texts)
length(glove_token[lengths(glove_token) == 0])

#Function to create the document context vector
average_vector <- function (tokens, word_vectors){
  count = 0
  final = data.frame()
  for (doc_num in names(tokens)){
    for(word in tokens[[doc_num]]){
      context_vec = word_vectors[word, ]
      word_length = length(tokens[[doc_num]])
      if (count == 0){
        cached_vec = context_vec
      } else {
        cached_vec = context_vec + cached_vec
      }
      count = count + 1
      if (count == word_length){
        cached_vec = cached_vec/word_length
        count = 0
        final = bind_rows(final, cached_vec)
        
      } else {
        cached_vec = cached_vec
      }
    }
  }
  return(final)
}

View(word_vectors)

final <- average_vector(tokens = glove_token, word_vectors = word_vectors)

#final <- rowMeans(cbind(word_vectors = word_vectors, tokens = glove_token))

#lenghts() check for the length of all elements in a list
empty_text <- tokens[lengths(tokens) == 0] 

#remove the word 'text' to get and turn them into integer to serve as indeces
empty_text <- as.integer(gsub("text", "", names(empty_text))) 

#Id all the twitter data
twitter_df <- df %>%
  mutate(id = 1:1260) %>%
  select(id, text, score_chr)


#remove data that have empty charaters
#twitter_df <- twitter_df[empty_text,]

#adding id to the context matrix
 View(twitter_df)

#create partition for training and testing data
set.seed(2)

indexes <- createDataPartition(twitter_df$score_chr, times = 1, p = 0.7, list = F)

training <- final[indexes,]
testing <- final[-indexes,]

View(indexes)

twitter_training <- twitter_df[indexes,]
twitter_testing <- twitter_df[-indexes,]

#training model

#setting cross-valiadation and control parameters
set.seed(12)
cv.folds <- createMultiFolds(twitter_training$score_chr, k = 10, times = 3)
cv.cntrl <- trainControl( method = 'repeatedcv', number = 10,
                          repeats = 3, index = cv.folds)

library(doParallel)
View(training)

cl <- makePSOCKcluster(10) #using multicores
registerDoParallel(cl)#register

rf_model_glove <- train(x = training, y = twitter_training$score_chr , method = "rf",
                        trControl = cv.cntrl, tuneLength = 7)



stopCluster(cl)
rf_model_glove


pred <- predict(rf_model_glove, newdata = testing)

#### 5.2 Model Assessment ####
confusionMatrix(pred, twitter_testing$score_chr)

