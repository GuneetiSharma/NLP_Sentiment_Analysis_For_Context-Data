#Lexicon based sentiment analysis on workership data
#libraries required for Lexicon based sentiment analysis and visualization

install.packages("plyr")
install.packages("stringr")
install.packages("ggplot2")

#loading the library
library(plyr)
library(stringr)
library(ggplot2)

# Read the CSV file with the sentence level breakdown for comments
comment_ws_df <- read.csv(choose.files())
#Let's view the data we retrieved for our analysis
View(comment_ws_df)

###
###TextCleaning###
###

#for Mac or Unix OS
comments_data <- sapply(comment_ws_df$comment_value, function(x) iconv(x, to='UTF-8-MAC', sub='byte'))

#for Windows based OS
comments_data <- sapply(comment_ws_df$comment_value,function(row) iconv(row, "latin1", "ASCII", sub=""))

#common for any platform
comments_data <- gsub("@\\w+", "", comments_data)
comments_data <- gsub("#\\w+", '', comments_data)
comments_data <- gsub("RT\\w+", "", comments_data)
comments_data <- gsub("http.*", "", comments_data)
comments_data <- gsub("RT", "", comments_data)
comments_data <- sub("([.-])|[[:punct:]]", "\\1", comments_data)
comments_data <- sub("(['])|[[:punct:]]", "\\1", comments_data)

#View the cleaned Data
View(comments_data)

#Reading the Lexicon positive and negative words
pos<-readLines("positive_words.txt")
neg<-readLines("negative_words.txt")

#function to calculate sentiment score
score.sentiment <- function(sentences, pos.words, neg.words, .progress='none')
{
  # Parameters
  # sentences: vector of text to score
  # pos.words: vector of words of postive sentiment
  # neg.words: vector of words of negative sentiment
  # .progress: passed to laply() to control of progress bar
  
  # create simple array of scores with laply
  scores <- laply(sentences,
                  function(sentence, pos.words, neg.words)
                  {
                    # remove punctuation
                    sentence <- gsub("[[:punct:]]", "", sentence)
                    # remove control characters
                    sentence <- gsub("[[:cntrl:]]", "", sentence)
                    # remove digits
                    sentence <- gsub('\\d+', '', sentence)
                    
                    #convert to lower
                    sentence <- tolower(sentence)
                    
                    
                    # split sentence into words with str_split (stringr package)
                    word.list <- str_split(sentence, "\\s+")
                    words <- unlist(word.list)
                    
                    # compare words to the dictionaries of positive & negative terms
                    pos.matches <- match(words, pos)
                    neg.matches <- match(words, neg)
                    
                    # get the position of the matched term or NA
                    # we just want a TRUE/FALSE
                    pos.matches <- !is.na(pos.matches)
                    neg.matches <- !is.na(neg.matches)
                    
                    # final score
                    score <- sum(pos.matches) - sum(neg.matches)
                    return(score)
                  }, pos.words, neg.words, .progress=.progress )
  # data frame with scores for each sentence
  scores.df <- data.frame(text=sentences, score=scores)
  return(scores.df)
}

#sentiment score
score_ws <- score.sentiment(comments_data, pos, neg, .progress='text')
View(score_ws)

#Summary of the sentiment scores
summary(score_ws)

#Convert sentiment scores from numeric to character to enable the gsub function 
score_ws$score_chr <- as.character(score_ws$score)

#After looking at the summary(scores_twitter$score) decide on a threshold for the sentiment labels
score_ws$score_chr <- gsub("^0$", "Neutral", score_ws$score_chr)
score_ws$score_chr <- gsub("^1$|^2$|^3$|^4$|^5$|^6$|^7$|^8$|^9$|^10$", "Positive", score_ws$score_chr)
score_ws$score_chr <- gsub("^11$|^12$|^13$|^14$|^15$", "Very Positive", score_ws$score_chr)
score_ws$score_chr <- gsub("^-1$|^-2$|^-3$|^-4$|^-5$|^-6$|^-7$|^-8$|^-9$|^-10$", "Negative", score_ws$score_chr)
score_ws$score_chr <- gsub("^-11$|^-12$|^-13$|^-14$|^-15$|^-16$|^-17$|^-18$|^-19$|^-20$|^-21$|^-22$|^-23$|^-24$", "Very Negative", score_ws$score_chr)

View(score_ws)

#Convert score_chr to factor for visualizations
score_ws$score_chr<-as.factor(score_ws$score_chr)

#plot to show number of negative, positive and neutral comments
Viz1 <- ggplot(score_ws, aes(x=score_chr))+geom_bar()
Viz1

#writing to csv file
write.csv(score_ws, file = "sentiment_score_comment.csv")
