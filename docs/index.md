<p align="center">
  <img width="600" height="300" src="https://i.imgur.com/HmRWKfG.jpg">
</p>

# FinBox NLP Exercise

Documentation for FinBox NLP Exercise

## About Dataset

The Dataset is a collection of tweets collected from Twitter and dumped it in a csv format.The csv file contains three columns/features: 

1. **created_at:** The time when the tweet was created.
2. **text:** The text of the tweet.
3. **is_retweet:** A boolean value(True/False) indicating whether the tweet is a retweet or not.
   
 | created_at | text | is_retweet |
 | -------- | -------- | -------- |
 | 2019-04-22 14:53:19  | If my heart can barely handle a break up, I don<U+2019>t even want to know what<U+2019>s going to happen to it on the next episode of Game of Thrones | False  |

### Distribution of Dataset

The dataset contains around ***1 lakh tweets***, which has a lot of special characters,hashtags,links,emojis etc.

Dataset also donot have any null values.

Below is the distribution of the dataset in pandas dataframe:

```python
import pandas as pd
train_df = pd.read_csv('/content/finbox/finbox_nlp_test_100k.csv')
train_df.info()
```
Output:
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 102689 entries, 0 to 102688
Data columns (total 3 columns):
 #   Column      Non-Null Count   Dtype 
---  ------      --------------   ----- 
 0   created_at  102689 non-null  object
 1   text        102689 non-null  object
 2   is_retweet  102689 non-null  bool  
dtypes: bool(1), object(2)
memory usage: 1.7+ MB
```

So, I have decided to perform some kind of a analysis to get to know what is the data is all about, which word has more count and which word has less count & Common words by using some plot's.

Below are some results of analysis.I have taken only the ***top 20 Common words from the dataset and their count.***

![count](https://i.imgur.com/88eKSkg.png)

From this Analysis I have got to know few intresting things about the dataset.

1. The tweet's are based on Game of the Thrones web series.
2. Most common words are(excluding stop words) are:
    1. Game
    2. Thrones
3. As Dataset require some extensive cleaning as it contains many stopwords which are contributing a lot to the dataset this act as a blockage for other word's which are important in tweet's.

Below are some nice plot's on above analysis which I have done.

***Bar Plot Illustrating Common_Word's Vs Word_Count***

![Placeholder](https://i.imgur.com/4Yq05Kk.png)

***Treemap Illustrating Common_Word's Vs Word_Count***

![Placeholder](https://i.imgur.com/2SGFBbF.png)


## Text Preprocessing

After getting the basic insights, I found that there are some special characters,unicodes,emojis,hashtags,urls,stopwords and mentions in the dataset.

And I have also noticed that the dataset also have these text in more ammount:

1. Telegram Link's : https://t.co/hn8P0SlF43
2. Unicode's : ***uf602f602***



So,I decided to perform a extensive cleaning.

For performing text cleaning I have used the following package:

1. [Regex](https://pypi.org/project/regex/) - Regular Expression
2. [NLTK](https://www.nltk.org/) - Natural Language Toolkit

These two packages helped me to perform a lot of text cleaning onto this dataset.

Also I have taken a list of custom Stopwords which are meant for twitter tweets, after creating a list I have extended it with NLTK Stopwords.
***(list of custom stopword present in notebook)***

Below is the code for performing text cleaning.After cleaning the cleaned dataset is dumped in a csv file named as ***clean_data.csv***.

```python
import nltk
from nltk.corpus import stopwords
import re

Combined = re.compile('(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)') #This will remove every link,symbols,bad_symbols form text

unicodes = re.compile('u\d+') # for unicodes
ALL_STOPWORDS = stopwords.words('english')
ALL_STOPWORDS.extend(STOPWORDS) #NLTK STOPWORDS+CUSTOM TWITTER STOPWORDS

def clean_text(text):
    
    if(text!=text):
        return ""
    text = text.lower() # lowercase text
    
    
    text = Combined.sub('',text)
    text = unicodes.sub('',text)
    

    text = ' '.join(word for word in text.split() if word not in ALL_STOPWORDS) # delete stopwors from text
    return text

clean_train_data = dict()
clean_train_data['text'] = df['text'].apply(clean_text)
clean_train_data['is_retweet'] = df['is_retweet']

```

Now I have got the cleaned dataset, now let's see the common_word and it's count.

![Placeholder](https://i.imgur.com/zcYr2P2.png)

From the above table I have got to know that the most common words are:

1. game
2. thrones
3. season
4. episode

Now if we see their is a huge difference in the common words between the original dataset and the cleaned dataset.

So,what I have done in data cleaning is the removal of:

1. Stopwords (the,a,is,are,etc)
2. Special Characters (@,#,$,%,&,*,_,\,/,:,;,<,>,?,\',\",.,!,~,`)
3. Unicodes (u000f,u00a0,u00a1,u00a2,u00a3, etc)
4. Links (https://t.co/hn8P0SlF43)

Below is a bar plot on cleaned dataset which I have done.

***Bar Plot Illustrating Common_Word's Vs Word_Count on Clean_Data***

![Placeholder](https://i.imgur.com/kue9Wla.png)

At the end the text is cleaned more than 90% and the dataset is ready for further analysis and language modelling related tasks.

Their are also two more columns present in the dataset which are datetime and is_retweeted, due to the single label(**only False**) in is_retweeted column I have decided to not to involve it in the analysis.

For ***created_at column***(Datetime) I have decided to use it for plotting the time series but as their is no feature is present for comparsion wrt datetime, I have decided to not to use that column.

So at the end I have decided to drop the created_at column and is_retweeted column.

And now I have left with the following columns:

1. idx
2. text

 ![Placeholder](https://i.imgur.com/Zip9Pwy.png)

## Getting Sentiment's using Vader (Unsupervised Learning)

Now, after performing preprocessing,we have got a dataframe which I have shown you above, now we have text but we don't have sentiments,so to get the sentiment's or labels on 102600 tweets I am going to use vader library.

### Vader

VADER (***Valence Aware Dictionary and sEntiment Reasoner***) is a lexicon and rule-based sentiment analysis tool that is specifically attuned to sentiments expressed in social media.

So, this library is majorly based on giving a sentiment on social media dataset's like tweets,instagram posts,facebook posts that's why I have decided to use it.

Below is the code-block for performing sentiment analysis using **VaderSentiment**.

```python
from nltk.sentiment.vader import SentimentIntensityAnalyzer
#0 for negative sentiment
#1 for positive sentiment
analyzer = SentimentIntensityAnalyzer()

def vader_sentiment_result(sent):
    scores = analyzer.polarity_scores(str(sent))
    
    if scores["neg"] > scores["pos"]:
        return 0

    return 1

clean_train_data['label'] = clean_train_data['text'].apply(lambda x: vader_sentiment_result(x))
```

The above code-block will assign a sentiment/label to all the text.

The two sentiment's are:

1. Negative(0)
2. Positive(1)

Now our Dataframe will look like this:

 ![Placeholder](https://i.imgur.com/YQWX1sJ.png)

**Mapping the label columns:**

```python
clean_train_data.label=clean_train_data.label.map({1:"Positive",0:"Negative"})
```

```python
clean_train_data
```
![Placeholder](https://i.imgur.com/S3eP3EE.png)

## Sentiment Analysis using AWD-LSTM and Fastai

I am going to create a DataBlock in which we are goign to use a seq_len as 72, why 72? As if you look on to the sequences in detail you will find that most of the sequences are less than 72 words.

Here is the DataBlock:

```python
tweet_clas = DataBlock(
    blocks=(TextBlock.from_df('text', seq_len=72), CategoryBlock),
    get_x=ColReader('text'), get_y=ColReader('label'))
```

Now creating a DataLoader:

```python
tweet = tweet_clas.dataloaders(clean_train_data, bs = 64, is_lm=True)
```

I am using a `batch_size=64` because we have a lot of records and 64 is the optimal size, you can change it according to your need.

Now here is our batch:

```python
tweet.show_batch(max_n=3)
```
![Placeholder](https://i.imgur.com/nzE3JrL.png)

You can see we have some different special token's added in the sentence, you can get the more information about them below:

1. **UNK (xxunk)** is for an unknown word (one that isn't present in the current vocabulary)

2. **PAD (xxpad)** is the token used for padding, if we need to regroup several texts of different lengths in a batch

3. **BOS (xxbos)** represents the beginning of a text in your dataset

4. **FLD (xxfld)** is used if you set mark_fields=True in your TokenizeProcessor to separate the different fields of texts (if your texts are loaded from several columns in a dataframe)

5. **TK_MAJ (xxmaj)** is used to indicate the next word begins with a capital in the original text

6. **TK_UP (xxup)** is used to indicate the next word is written in all caps in the original text

7. **TK_REP (xxrep)** is used to indicate the next character is repeated n times in the original text (usage xxrep n {char})

8. **TK_WREP(xxwrep)** is used to indicate the next word is repeated n times in the original text (usage xxwrep n {word})

## Model Building

So,for model building I am going to use **ULMFit by Fastai** which uses **awd-lstm** as a base model, so lstm is the best choice for this task as it has shown great result's when used inside ulmfit as ULMFit has achived a state of the art performance for these kind of seqeunce classification tasks.

Let's intilize our learner:

```python
# fintuning the model:
learn = text_classifier_learner(tweet, AWD_LSTM, metrics = [accuracy, Perplexity()]).to_fp16()
```

I am using metric as **accuracy** and **perplexity** as it is the best metric for this kind of task classification task.

Let's find a Optimal learning rate:

```python
learn.lr_find()
```

 ![Placeholder](https://i.imgur.com/5gYVU3c.png)

Best `lr is 1e-3`, so let's set it and start training:

```
epochs = 10
lr = 1e-3
```

```python
learn.fit_one_cycle(10, 1e-3)
```

 ![Placeholder](https://i.imgur.com/OpYNiAt.png)

 So, after 10 epochs I have achived a accuracy of **0.77** and perplexity of **1.5**, this is just a intial result's, we can also fit our model with more epochs and see the result.

 Now, let's see the result:

 1. Confusion Matrix:

 ![Placeholder](https://i.imgur.com/xfAUJMa.png)

 2. Classification Report:

 ![Placeholder](https://i.imgur.com/wz1khCg.png)

 Now, at end I have a model which is trained and ready to use, I am going to train it on more epochs and then we can get more accurate result's.


## Requried Libraries

1. [NLTK](https://www.nltk.org/) - Natural Language Toolkit
2. [Regex](https://pypi.org/project/regex/) - Regular Expression
3. [Pandas](https://pandas.pydata.org/) - Data Analysis Library
4. [NumPy](https://www.numpy.org/) - Numerical Python
5. [Matplotlib](https://matplotlib.org/) - Python 2D/3D plotting library
6. [Plotly](https://plotly.com/) - Interactive graphing and visualization library
7. [Fastai](fast.ai) - Deep Learning Library
8. nltk vader sentiment analyzer - Sentiment Analysis Library
9. [vaderSentiment](https://github.com/cjhutto/vaderSentiment) - Sentiment Analysis Library
10. [twython](https://twython.readthedocs.io/en/latest/) - Twitter API

## Challenges Faced

I have faced few challenges while performing Text Cleaning and these are that the Text has a lot of unnecessary special characters, links, and Unicode, and finding a proper regex expression is a little bit tough for me as I have faced a lot of difficulty in removing Unicode because if we see the text data some Unicode are in proper format like this **u000f** but after performing some cleaning operation the Unicode which is in a single line separated by space come close to each other and form an alphanumeric expression, now regex can only remove a particular Unicode, but now I have a string of Unicode and this will differ from text to text and there is not a particular regex expression available for it.

But in the end, I have come up with a solution to solve this and remove some of the complex Unicode like this **u0001f64fu0001f3ffu0001f918u0001f3ffu0001f447u0001f3ff**

## My Model Experimentation

While working in the model building and selection phase, I have performed few experiments:

 **bert-base-uncased** : I have used this model with a fastai wraper on Hugging Face Transformer which is [blurr](https://github.com/ohmeow/blurr), but while intializing a learner, I am getting an error message `Cuda out of memory error` I have even controlled my batch size but it is throwing the same error.

 I will work upon this model later and will try to find a proper solution.


## Other Possible Solutions to this Unsupervised Sentiment Analysis Problem

We can have a two different cluster's which are **positive** and **negative** and then we can use **K-Means** to cluster the data.

## Instruction to run the project

* Find the link to the Google Colab Notebook Below, and run the Notebook.
* You can also clone the github repository present below, and run the Jupyter Notebook after installing the requirements.txt on your conda environment.

## Important Links

* Find Google Colab Notebook Here: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1V4_JT2rMNaZNPUPiGiiQWW97LEDYotCC?usp=sharing)
* Open in Binder: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/DARK-art108/FinBox-NLP-Exercise/main)
* Github Repository: [![GitHub](https://badgen.net/badge/icon/github?icon=github&label)](https://github.com/DARK-art108/FinBox-NLP-Exercise)


