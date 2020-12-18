# Trend Analysis of Computational Breast Cancer Research
## Updated: December 16, 2020.
### Background and Rationale
This is a workflow for trend analysis of computational breast cancer research using topic modeling.

The underlying approach to this workflow has four modules:
- Dataset Construction
- Pre-processing
- Topic Modeling
- Trend Analysis

Package Dependencies:
*You should have Python 3 and Miniconda installed for Linux*
  - matplotlib
  - numpy
  - pandas
  - scipy
  - scikit-learn
  - seaborn
  - stanza
  - torch
  - gensim
  - nltk
  - nltk_rake

### Python Workflow

1. `lda-stanza.py` is to create preprocessed data using the article text files numbered `1-60` in `/papers`. This step outputs csv files of `t1`, `t2`, `t3`.
2. `lda-gensum.py` is to create topics from constructing a topic model using LDA and the preprocessed data in csv files of `t1-3`. This step outputs csv files of `topic_1`, `topic_2`, `topic_3`.
3. `lda-sklearn.py` is to create exploratory data analysis of articles.
4. `lda-rake.py` is to create ground truth data using the article text files named `review01`, `review02`, and `review03` in `/papers`. This step outputs in console keywords of each review article, yet this REQUIRES expertise to shorten keywords. The expert should output `keyword_list_1`, `keyword_list_2`, `keyword_list_3`.
5. `lda-process.py` is a special usage file for experts to compare which topics in the time periods are like to be the same. This gives the output of the most similar topic from t_i to t_i+1.
6. `lda-plot.py` is to plot the trends using the expert processed data `processed_1`, `processed_2`, and `processed_3`. These processed files are made from `topic_1`, `topic_2`, `topic_3`. Running this file will generate the resulting graphs in `/supplementary_figures`.


### Usage

1. Clone this file in a proper directory. This will download all the files and datasets necessary for execution.
```
git clone https://github.com/ericlee0920/Topic-Model-Trend-Analysis.git
```
2. Download all the dependencies.
3. Inspect all your files in `/papers`, text files named `1` to `60` correspond to the 60 papers in the order of the file `breast-cancer-corpus.csv`, which makes us three corpora. `1-20` are from 2016-2020, `21-40` are from 2011-2015, and `41-60` are from 2006-2010.
Review papers corresponding to the three periods also go in order of text files named `review01`, `review02`, and `review03`.
4. Inspect all your csv files, topic modeling results that have numbered topics being labeled/named are processed here. The csv files corresponding to the three periods also go in order of `processed_1`, `processed_2`, `processed_3`. 
`t1-3` and `topic_1-3` are intermediate files for reference of the results in the paper. The former is the result of running `lda-stanza.py` to get preprocessed data, and the ladder is the result of running `lda-gensim.py` to get topics of each time period.

### Figures
In `supplementary_figures`, there are many figures: *NOTE* that figures with the same name that have three instances - 1 refers to 2016-2020, 2 refers to 2011-2015, and 3 refers to 2006-2010. 
1. `eda` refers to exploratory data analysis of the top ten words in the articles in each time period.
2. `corpus` refers to exploratory data analysis of creating wordclouds of the 20 corpus articles in each time period; while `review` refers to exploratory data analysis of creating wordclouds of the single review article in each time period.
3. `coherence` refers to the coherence trace of different number of topics using the same hyperparameters.
4. `vis` refers to the output of pyLDAvis for eahc topic in each time period.
5. `trend_results` refer to the final product of this study which are the trends of computational breast cancer research.





