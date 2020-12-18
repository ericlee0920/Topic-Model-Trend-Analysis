import pickle
import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
from gensim.models import Phrases, LdaModel
from gensim.models.phrases import Phraser
import gensim.corpora as corpora
from gensim.models import CoherenceModel
import spacy
import pyLDAvis
import pyLDAvis.gensim


def set_bigrams(text, bigram_phraser):
    return [bigram_phraser[i] for i in text]


def data_lemmatize(nlp, text, allowed_postags=['NOUN', 'VERB', 'ADJ']):
    """https://spacy.io/api/annotation"""
    return_list = []
    for i in text:
        doc = nlp(" ".join(i))
        return_list.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return return_list


def compute_coherence(texts, corpus, dictionary, k):
    lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=k, random_state=100,
                             chunksize=20, passes=10, alpha="auto", eta="auto")
    coherence_model_lda = CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v')
    return coherence_model_lda.get_coherence()


def get_topic_numbers(texts, corpus, word_ids, min, max):
    # evaluation range
    min_topics, max_topics = min, max
    topics_range = range(min_topics, max_topics + 1)
    topic_range = []
    coherence_results = []
    # iterate through number of topics
    for k in topics_range:
        # get the coherence score for the given parameters
        c_v = compute_coherence(texts=texts, corpus=corpus, dictionary=word_ids, k=k)
        # Save the model results
        topic_range.append(k)
        coherence_results.append(c_v)
    return topic_range, coherence_results


def plot_coherence(topic_range, coherence_results):
    sns.lineplot(x=topic_range, y=coherence_results)
    plt.title("Coherence Scores for Run {}".format(which_run))
    plt.xlabel("Number of Topics")
    plt.show()


if __name__ == "__main__":
    # read csv
    which_run = "1"
    paper = pd.read_csv("t{}.csv".format(which_run))
    paper = paper.iloc[:, 0].values.tolist()

    # tokenize
    for i in range(len(paper)):
        paper[i] = paper[i].split()
    # create bigram
    bigram = Phrases(paper, min_count=5, threshold=100)
    bigram_phraser = Phraser(bigram)
    data_bigrams = set_bigrams(paper, bigram_phraser)
    # lemmatize
    nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
    data_lemmatized = data_lemmatize(nlp, data_bigrams)
    # create dictionary
    word_ids = corpora.Dictionary(data_lemmatized)
    # create corpus, convert to bag of words
    corpus = [word_ids.doc2bow(text) for text in data_lemmatized]

    # build LDA model
    print("processing LDA...")
    lda_model = LdaModel(corpus=corpus, id2word=word_ids, num_topics=8, random_state=31, iterations=300,
                             chunksize=20, passes=10,  alpha="auto", eta="auto", per_word_topics=True)
    results = lda_model.print_topics()
    topic_df = pd.DataFrame(results, columns=["topic_no", "prob_words"])
    topic_df.to_csv("topic_{}.csv".format(which_run), index=False)

    # compute coherence score
    coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=word_ids, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('Coherence Score: ', coherence_lda)

    # train to get topic numbers
    topic_range, coherence_results = get_topic_numbers(data_lemmatized, corpus, word_ids, 6, 10)
    plot_coherence(topic_range, coherence_results)

    # visualize the topics
    LDAvis_prepared = pyLDAvis.gensim.prepare(lda_model, corpus, word_ids)
    pyLDAvis.show(LDAvis_prepared)






