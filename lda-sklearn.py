import re
import stanza
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
sns.set_style('whitegrid')
from nltk.stem import WordNetLemmatizer


def set_corpus_dictionary(interval):
    '''
    set up a dictionary of papers
    :param interval: time intervals examined
    :return: dictionary of cleaned papers
    '''
    # start stanza processors
    stanza.download('en')
    nlp = stanza.Pipeline(lang='en', tokenize_pretokenized=True, processors='tokenize,mwt,pos,lemma')

    # collection of context
    paper_content_list = []
    for i in range(interval):
        print("Paper" + str(i + 1))
        paper_num = str(i + 1).zfill(2)
        # read papers
        paper = open("papers/review{}.txt".format(paper_num), encoding="utf-8")
        content = paper.read()
        # proprocessing steps, call data_cleaning
        content = data_cleaning(content)
        # pos tagging
        doc = nlp(content)
        # filter xpos that is a verb, noun, or adjective
        verb_noun_adj = ["JJ", "JJR", "JJS", "NN", "NNS", "NNP", "NNPS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]
        outfile = ""
        for sent in doc.sentences:
            for word in sent.words:
                if word.xpos in verb_noun_adj:
                    outfile += (word.text + " ")
        content = outfile
        paper_content_list.append(content)
    return paper_content_list


def data_cleaning(content):
    '''
    citation, punctuation, stop word, symbol, link, paretheses removal
    :param content: raw paper
    :return: cleaned single paper
    '''
    # remove citations
    content = re.sub('\[.*\]', '', content)
    # remove parenthesis
    content = re.sub('[\(\)\{\}]', '', content)
    content = content.replace("\"", "")
    content = content.replace("'", "")
    # remove links
    pattern = re.compile(r'\bhttp\S*', flags=re.IGNORECASE)
    content = pattern.sub("", content)
    # remove specific words
    specific_stop_words = ["abstract", "introduction", "summary", "result", "results", "conclusion",
                           "discussion", "supplementary", "data", "dataset", "experiment", "experiments",
                           "fig", "figure", "figures", "caption", "author", "authors", "et al", "et", "al",
                           "method", "methods", "base", "based", "approach", "use", "uses", "used",
                           "obtain", "obtained", "propose", "proposed", "main", "background",
                           "computational", "biological", "biology", "science", "human", "bioinformatics",
                           "important", "similar", "different", "related", "known", "unknown", "using"]

    remove = '|'.join(specific_stop_words)
    pattern = re.compile(r'\b(' + remove + r')\b', flags=re.IGNORECASE)
    content = pattern.sub("", content)
    # remove unnecessary symbols
    content = re.sub('[,\.!?@®<=>/*×:;“”]', '', content)
    return content


def plot_wordcloud(content):
    """
    plot a word cloud
    :param content: string text
    :return: wordcloud plotted
    """
    wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
    wordcloud.generate(content)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()


def plot_most_common_words(count_data, count_vectorizer):
    words = count_vectorizer.get_feature_names()
    total_counts = np.zeros(len(words))
    for t in count_data:
        total_counts += t.toarray()[0]
    count_dict = (zip(words, total_counts))
    count_dict = sorted(count_dict, key=lambda x: x[1], reverse=True)[:10]
    words = [w[0] for w in count_dict]
    counts = [w[1] for w in count_dict]
    x_pos = np.arange(len(words))
    sns.barplot(x_pos, counts, palette='husl')
    plt.xticks(x_pos, words, rotation=90)
    plt.xlabel('Keywords')
    plt.ylabel('Count')
    plt.show()


def print_topics(model, count_vectorizer, n_top_words):
    words = count_vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(model.components_):
        print("\nTopic #{}".format(topic_idx))
        print(" ".join([words[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))


if __name__ == "__main__":
    # number of intervals
    interval = 3
    # prepare dataset
    paper_list = set_corpus_dictionary(interval)

    # lemmatize
    lemma = []
    for i in range(interval):
        lemmatizer = WordNetLemmatizer()
        lemma.append(paper_list[i].split())
        lemma[i] = ' '.join([lemmatizer.lemmatize(w) for w in lemma[i]])


    # perform LDA on all review papers
    for i in range(interval):
        plot_wordcloud(lemma[i])
        count_vectorizer = CountVectorizer(stop_words='english')
        count_data = count_vectorizer.fit_transform([lemma[i]])
        plot_most_common_words(count_data, count_vectorizer)
