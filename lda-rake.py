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
from rake_nltk import Rake


def set_corpus_dictionary(interval):
    '''
    set up a dictionary of papers
    :param interval: time intervals examined
    :return: dictionary of cleaned papers
    '''
    # start stanza processors
    # stanza.download('en')
    # nlp = stanza.Pipeline(lang='en', tokenize_pretokenized=True, processors='tokenize,mwt,pos,lemma')

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
        # doc = nlp(content)
        # filter xpos that is a verb, noun, or adjective
        # verb_noun_adj = ["JJ", "JJR", "JJS", "NN", "NNS", "NNP", "NNPS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]
        # outfile = ""
        # for sent in doc.sentences:
        #     for word in sent.words:
        #         if word.xpos in verb_noun_adj:
        #             outfile += (word.text + " ")
        # content = outfile
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
    content = re.sub('[,!?@®<=>/*×:;“”]', '', content)
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

    # # perform LDA on all review papers
    # for i in range(interval):
    #     plot_wordcloud(lemma[i])
    #     count_vectorizer = CountVectorizer(stop_words='english')
    #     count_data = count_vectorizer.fit_transform([lemma[i]])
    #     plot_most_common_words(count_data, count_vectorizer)

    keyword_list = []
    for i in range(interval):
        r = Rake()
        r.extract_keywords_from_text(paper_list[i])
        keyword_list.append(r.get_ranked_phrases_with_scores()[:10])

    # manual correction
    keyword_list[0] = [(407.9345291064145, 'gene,  pathways'),
                       (147.99235886836394, 'therapy, treatment'),
                       (106.30436056792016, 'key proteins, growth factors'),
                       (100.88506758351029, 'cell, receptors'),
                       (82.08239723938075, 'breast cancer, diagnosed'),
                       (81.78571428571429, 'mutation, pathway'),
                       (77.32178649237473, 'hormone, disease'),
                       (73.93594654992114, 'tumor, metastasis'),
                       (72.94799552997907, 'breast cancer, female'),
                       (72.93273092369478, 'tumor, model')]
    keyword_list[1] = [(81.75317460317461, 'gene expression'),
                       (69.97575427792819, 'sequencing, microrna'),
                       (59.30691721132897, 'methylation, analysis'),
                       (57.509661835748794, 'cell cycle, DNA'),
                       (54.66666666666667, 'proteins, enzymes'),
                       (49.86857142857143, 'treatment, target genes'),
                       (49.81397860773798, 'CNA, epigenetics'),
                       (49.731884057971016, 'dna, proteins'),
                       (48.22450980392158, 'IHC, breast cancer'),
                       (46.477777777777774, 'tumor, treatment')]
    keyword_list[2] = [(73.26587301587301, 'cancer cell, therapeutic'),
                       (71.76666666666668, 'dna, function'),
                       (61.607142857142854, 'interaction, network'),
                       (60.18888888888889, 'gene, association'),
                       (41.0, 'breast cancer, gene'),
                       (32.33333333333333, 'breast cancer, morphogenesis'),
                       (28.0, 'gene, analysis'),
                       (27.058441558441558, 'two genes encode protein kinase domains'),
                       (25.672619047619047, 'protein, annotations'),
                       (24.4106240981241, 'gene, functional')]

    f1 = pd.DataFrame(keyword_list[0], columns=["score", "keywords"])
    f2 = pd.DataFrame(keyword_list[1], columns=["score", "keywords"])
    f3 = pd.DataFrame(keyword_list[2], columns=["score", "keywords"])
    f1.to_csv("keyword_list_1.csv", index=False)
    f2.to_csv("keyword_list_2.csv", index=False)
    f3.to_csv("keyword_list_3.csv", index=False)
