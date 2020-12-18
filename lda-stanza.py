import re
import stanza
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
sns.set_style('whitegrid')


def set_corpus_dictionary(interval):
    '''
    set up a dictionary of papers
    :param interval: time intervals examined
    :return: dictionary of cleaned papers
    '''
    # collection of context
    timepoints = {key: [] for key in range(interval)}
    # adding paper information
    for t in range(interval):
        start_paper_index = 0 + t * 20
        end_paper_index = 20 + t * 20
        # temporary list for collection
        paper_content_list = []
        for i in range(start_paper_index, end_paper_index):
            print("Paper"+str(i+1))
            paper_num = str(i + 1).zfill(2)
            # read papers
            paper = open("papers/{}.txt".format(paper_num), encoding="utf-8")
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
        # send to dictionary and clear list
        timepoints[t] = paper_content_list
        paper_content_list = []
    return timepoints


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


if __name__ == "__main__":
    # start stanza processors
    stanza.download('en')
    nlp = stanza.Pipeline(lang='en', tokenize_pretokenized=True, processors='tokenize,mwt,pos,lemma')
    # number of intervals
    interval = 3

    # prepare dataset
    paper_dictionary = set_corpus_dictionary(interval)

    # output results to csv
    for i in range(interval):
        plot_wordcloud(" ".join(paper_dictionary[i]))
        pd.DataFrame(paper_dictionary[i]).to_csv("t{}.csv".format(i+1), index=False)

