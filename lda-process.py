import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')

# reading all processed topic models
interval = 3
topic_models = []
for i in range(interval):
    topic_models.append(pd.read_csv("processed_{}.csv".format(i + 1)))

# processed matrix
# model_matrix[0/1/2][0/1][0-8/9] <=> [period of time],  [topic_number], [word/prob]
model_matrix = []
words = []
for i in range(interval):
    processing = topic_models[i]["prob_words"]
    # need to normalize
    token_prob = list(topic_models[i]["perc_token"])
    token_prob = [t / sum(token_prob) for t in token_prob]
    topic_count = processing.shape[0]
    # prob_matrix = []
    # word_matrix = []
    dict_matrix = []
    for j in range(topic_count):
        sentence = processing[j]
        prob = re.sub('[^\d.+]', '', sentence)
        prob = [float(k) for k in prob.split('+')]
        # prob_matrix.append(prob)
        word = re.sub('[^a-z+]', '', sentence)
        word = [w for w in word.split('+')]
        words.append(word)
        # word_matrix.append(word)
        dict_matrix.append({word[d]: prob[d] for d in range(10)})
    topic = []
    # topic_matrix.append(prob_matrix)
    topic = list(zip(dict_matrix, token_prob))

    # topic_matrix.append(word_matrix)
    # topic_matrix.append(token_prob)
    model_matrix.append(topic)

# find max similarity
# similarity_matrix[0/1]{0-len} <=> [period of time],  {topic_number: most similar one in next period}
similarity_matrix = []
for i in [0, 1]:
    processing = model_matrix[i]
    index = {d: [] for d in range(len(processing))}
    for j in range(len(processing)):
        topic = processing[j][0]
        referencing = model_matrix[i - 1]
        max_index = []
        max = 0
        max_sum = 0
        for k in range(len(referencing)):
            words_in_common = set([*topic]) & set([*referencing[k][0]])
            len_in_common = len(words_in_common)
            sum = 0
            for w in words_in_common:
                sum += topic[w]

            if sum > max_sum:
                max = len_in_common
                max_index = [k]
                max_sum = sum
            elif sum == max_sum:
                if len_in_common > max:
                    max = len_in_common
                    max_index = [k]
                elif len_in_common == max:
                    max_index.append(k)
        index[j] = max_index
    similarity_matrix.append(index)
