import pandas
import re
import nltk
import emoji
from langdetect import detect
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.naive_bayes import MultinomialNB

train_data = pandas.read_csv("mediaeval-2015-trainingset.txt", sep="\\t", engine='python')
test_data = pandas.read_csv("mediaeval-2015-testset.txt", sep="\\t", engine='python')

ps = PorterStemmer()

list_stopwords = nltk.corpus.stopwords.words()
list_stopwords.extend([':', ';', '[', ']', '"', "'", '(', ')', '-', '.', '?', '#', '@', '&', '{', '}'])


def filter_tweet_by_language(tweet):
    try:
        language = detect(tweet)
        if language != "en":
            return ""
    except:
        return ""

    return tweet


def give_emoji_free_text(text):
    allchars = [str for str in text]
    emoji_list = [c for c in allchars if c in emoji.UNICODE_EMOJI]
    clean_text = ' '.join([str for str in text.split() if not any(i in str for i in emoji_list)])

    return clean_text


def pre_process(text):
    text = re.sub("http(s?):.*", "", text)  # remove urls incl. broken
    text = re.sub(".*RT:? @.*", "", text)  # remove all retweets
    text = give_emoji_free_text(text) # remove emojis
    text = re.sub("[.\'!/\\\\¿#|~_&?,\";“”*:\n]", " ", text)  # remove punctuation
    text = text.lower()  # make text lowercase
    text = re.sub(r'(.)\1{2,}', r'\1', text)  # cut words with more than 2 repeating characters
    text = re.sub("[ ]{2,}", " ", text)  # remove all unnecessary spaces
    text = stem(text) # perform stemming on the post

    return text


def stem(text):
    stemmed_tweet = ""
    stemmed_words = []
    tokenized_text = text.split()

    for word in tokenized_text:
        stemmed_words.append(ps.stem(word))

    for word in stemmed_words:
        stemmed_tweet = stemmed_tweet + ("" if stemmed_tweet == "" else " ") + word

    return stemmed_tweet


def tokenize():
    tokens = []
    for post in train_data.tweetText:
        tokenized_post = word_tokenize(post)
        filtered_tokenized_post = []

        for token in tokenized_post:
            if token not in list_stopwords:
                filtered_tokenized_post.append(token)
        tokens.append(filtered_tokenized_post)

    return tokens


def pos(list_tokens):
    pos_list = []
    for tokenized_post in list_tokens:
        pos_list.append(pos_tag(tokenized_post))

    return pos_list


def create_ngrams(list_pos, min_gram, max_gram, allow_pos):
    # list of ngram phrases
    list_ngrams = []

    for pos_tagged_sent in list_pos:
        sent_ngram_phrase = []
        for ngram in range(min_gram, max_gram + 1):
            list_pos_tokens = []
            list_pos_labels = []
            for (str_token, str_pos) in pos_tagged_sent:
                list_pos_tokens.append(str_token)
                list_pos_labels.append(str_pos)

            sent_token_ngrams = list(ngrams(sequence=list_pos_tokens, n=ngram))

            for tokens_ngram in sent_token_ngrams:
                sent_ngram_phrase.append(" ".join(i for i in tokens_ngram))

            if allow_pos:
                sent_label_ngrams = list(ngrams(sequence=list_pos_labels, n=ngram))

                for pos_ngram in sent_label_ngrams:
                    sent_ngram_phrase.append(" ".join(i for i in pos_ngram))

        list_ngrams.append(sent_ngram_phrase)

    return list_ngrams


def generate_feature_index(list_pos, list_ngrams, allow_pos):
    dict_feature_index = {}
    list_features = []
    feature_id = 0

    for pos_tagged_sent in list_pos:
        for(str_token, str_pos) in pos_tagged_sent:
            if str_token not in dict_feature_index:
                dict_feature_index[str_token] = feature_id
                list_features.append(str_token)
                feature_id += 1

            if allow_pos and str_pos not in dict_feature_index:
                dict_feature_index[str_pos] = feature_id
                list_features.append(str_pos)
                feature_id += 1

    for ngram_sent in list_ngrams:
        for ngram in ngram_sent:
            if ngram not in dict_feature_index:
                dict_feature_index[ngram] = feature_id
                list_features.append(ngram)
                feature_id += 1

    return dict_feature_index, list_features


# Currently replaced by Tfidf
def calc_count_vector(dict_index, list_pos, list_ngrams, allow_pos):
    list_count_vector = []
    labels = [0, 1]

    for label_index in range(len(labels)):
        list_count_vector.append([0] * len(dict_index))

    for index, pos_sent in enumerate(list_pos):
        label = 0 if train_data.loc[index].label == "fake" else 1
        for (str_token, str_pos) in pos_sent:
            if str_token in dict_index:
                list_count_vector[label][dict_index[str_token]] += 1
            if allow_pos and str_pos in dict_index:
                list_count_vector[label][dict_index[str_pos]] += 1

    for index, ngram_sent in enumerate(list_ngrams):
        label = 0 if train_data.loc[index].label == "fake" else 1
        for ngram in ngram_sent:
            if ngram in dict_index:
                list_count_vector[label][dict_index[ngram]] += 1

    return list_count_vector, labels


# Currently replaced by Tfidf
def calc_test_train_matrix(data, set_features, list_pos, list_ngrams, allow_pos):
    index_labels = {"fake": 0, "real": 1}
    index_features = {}
    feature_index = 0

    for feature in set_features:
        index_features[feature] = feature_index
        feature_index = feature_index + 1

    X = []
    Y = []
    list_feature_set = list(set_features)

    for index, post in data.iterrows():
        label = index_labels[post.label]
        Y.append(label)

        list_freq_vector = [0] * len(list_feature_set)

        for (str_token, str_pos) in list_pos[index]:
            if str_token in index_features:
                list_freq_vector[index_features[str_token]] += 1
            if allow_pos and str_pos in index_features:
                list_freq_vector[index_features[str_pos]] += 1

        for ngram in list_ngrams[index]:
            if ngram in index_features:
                list_freq_vector[index_features[ngram]] += 1

        X.append(list_freq_vector)

    return X, Y


# Preprocessing
train_data.loc[train_data['label'] == 'humor', 'label'] = 'fake'
train_data.tweetText = train_data.tweetText.apply(lambda x: pre_process(x))
test_data.tweetText = test_data.tweetText.apply(lambda x: pre_process(x))

# Feature selection
tokenized_posts = tokenize()
pos_tagged_tokens = pos(tokenized_posts)
list_all_ngrams = create_ngrams(pos_tagged_tokens, 2, 3, False)
(dict_feature_index, list_features) = generate_feature_index(pos_tagged_tokens, list_all_ngrams, False)

tfidf_vectorizer=TfidfVectorizer(stop_words=list_stopwords, vocabulary=dict_feature_index)

X_train = tfidf_vectorizer.fit_transform(train_data.tweetText)
Y_train = train_data.label
X_test = tfidf_vectorizer.transform(test_data.tweetText)
Y_test = test_data.label

# Classification
clf = MultinomialNB(alpha=1.3)
clf.fit(X_train, Y_train)

Y_pred = clf.predict(X_test)
score = f1_score(Y_test, Y_pred, pos_label="fake")
F1 = round(score*100,2)
print(f'F1 score: {F1}%')

print(classification_report(Y_test, Y_pred))

