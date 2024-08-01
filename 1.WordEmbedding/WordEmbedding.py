import pandas as pd
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load and preprocess data
train = pd.read_csv("engtamilTrain.csv")
train = train.drop(["Unnamed: 0"], axis=1)
english_sentences = train["en"].head(1000)
tamil_sentences = train['ta'].head(1000)

def sent_token(sentence):
    """Tokenize sentences into words."""
    return [s.split() for s in sentence]

eng_sentence = sent_token(english_sentences)
tam_sentence = sent_token(tamil_sentences)

def own_word_model(lang_sentence, model_name):
    """Train and save Word2Vec model, then plot PCA visualization."""
    model = Word2Vec(lang_sentence, vector_size=100, window=5, min_count=1, workers=4)
    model.save(model_name)
    
    X = model.wv.vectors
    pca = PCA(n_components=2)
    result = pca.fit_transform(X)
    
    plt.scatter(result[:, 0], result[:, 1])
    words = list(model.wv.index_to_key)
    for i, word in enumerate(words):
        plt.annotate(word, xy=(result[i, 0], result[i, 1]))
    plt.show()

own_word_model(eng_sentence, "engmodel.bin")
own_word_model(tam_sentence, "tammodel.bin")
