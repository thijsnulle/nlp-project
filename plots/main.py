import plotly
import numpy as np
import plotly.graph_objs as go
from sklearn.decomposition import PCA

from model import get_model, save_model
from words import get_flatten_words, get_only_successful_words

def display_pca_scatterplot_3D(model, user_input=None, allwords=None, label=None, color_map=None, topn=10, sample=10):
    print("len words", len(allwords), len(label), len(color_map), len(user_input))
    rnn, embedding, attention = allwords

    words = rnn + embedding + attention
    print("rnn, embedding, attention", len(rnn), len(embedding), len(attention))

    print("words", words)
    if words == None:
        print("we dont have words?")
        if sample > 0:
            print("No do we get here?")
            words = np.random.choice(list(model.vocab.keys()), sample)
        else:
            print("we got words right?")
            words = [ word for word in model.vocab ]

    word_vectors = np.array([model[w] for w in words])
    print("word vectors", len(word_vectors))

    three_dim = PCA(random_state=0).fit_transform(word_vectors)[:,:2]
    # For 2D, change the three_dim variable into something like two_dim like the following:
    # two_dim = PCA(random_state=0).fit_transform(word_vectors)[:,:2]

    data = []
    count = 0

    for i in range (len(user_input)):
                if i == 0:
                    topn = len(rnn)
                elif i == 1:
                    topn = len(embedding)
                elif i == 2:
                    topn = len(attention)

                trace = go.Scatter(
                    x = three_dim[count:count+topn,0],
                    y = three_dim[count:count+topn,1],
                    text = words[count:count+topn],
                    name = user_input[i],
                    textposition = "top center",
                    textfont_size = 20,
                    mode = 'markers+text',
                    marker = {
                        'size': 10,
                        'opacity': 0.8,
                        'color': 2
                    }

                )

                # For 2D, instead of using go.Scatter3d, we need to use go.Scatter and delete the z variable. Also, instead of using
                # variable three_dim, use the variable that we have declared earlier (e.g two_dim)

                data.append(trace)
                count = count+topn

    trace_input = go.Scatter(
                    x = three_dim[count:,0],
                    y = three_dim[count:,1],
                    text = words[count:],
                    name = 'input words',
                    textposition = "top center",
                    textfont_size = 20,
                    mode = 'markers+text',
                    marker = {
                        'size': 15,
                        'opacity': 1,
                        'color': 'black'
                    }
                    )

    # For 2D, instead of using go.Scatter3d, we need to use go.Scatter and delete the z variable.  Also, instead of using
    # variable three_dim, use the variable that we have declared earlier (e.g two_dim)

    data.append(trace_input)

# Configure the layout

    layout = go.Layout(
        margin = {'l': 0, 'r': 0, 'b': 0, 't': 0},
        showlegend=True,
        legend=dict(
        x=1,
        y=0.5,
        font=dict(
            family="Courier New",
            size=25,
            color="black"
        )),
        font = dict(
            family = " Courier New ",
            size = 15),
        autosize = False,
        width = 1100,
        height = 1000
        )


    plot_figure = go.Figure(data = data, layout = layout)
    print("plotting it now?")
    plot_figure.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    """
    This is the file to create the plots!
    **Paste the words in words.py**
    """
    PLOT_SUCCESSFUL_ONLY = False
    MAX_WORDS = -1 # -1 means all words

    # Check which words you want to plot
    if PLOT_SUCCESSFUL_ONLY:
        rnn_words, embedding_words, attention_words = get_only_successful_words()
    else:
        rnn_words, embedding_words, attention_words = get_flatten_words()

    print("Plotting", len(rnn_words))
    # Number of words you want to plot.
    if MAX_WORDS == -1:
        MAX_WORDS = max(len(rnn_words), len(embedding_words), len(attention_words))

    print("Words", len(rnn_words), len(embedding_words), len(attention_words))

    # Only select the subset of words
    rnn_words = rnn_words[:min(MAX_WORDS, len(rnn_words))]
    embedding_words = embedding_words[:min(MAX_WORDS, len(embedding_words))]
    attention_words = attention_words[:min(MAX_WORDS, len(attention_words))]

    # Get the GloVe model. This one is used
    model = get_model()

    def append_list(sim_words, words):

        list_of_words = []

        for i in range(len(sim_words)):
            sim_words_list = list(sim_words[i])
            sim_words_list.append(words)
            sim_words_tuple = tuple(sim_words_list)
            list_of_words.append(sim_words_tuple)

        return list_of_words


    input_word = 'positive'
    user_input = [x.strip() for x in input_word.split(',')]

    result_word = []

    for words in user_input:
        sim_words = model.most_similar(words, topn=5)
        sim_words = append_list(sim_words, words)
        result_word.extend(sim_words)

    # Need to adjust the classes.
    labels_rnn = ["rnnz" for word in rnn_words]
    labels_embedding = ["embeddingz" for word in embedding_words]
    labels_attention = ["attentionz" for word in attention_words]
    labels = labels_rnn + labels_embedding + labels_attention

    print("LAbels", labels)
    print("WoWrds", rnn_words + embedding_words + attention_words)

    label_dict = dict([(y, x + 1) for x, y in enumerate(set(labels))])
    color_map = [label_dict[x] for x in labels]
    print("color_map", color_map)


    display_pca_scatterplot_3D(model,['rnn', 'embedding', 'attention'], [rnn_words, embedding_words, attention_words], labels, color_map)
    # display_pca_scatterplot_3D(model, words=["male", "female"])


