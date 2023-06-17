import plotly
import numpy as np
import plotly.graph_objs as go
from sklearn.decomposition import PCA

from model import get_model
from words import get_flatten_words

def display_pca_scatterplot_3D(model, user_input=None, words=None, label=None, color_map=None, topn=10, sample=10):
    print("len words", len(words), len(label), len(color_map), len(user_input))
    topn = len(words)
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

    three_dim = PCA(random_state=0).fit_transform(word_vectors)[:,:3]
    # For 2D, change the three_dim variable into something like two_dim like the following:
    # two_dim = PCA(random_state=0).fit_transform(word_vectors)[:,:2]

    data = []
    count = 0

    for i in range (len(user_input)):

                trace = go.Scatter3d(
                    x = three_dim[count:count+topn,0],
                    y = three_dim[count:count+topn,1],
                    z = three_dim[count:count+topn,2],
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

    trace_input = go.Scatter3d(
                    x = three_dim[count:,0],
                    y = three_dim[count:,1],
                    z = three_dim[count:,2],
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
    # similar_word = ["male", "woman", "female", "programmer", "household"]
    print("Getting similar words")
    similar_word = get_flatten_words()
    print("similar words len", len(similar_word))


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
    print("user_input", user_input)
    result_word = []

    for words in user_input:
        sim_words = model.most_similar(words, topn=5)
        sim_words = append_list(sim_words, words)
        print("simwords", sim_words)
        result_word.extend(sim_words)

    print("Most similar top 5,", result_word)
    # similar_word = [word[0] for word in result_word]
    similarity = [word[1] for word in result_word]
    # similar_word.extend(user_input)
    # print("result_word", similar_word)
    labels = ["class_1" for word in similar_word]
    label_dict = dict([(y, x + 1) for x, y in enumerate(set(labels))])
    color_map = [label_dict[x] for x in labels]
    print('labels', labels, user_input)
    print("SIM Words", sim_words)
    print("label v2", ["rnn" for i in similar_word])
    print("color_map", color_map)
    display_pca_scatterplot_3D(model, ["rnn"], similar_word[:300], ["rnn" for i in similar_word][:300], color_map[:300])
    # display_pca_scatterplot_3D(model, words=["male", "female"])


