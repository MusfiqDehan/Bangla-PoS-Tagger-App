from json.tool import main
from flask import Flask, request, render_template
from pprint import pprint
from data_preprocessors import text_preprocessor as tp
from bangla_postagger import (en_postaggers as ep,
                              bn_en_mapper as bem,
                              translators as trans)

app = Flask(__name__)
# FLASK_APP = app.main.py


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/getpostags', methods=['GET', 'POST'])
def getpostags():
    '''
    For rendering results on HTML GUI
    '''
    if request.method == "POST":
        src_mod = trans.get_translated_digit(src)
        tgt = trans.get_translation(src_mod)
        tgt = tp.decontracting_words(tgt)
        tgt = tgt.replace('rupees', 'takas').replace('Rs', 'takas')

        src = tp.space_punc(src)
        tgt = tp.space_punc(tgt)

        word_mapping = bem.get_word_mapping(
            source=src, target=tgt, model_path="musfiqdehan/bangla-awesome-align")

        nltk_postag = bem.get_nltk_postag(
            source=src, target=tgt, model_path="musfiqdehan/bangla-awesome-align")

        spacy_postag = bem.get_spacy_postag(
            source=src, target=tgt, model_path="musfiqdehan/bangla-awesome-align")

        flair_postag = bem.get_flair_postag(
            source=src, target=tgt, model_path="musfiqdehan/bangla-awesome-align")

    output = {
        'src': src,
        'tgt': tgt,
        'word_mapping': word_mapping,
        'nltk_postag': nltk_postag,
        'spacy_postag': spacy_postag,
        'flair_postag': flair_postag,
    }

    return render_template('index.html', **output)


if __name__ == "__main__":
    app.run(debug=True)
