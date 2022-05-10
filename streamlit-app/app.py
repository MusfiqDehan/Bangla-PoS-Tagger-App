import streamlit as st
from pprint import pprint
from data_preprocessors import text_preprocessor as tp
from bangla_postagger import (en_postaggers as ep,
                              bn_en_mapper as bem,
                              translators as trans)

st.set_page_config(
    page_title="Bangla PoS Tagger | Home",
    page_icon=":sunflower:",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={
        'Get help': None,
        'Report a bug': None,
        'About': '''

        ### Bangla PoS Tagger
        *Version: 0.0.3* \n
        It is a web application for tagging the parts of speech of words in Bangla Language.

        ---

        #### Pos Tags
        1. ***NN:*** noun, singular or mass
        1. ***NNS:*** noun, plural
        1. ***NNP:*** proper noun, singular
        1. ***NNPS:*** proper noun, plural
        1. ***JJ:*** adjective or numeral, ordinal
        1. ***JJR:*** adjective, comparative
        1. ***JJS:*** adjective, superlative
        1. ***RB:*** adverb
        1. ***RBR:*** adverb, comparative
        1. ***RBS:*** adverb, superlative
        1. ***VB:*** verb, base form
        1. ***VBD:*** verb, past tense
        1. ***VBG:*** verb, gerund or present participle
        1. ***VBN:*** verb, past participle
        1. ***VBP:*** verb, non-3rd person singular present
        1. ***VBZ:*** verb, 3rd person singular present
        1. ***CC:*** coordinating conjunction
        1. ***DT:*** determiner
        1. ***EX:*** existential there (like: "there is" ... think of it like "there exists")
        1. ***FW:*** foreign word
        1. ***IN:*** preposition/subordinating conjunction
        1. ***MD:*** modal
        1. ***NCD:*** number cardinal
        1. ***NPD:*** number ordinal
        1. ***POS:*** possessive ending
        1. ***PRP:*** pronoun, personal
        1. ***PRP$:*** pronoun, possessive
        1. ***TO:*** to
        1. ***UH:*** interjection
        1. ***WP:*** pronoun, wh-personal
        1. ***WP$:*** pronoun, possessive wh-personal
        1. ***WRB:*** wh-adverb
        1. ***PUNC:*** punctuation
        ---
        '''
    }
)

st.title('Bangla PoS Tagger')

src = st.text_input("Enter a Bangla Sentence (Max: 5000 Characters):")

if st.button('Find PoS'):
    src_mod = trans.get_translated_digit(src)
    tgt = trans.get_translation(src_mod)
    tgt = tp.decontracting_words(tgt)
    tgt = tgt.replace('rupees', 'takas').replace('Rs', 'takas')

    st.subheader("Translated Sentence:")
    st.write(tgt)

    src = tp.space_punc(src)
    tgt = tp.space_punc(tgt)

    st.subheader("Word Mapping:")
    myresult = bem.get_word_mapping(
        source=src, target=tgt, model_path="musfiqdehan/bangla-awesome-align")
    st.write(myresult[2])

    st.subheader("NLTK PoS Tagged Words:")
    st.write(bem.get_nltk_postag(
        source=src, target=tgt, model_path="musfiqdehan/bangla-awesome-align"))

    st.subheader("Spacy PoS Tagged Words:")
    st.write(bem.get_spacy_postag(
        source=src, target=tgt, model_path="musfiqdehan/bangla-awesome-align"))

    st.subheader("Flair PoS Tagged Words:")
    st.write(bem.get_flair_postag(
        source=src, target=tgt, model_path="musfiqdehan/bangla-awesome-align"))
