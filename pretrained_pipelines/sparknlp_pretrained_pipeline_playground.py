from pyspark.sql import SparkSession
from pyspark.ml import Pipeline

from sparknlp.annotator import *
from sparknlp.common import *
from sparknlp.base import *
from sparknlp.pretrained import PretrainedPipeline

import pandas as pd

import sparknlp
spark = sparknlp.start()

import streamlit as st

SPARK_NLP_PIPELINES = ['explain_document_ml',
'explain_document_dl',
'recognize_entities_dl', 
'explain_document_dl_fast',
'onto_recognize_entities_sm',
'onto_recognize_entities_lg',
'match_datetime',
'match_pattern',
'match_chunks',
'match_phrases',
'clean_stop',
'clean_pattern',
'clean_slang',
'check_spelling',
'analyze_sentiment',
'dependency_parse']

DEFAULT_TEXT = "Other than being the king of the north, John Snow is a an english physician and a leader in the development of anaesthesia and medical hygiene. He is considered for being the first one using data to cure cholera outbreak in 1834."

HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem">{}</div>"""

@st.cache(allow_output_mutation=True)
def load_pipeline(name):
    return PretrainedPipeline(name, lang='en')


@st.cache(allow_output_mutation=True)
def process_text(model_name, text, mode='slim'):
    pipeline = load_pipeline(model_name)
    if mode=='slim':
        return pipeline.annotate(text)
    else:
        return pipeline.fullAnnotate(text)

st.sidebar.title("Interactive Spark NLP UI")
st.sidebar.markdown(
    """
Process text with Spark NLP pretrained pipelines and more. Using Spark NLP LightPipelines under the hood.
"""
)

sparknlp_model = st.sidebar.selectbox("Pipeline name", SPARK_NLP_PIPELINES)
model_load_state = st.info(f"Loading pretrained pipeline '{sparknlp_model}'...")
#pipeline = load_pipeline(sparknlp_model)
model_load_state.empty()

#st.markdown("Text to analyze")

text = st.text_area("Text to analyze", DEFAULT_TEXT)

try:
    annotated_text = process_text(sparknlp_model, text, mode='slim')
    full_annotated_text = process_text(sparknlp_model, text, mode='full')[0]
except:
    st.write('error in loading the pipeline !')
    annotated_text={}
    full_annotated_text={}
    pass
#stages = pretrained_pipeline.model.stages

#stages = ['_'.join(s.split('_')[:-1]) for s in stages]

#stages = [s['name'] for s in pretrained_pipeline.model.stages]


import random

def get_color():
    r = lambda: random.randint(100,255)
    return '#%02X%02X%02X' % (r(),r(),r())


def get_onto_NER_html (annotated_text):
    
    light_data=annotated_text
    
    #html_output = '<center><h3>Results of NER Annotation Pipeline</h3></center>'
    #html_output += '<div style="border:2px solid #747474; margin: 5px; padding: 10px">'
    html_output=''
    
    problem_flag = False
    new_problem = []
    problem_list = []
    
    label_list = list(set([i.split('-')[1] for i in light_data['ner'] if i!='O']))
    
    label_color={}
    
    for l in label_list:
        
        label_color[l]=get_color()
        
    
    for index, this_token in enumerate(light_data['token']):

        if light_data['ner'][index]=='O':
            html_output+=this_token + " "
        else:
            color = label_color[light_data['ner'][index].split('-')[1]]
            html_output+='<SPAN style="background-color: {}">'.format(color) + this_token + " </SPAN>"
        

    html_output += '</div>'
    html_output += '<div>Color codes:'

    for l in label_list:
        
        html_output += '<SPAN style="background-color: {}">{}</SPAN>, '.format(label_color[l],l)
   
    return html_output
    
    
if sparknlp_model == 'xx':#'recognize_entities_dl':
    
    st.header("Named Entities")
    st.sidebar.header("Named Entities")
    
    label_set = ['LOC','PER','ORG','MISC']
    
    labels = st.sidebar.multiselect(
        "Entity labels", options=label_set, default=list(label_set)
    )
    html = get_onto_NER_html (annotated_text) #displacy.render(doc, style="ent", options={"ents": labels})
    # Newlines seem to mess with the rendering
    html = html.replace("\n", " ")
    st.write(HTML_WRAPPER.format(html), unsafe_allow_html=True)
    
    st.write('')
    st.write('')

    df = pd.DataFrame({'token':annotated_text['token'], 'label':annotated_text['ner'],
                      'corrected':annotated_text['spell'], 'POS':annotated_text['pos'],
                      'lemmas':annotated_text['lemmas'], 'stems':annotated_text['stems']})
    st.dataframe(df)
    
    
if 'ner' in annotated_text.keys():
    
    st.header("Named Entities")
    st.sidebar.header("Named Entities")
    
    label_set = list(set([i.split('-')[1] for i in annotated_text['ner'] if i!='O']))
    
    #['LOC','PER','ORG','MISC']
    
    labels = st.sidebar.multiselect(
        "Entity labels", options=label_set, default=list(label_set)
    )
    html = get_onto_NER_html (annotated_text) #displacy.render(doc, style="ent", options={"ents": labels})
    # Newlines seem to mess with the rendering
    html = html.replace("\n", " ")
    st.write(HTML_WRAPPER.format(html), unsafe_allow_html=True)
    
    st.write('')
    st.write('')

if sparknlp_model == 'explain_document_dl':
    
    df = pd.DataFrame({'token':annotated_text['token'], 'label':annotated_text['ner'],
                      'corrected':annotated_text['checked'], 'POS':annotated_text['pos'],
                      'lemmas':annotated_text['lemma'], 'stems':annotated_text['stem']})
    st.dataframe(df)
    

elif sparknlp_model == 'explain_document_ml':
    
    df = pd.DataFrame({'token':annotated_text['token'], 
                      'corrected':annotated_text['spell'], 'POS':annotated_text['pos'],
                      'lemmas':annotated_text['lemmas'], 'stems':annotated_text['stems']})
    st.dataframe(df)
    
    
elif sparknlp_model in ['recognize_entities_dl','onto_recognize_entities_sm']:
    
    df = pd.DataFrame({'token':annotated_text['token'], 'label':annotated_text['ner']})
    st.dataframe(df)
    
    
elif sparknlp_model == 'check_spelling':
    
    df = pd.DataFrame({'token':annotated_text['token'],
                      'corrected':annotated_text['checked']})
    st.dataframe(df)
    
elif sparknlp_model == 'check_spelling':
    
    df = pd.DataFrame({'token':annotated_text['token'],
                      'corrected':annotated_text['checked']})
    st.dataframe(df)
    
    
elif sparknlp_model == 'dependency_parse':
    
    df = pd.DataFrame({'token':annotated_text['token'],
                       'pos':annotated_text['pos'],
                       'dep_mod':annotated_text['dep_mod'],
                      'dep_root':annotated_text['dep_root']})
    st.dataframe(df)
    
    
elif sparknlp_model == 'clean_slang':
    
    try:
        df = pd.DataFrame({'token':annotated_text['token'],
                       'normal':annotated_text['normal']})
        st.dataframe(df)
    except:
        pass
    
if 'entities' in annotated_text.keys():
    st.write('')
    st.write('Entities')
    st.write('')
    
    chunks=[]
    entities=[]
    for n in full_annotated_text['entities']:
        
        chunks.append(n.result)
        entities.append(n.metadata['entity']) 
        
    st.dataframe(pd.DataFrame({'chunks':chunks, 'entities':entities}))
    #st.write(annotated_text['entities'])
    
    
if 'sentence' in annotated_text.keys():
    st.write('')
    st.write('Sentences')
    st.write('')
    st.write(annotated_text['sentence'])
    #st.dataframe(pd.DataFrame({'sentences':annotated_text['sentence']}))

if 'sentiment' in annotated_text.keys():
    
    st.write('')
    st.write('Sentiment')
    st.write('')
    st.dataframe(pd.DataFrame({'sentence':annotated_text['sentence'], 'sentiment':annotated_text['sentiment']}))
    
