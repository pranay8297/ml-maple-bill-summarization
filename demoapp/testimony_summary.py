import textwrap
import pandas as pd
import tiktoken
import os
import json
import sys

from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.docstore.document import Document
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.summarize import load_summarize_chain
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

from langchain_community.chat_models import ChatOpenAI
from ipdb import set_trace as st
from tagging import *

try: 
    with open('generated_summary.json', 'r') as f: SUMMARY_CACHE = json.load(f)
except Exception as e: SUMMARY_CACHE = {}

open_ai_key, bill_id = sys.argv[1:]
os.environ['OPENAI_API_KEY'] = open_ai_key 
MODEL_NAME = 'gpt-4-1106-preview'
MAX_TOKENS = 4097

mgl_df = pd.read_csv("./12_bills_with_mgl_more_sections.csv")
summaries = pd.read_csv('./generated_bills.csv')
testimonies = pd.read_csv('./maple_testimonies.csv')

text_splitter = CharacterTextSplitter.from_tiktoken_encoder(model_name = MODEL_NAME)

bills_to_select = {
    '#H3121': 'An Act relative to the open meeting law',
    '#S2064': 'An Act extending the public records law to the Governor and the Legislature',
    '#H711': 'An Act providing a local option for ranked choice voting in municipal elections',
    '#S1979': 'An Act establishing a jail and prison construction moratorium',
    '#H489': 'An Act providing affordable and accessible high-quality early education and care to promote child development and well-being and support the economy in the Commonwealth',
    '#S2014': 'An Act relative to collective bargaining rights for legislative employees',
    '#S301': 'An Act providing affordable and accessible high quality early education and care to promote child development and well-being and support the economy in the Commonwealth',
    '#H3069': 'An Act relative to collective bargaining rights for legislative employees',
    '#S433': 'An Act providing a local option for ranked choice voting in municipal elections',
    '#H400': 'An Act relative to vehicle recalls',
    '#H538': 'An Act to Improve access, opportunity, and capacity in Massachusetts vocational-technical education',
    '#S257': 'An Act to end discriminatory outcomes in vocational school admissions'
}

def find_bills(bill_number, df = mgl_df):
    """input:
    args: bill_number: (str), Use the number of the bill to find its title and content
    """
    bill = df[df['BillNumber'] == bill_number]['DocumentText']

    try:
         # Locate the index of the bill
        idx = bill.index.tolist()[0]
        # Locate the content and bill title of bill based on idx
        content = df['DocumentText'].iloc[idx]
        bill_title = df['Title'].iloc[idx]
        bill_number = df['BillNumber'].iloc[idx]
        # laws
        # law = df['combined_MGL'].iloc[idx]

        return content, bill_title, bill_number
    
    except Exception as e:
        content = "blank"
        print(e)

def generate_categories(text):

    category_prompt = """According to this list of category {category}.

        classify this bill {context} into a closest relevant category.

        Do not output a category outside from the list
    """

    prompt = PromptTemplate(template=category_prompt, input_variables=["context", "category"])
    llm = LLMChain(llm = ChatOpenAI(openai_api_key=open_ai_key, temperature=0, model=MODEL_NAME), prompt = prompt)
        
    response = llm.predict(context = text, category = category_for_bill) # grab from tagging.py
    return response

def cache(func):
    def w(arg):
        if arg in SUMMARY_CACHE: return SUMMARY_CACHE[arg]
        else: 
            out = func(arg)
            SUMMARY_CACHE[arg] = out
            with open('generated_summary.json', 'w') as json_file: json.dump(SUMMARY_CACHE, json_file)
            return out
    return w

@cache
def _generate_summary(bill_id, df = mgl_df):
    bill_c, bill_t, bill_id = find_bills(bill_id)
    category = generate_categories(bill_c)
    mgl_ref = str(df.loc[df['BillNumber']== bill_id, 'Combined_MGL'].values[0])
    text_splitter = CharacterTextSplitter(chunk_size=4000, chunk_overlap=0)

    if mgl_ref is not None:
        #splits MGL text into chunks of sizes specified in the previous step
        documents = [Document(page_content=x) for x in text_splitter.split_text(mgl_ref)]
    else:
        documents = ""

    vectorstore = Chroma.from_documents(documents, OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()

    template = """You are a trustworthy assistant for question-answering tasks.
            Use the following pieces of retrieved context to answer the question.
            Question: {question}
            Context: {context}
            Answer:
            """

    prompt = PromptTemplate.from_template(template)
    llm = ChatOpenAI(temperature=0, model = MODEL_NAME, model_kwargs={'seed': 42}) 
    rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
    query = f""" Can you please explain what the following MA bill means to a regular resident without specialized knowledge? 
            Please provide a one paragraph summary in 4 sentences. Please be simple, direct and concise for the busy reader. Make bullet points if possible.
            Note that the bill refers to specific existing chapters and sections of the Mass General Laws. Use the information from those chapters and sections in your context to construct your summary
            Summarize the bill that reads as follows:\n{bill_c}\n\n
            
            After generating summary, output Category: {category}.
            Then, output top 2 tags in this specific category from the list of tags {tags_for_bill} that are relevant to this bill. \n"""
    return rag_chain.invoke(query)

def get_summary(bill_id):
    out = summaries.loc[summaries['Bill Number'] == bill_id]['Summarized Bill']
    if len(out) > 0: return out[0]
    else:
        summary = _generate_summary(bill_id)
        return summary

def get_testimonies(bill_id):
    this_testimoies = list(testimonies.loc[testimonies['bill_id'] == bill_id].content)
    this_testimonies_str = ''
    for i in range(len(this_testimoies)): this_testimonies_str += '\n\n testimony - ' + str(i + 1) + ' \n\n{}'
    return this_testimonies_str.format(*this_testimoies)

def get_formatted_summary_and_testimonies(bill_id):
    formatted_testimonies = get_testimonies(bill_id)
    summary = get_summary(bill_id)
    return f'Here is the summary of the bill: {summary} \n Here are all the testimonies that users have submitted : {formatted_testimonies}'

def num_tokens_from_string(string: str, encoding_name: str) -> int:    
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def get_testimony_summary(bill_id):
    data = get_formatted_summary_and_testimonies(bill_id)
    # docs = [Document(page_content=t) for t in text_splitter.split_text(data)]
    llm = ChatOpenAI(temperature=0, model_name = MODEL_NAME)

    prompt_template = """Write a concise summary of testimonies submitted by various residents of the state of Massachusetts for a particular bill:

    {text}

    Please do not use any other information other than what is provided here. 

    CONSCISE SUMMARY:"""

    prompt = PromptTemplate(template = prompt_template, input_variables = ['text'])
    num_tokens = num_tokens_from_string(data, MODEL_NAME)

    if num_tokens < MAX_TOKENS:
      chain = load_summarize_chain(llm, chain_type = "stuff", prompt = prompt, verbose = True)
    else:
      chain = load_summarize_chain(llm, chain_type = "map_reduce", map_prompt = prompt, combine_prompt = prompt, verbose = True)

    return chain.run([Document(page_content=t) for t in text_splitter.split_text(data)])

if __name__ == '__main__': print(get_testimony_summary(bill_id))
