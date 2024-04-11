
'''
1. What does this script do?

This script has code to generate the summary of all the testimonies of a certain bill. 
Ideally it collects all the testimonies of a bill along with summary of what the bill is and 
prompts the LLM to summarize all the testimonies, highlight on what points do people agree and disagree with. 


2. How to generate the summaries of testimonies using this script?

To generate the summary of testimonies, we require the "bill_id" of the bill and an API key to call the model.

Here is how you call this script. 
-> Pass in OpenAI "api key" and "bill id" as inputs. As of now this program can only work with 3 bills (H3069, H3121, H711)
python testimony_summary.py <<open ai secret key>> <<bill number>>

Example command to run this program:

python testimony_summary.py my_OpenAI_key H711

Example Output
Residents of Massachusetts have submitted various testimonies regarding a bill that would introduce ranked choice voting (RCV) for local elections 
in cities and towns that opt to adopt it. The testimonies reflect a mix of strong support and opposition to the bill.

Supporters of the bill argue that RCV would improve community consensus, increase voter interest, and lead to more inclusive and competitive elections.
They emphasize the importance of allowing municipalities the freedom to choose their voting methods without state permission and believe that RCV 
would encourage positive campaigning and ensure majority support for elected officials. Proponents also highlight the need for communities to serve 
as laboratories for innovation in democratic processes.

On the other hand, one testimony stands out in opposition, raising concerns about the potential for RCV to dilute majority rule and invite extremism. 
The opposer points out the risks of minority factions gaining disproportionate representation and the possibility of disruptive outliers winning 
elections despite majority opposition. They also criticize the incorrect documentation and description of RCV proposals and the unintended consequences
 of allowing communities to experiment with complex changes to election methods.

Overall, the testimonies show a general trend towards favoring the adoption of RCV, with the belief that it would empower voters and strengthen 
democracy, while also acknowledging the need for careful consideration of the potential drawbacks and implementation challenges
'''

import textwrap
import pandas as pd
import tiktoken
import os
import json
import sys

from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.cache import SQLiteCache
from langchain.globals import set_llm_cache
from langchain.docstore.document import Document
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.summarize import load_summarize_chain
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser


from langchain_community.chat_models import ChatOpenAI
from tagging import *

try: 
    with open('generated_summary.json', 'r') as f: SUMMARY_CACHE = json.load(f)
except Exception as e: SUMMARY_CACHE = {}


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

def find_bills(bill_number, df):
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

def generate_categories(text: str) -> str:

    '''
    This function predicts which category a bill belongs to from pre defined categories by calling the LLM API. 

    Args: 
        test (str): The bill text

    Returns:
        str : Predicted category of the bill

    Example: 
        >>> generate_categories('Bill Text')
        Education
    '''

    category_prompt = """According to this list of category {category}.

        classify this bill {context} into a closest relevant category.

        Do not output a category outside from the list
    """

    prompt = PromptTemplate(template=category_prompt, input_variables=["context", "category"])
    llm = LLMChain(llm = ChatOpenAI(openai_api_key=open_ai_key, temperature=0, model=MODEL_NAME), prompt = prompt)
        
    response = llm.predict(context = text, category = category_for_bill) # grab from tagging.py
    return response

def cache(func):

    '''
    Cache is a decorator that is used for caching the data. Ideally before calling the LLM API, this function checks weather the 
    data(In our case summary) that we want is in the cache or not, if in cache, it skips calling the LLM API and returns it from cache
    else it calls the LLM, gets the data and updates the cache

    Args: 
        func: function object

    Returns: 
        func: function object 
    '''

    def w(arg):
        if arg in SUMMARY_CACHE: return SUMMARY_CACHE[arg]
        else: 
            out = func(arg)
            SUMMARY_CACHE[arg] = out
            with open('generated_summary.json', 'w') as json_file: json.dump(SUMMARY_CACHE, json_file)
            return out
    return w

# @cache
def _generate_summary(bill_id: str, df: pd.DataFrame) -> str:

    '''
    This is a helper function to generate summary of a bill - Specifically this function calls the LLM to generate it.  

    Args: 
        bill_id (str): The ID of the bill
        df (pd.DataFrame): Dataframe that consists of data related to bill and Mass General Law

    Returns: 
        str: LLM generated summary of a bill 

    Examples: 
        >>> df = pd.read_csv('path_to_file')    
        >>> get_summary('H711', df)

        'Conscise summary of the bill.'
    '''
    st()
    bill_c, bill_t, bill_id = find_bills(bill_id, df = df)
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

def get_summary(bill_id: str, mgl_df: pd.DataFrame) -> str:
    '''
    This function generates the summary of a bill. 

    Args: 
        bill_id (str): The ID of the bill
        mgl_df (pd.DataFrame): Dataframe that consists of data related to bill and Mass General Law

    Returns: 
        str: LLM generated summary of a bill 

    Examples: 
        >>> mgl_df = pd.read_csv('path_to_file')    
        >>> get_summary('H711', mgl_df)

        'Conscise summary of the bill.'

    '''

    # This files consists of summaries of each bill. It is generated using run_demo_app.py
    summaries = pd.read_csv('./generated_bills.csv')

    out = summaries.loc[summaries['Bill Number'] == bill_id]['Summarized Bill']
    if len(out) > 0 and False: return out[0]
    else:
        summary = _generate_summary(bill_id, mgl_df)
        return summary

def get_testimonies(bill_id: str, testimonies: pd.DataFrame) -> str:

    '''
    This function formats all the testimonies pertaining to the bill into a single string

    Args: 
        bill_id (str): The ID of the bill
        testimonies (pd.DataFrame): Pandas dataframe object that has the details of bills and testimonies.

    Returns: 
        str: Formatted string of all the summaries

    Examples: 
        >>> get_testimonies('H711')
        '
        testimony_1

        testimony_2
        ....
        testimony_n
        '
    '''

    this_testimoies = list(testimonies.loc[testimonies['bill_id'] == bill_id].content)
    this_testimonies_str = ''
    for i in range(len(this_testimoies)): this_testimonies_str += '\n\n testimony - ' + str(i + 1) + ' \n\n{}'
    return this_testimonies_str.format(*this_testimoies)

def get_formatted_summary_and_testimonies(bill_id: str, mgl_df: pd.DataFrame, testimonies: pd.DataFrame):

    '''
    This functions formats the prompt with bill summary details and all the testimonies. 

    Args: 
        bill_id (str): The ID of the bill for which you want the summary of all the testimonies.
        mgl_df (pd.DataFrame): Pandas dataframe object that has the details of the Mass General Law and specific bills.
        testimonies (pd.DataFrame): Pandas dataframe object that has the details of bills and testimonies.

    Returns: 
        str: Formatted prompt string. 

    Examples: 
        >>> get_testimony_summary('H711', mgl_df, testimonies)
        'Here is the summary of the bill: this bill is so and so
         Here are all the testimonies that users have submitted : 
         testimony_1

         testimony_2
         ....
         testimony_n'
    '''

    formatted_testimonies = get_testimonies(bill_id, testimonies)
    summary = get_summary(bill_id, mgl_df)
    return f'Here is the summary of the bill: {summary} \n Here are all the testimonies that users have submitted : {formatted_testimonies}'

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    '''
    This function takes in a string and calculates how many tokes this string contains based on the model

    Args: 
        string (str): The string we want to find the number of tokens on
        encoding_name (str): The name of the model

    Returns: 
        int: Number of tokens the string contains

    Examples:
        >>> num_tokens_from_string("Hello world, let's test tiktoken.", "gpt-3.5-turbo")
        8
    '''
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def get_testimony_summary(bill_id: str, mgl_df: pd.DataFrame, testimonies: pd.DataFrame) -> str:

    '''
    This functions calls the LLM with a prompt to generate summary of all the testimonies that bill has. 

    Args: 
        bill_id (str): The ID of the bill for which you want the summary of all the testimonies.
        mgl_df (pd.DataFrame): Pandas dataframe object that has the details of the Mass General Law and specific bills.
        testimonies (pd.DataFrame): Pandas dataframe object that has the details of bills and testimonies.

    Returns: 
        str: LLM generated summary of the testimonies. 

    Examples: 
        >>> mgl_df = pd.read_csv('path_to_file')
        >>> get_testimony_summary('H711', mgl_df)
        The Massachusetts bill under consideration introduces ranked choice voting (RCV) for local elections in cities and towns that opt to adopt it. 
        This system allows voters to rank candidates by preference, with votes counted in rounds until a winner is determined. Local ordinances will 
        define the specific rules for RCV, and municipalities must conduct voter education campaigns. They also have the option to revert to their previous
        voting method after four years.

        Residents of Massachusetts have submitted testimonies both in support and opposition to the bill. Proponents argue that RCV improves community
        consensus, increases voter interest, and allows municipalities to choose the best voting method for their communities without state permission. 
        They believe RCV leads to more inclusive and competitive elections and serves as a laboratory for democratic innovation. Supporters also emphasize 
        the need for voter empowerment and the moderation of political extremes.

        On the other hand, one resident, Neil Gordon, opposes the bill, citing misinformation about RCV and potential risks, such as the dilution of 
        majority rule and the possibility of electing extreme minority candidates. He expresses concern about the unintended consequences of RCV in 
        multi-winner elections and the potential for disruptive outliers to win despite majority opposition. 
    '''

    data = get_formatted_summary_and_testimonies(bill_id, mgl_df, testimonies)
    # docs = [Document(page_content=t) for t in text_splitter.split_text(data)]
    llm = ChatOpenAI(temperature=0, model_name = MODEL_NAME)

    prompt_template = """Write a concise summary of testimonies submitted by various residents of the state of Massachusetts for a particular bill:

    The below text consists of summary of the bill followed by each testimony. 

    {text}

    Here are some instructions to follow: 
    1. Please do not use any other information other than what is provided here. 
    2. Do not mention any names.
    3. Do not provide information related to bill that will make the summary redundant. 

    
    CONSCISE SUMMARY:"""

    prompt = PromptTemplate(template = prompt_template, input_variables = ['text'])
    num_tokens = num_tokens_from_string(data, MODEL_NAME)

    if num_tokens < MAX_TOKENS: chain = load_summarize_chain(llm, chain_type = "stuff", prompt = prompt, verbose = True)
    else: chain = load_summarize_chain(llm, chain_type = "map_reduce", map_prompt = prompt, combine_prompt = prompt, verbose = True)

    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(model_name = MODEL_NAME)
    return chain.run([Document(page_content=t) for t in text_splitter.split_text(data)])

def main(bill_id: str) -> str:

    '''
    This is the main function takes in bill_id as input and produces the summary of all the testimonies of that bill as output. 


    Args: 
        bill_id (str): The ID of the bill for which you want the summary of all the testimonies.

    Returns:
        str: LLM generated summary of the testimonies. 

    Examples:
        >>> main('H711')
        The Massachusetts bill under consideration introduces ranked choice voting (RCV) for local elections in cities and towns that opt to adopt it. 
        This system allows voters to rank candidates by preference, with votes counted in rounds until a winner is determined. Local ordinances will 
        define the specific rules for RCV, and municipalities must conduct voter education campaigns. They also have the option to revert to their previous
        voting method after four years.

        Residents of Massachusetts have submitted testimonies both in support and opposition to the bill. Proponents argue that RCV improves community
        consensus, increases voter interest, and allows municipalities to choose the best voting method for their communities without state permission. 
        They believe RCV leads to more inclusive and competitive elections and serves as a laboratory for democratic innovation. Supporters also emphasize 
        the need for voter empowerment and the moderation of political extremes.

        On the other hand, one resident, Neil Gordon, opposes the bill, citing misinformation about RCV and potential risks, such as the dilution of 
        majority rule and the possibility of electing extreme minority candidates. He expresses concern about the unintended consequences of RCV in 
        multi-winner elections and the potential for disruptive outliers to win despite majority opposition.
    '''

    # This CSV file consists of Bills and pertinant sections of Mass General Law
    mgl_df = pd.read_csv("./12_bills_with_mgl_more_sections.csv")

    # This file consists of all the testimonies of each bill. 
    testimonies = pd.read_csv('./maple_testimonies.csv')

    summary = get_testimony_summary(bill_id, mgl_df, testimonies)
    print(summary)
    return summary

if __name__ == '__main__': 
    # Read the arguments
    
    set_llm_cache(SQLiteCache(database_path="../summary.db"))

    open_ai_key, bill_id = sys.argv[1:]
    os.environ['OPENAI_API_KEY'] = open_ai_key 
    MODEL_NAME = 'gpt-4-1106-preview'
    MAX_TOKENS = 4097
    main(bill_id)


   


