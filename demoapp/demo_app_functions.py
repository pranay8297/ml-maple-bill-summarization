"""
This code implements text summarization, category selection and tagging bills using a single prompt
MGL sections in 'all_bills_with_mgl.pq' files were generated using extract_mgl_section.py 
For token size > 120K,  the code impelemtns OpenAIEmbeddings and Vectorstore to split the MGL text into chunks of len = 90K for vector embeddings
For token size < 120K, all the documents including, bill text, MGL, chapter and section names and bill title are passed into a single prompt. 

"""

import json
import os
import pandas as pd
import tiktoken
import streamlit as st
import urllib.request
import chromadb

from chromadb.config import Settings

from langchain.globals import set_llm_cache
from langchain.prompts import PromptTemplate
from langchain_community.cache import SQLiteCache
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_community.callbacks import get_openai_callback
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_text_splitters import TokenTextSplitter
from langchain.docstore.document import Document
from langchain.chains.combine_documents import create_stuff_documents_chain

from operator import itemgetter
from pathlib import Path
from rouge_score import rouge_scorer
from sidebar import sbar
from sidebar import sbar
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import CrossEncoder
from typing import Tuple, List

from extract_mgl_sections import extract_sections, query_section_text_all_bills
from tagging import *

GPT_MDOEL_VERSION = 'gpt-4o-mini'
MAX_TOKEN_LIMIT = 128000

CHROMA_DB_PATH = "./databases/chroma_db"
LLM_CACHE = Path("./databases/llm_cache.db")

API_KEY = "" #Please Enter your API key here

def set_page_config():
    """
    Configures the Streamlit page with a custom title and layout, and displays a custom title for the app.
    """
    st.set_page_config(page_title="Summarize and Tagging MA Bills", layout='wide')
    st.title('Summarize Bills')
    sbar()

def get_mgl_sections_file_all_bills() -> pd.DataFrame:
    """
    Retrieves a Parquet file containing sections of the Massachusetts General Laws (MGL) associated with various bills.
    
    all_bills_with_mgl.pq' file was generated using extract_mgl_sections.py and then saving the resulting pandas dataframe into parque format.

    This function checks for the presence of a file named 'all_bills_with_mgl.pq' in the 'demoapp' directory. If the file exists, it is read into a pandas DataFrame and returned. If the file does not exist, the function attempts to download it from a specified Google Drive link. The download may take a few minutes. Once downloaded, the file is again read into a pandas DataFrame and returned.
    
    Returns:
        pandas.DataFrame: A DataFrame containing the data from the 'all_bills_with_mgl.pq' file.

    Raises:
        FileNotFoundError: If the file cannot be downloaded or read.
    """
    if os.path.isfile("demoapp/all_bills_with_mgl.pq"): # this file was generated using functions in the extract_mgl_sections.py
        return pd.read_parquet("demoapp/all_bills_with_mgl.pq")
    else:
        print("May take a few minutes")
        urllib.request.urlretrieve("https://drive.google.com/file/d/1eYMmxW4gLyh7Zxh8BBJTvMOUNtdbK5B3/view?usp=share_link", "demoapp/all_bills_with_mgl.pq")
        # urllib.request.urlretrieve("https://munira.blob.core.windows.net/public/all_bills_with_mgl.pq", "demoapp/all_bills_with_mgl.pq")
        return pd.read_parquet("demoapp/all_bills_with_mgl.pq")

def get_selected_bill_info_all_bills(df) -> tuple[str, str, str]:
    """
    Retrieves the number, title, and text of a bill selected via Streamlit UI from a DataFrame.

    Args:
    - df (pd.DataFrame): DataFrame with 'Bill Info' (format "BillNumber: BillTitle") and 'DocumentText'.

    Returns:
    - tuple[str, str, str]: Selected bill's number, title, and text. Returns "blank" for text if not found.

    The function assumes the DataFrame has the necessary columns.
    """
    
    bill_info = df["Bill Info"].values

    # creates a dictionary of bill number and title from the selected bill
    bills_to_select = {parts[0]: parts[1] for s in bill_info if s is not None and (parts := s.split(": "))}

    selectbox_options = [f"{number}: {title}" for number, title in bills_to_select.items()]
    option = st.selectbox(
        'Select a Bill',
        selectbox_options
    )

    # Extracting the bill number from the selected option
    selected_bill_num = option.split(":")[0]
    selected_bill_title = option.split(":")[1]

    try:
        selected_bill_text = df.loc[df['BillNumber']== selected_bill_num, 'DocumentText'].values[0]
    except Exception as e:
        selected_bill_text = "blank"
        st.error(f"Cannot find such bill from the source{e}")

    return selected_bill_num, selected_bill_title, selected_bill_text

def get_selected_bill_info_12_bills(df: pd.DataFrame) -> Tuple[str, str, str]:
    """
    Displays a select box with bill options and returns the selected bill's number, title, and text from the given DataFrame.
    
    Args:
    - df (DataFrame): The DataFrame containing bills information.
    
    Returns:
    - tuple: Selected bill's number, title, and text.
    """
    
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
    selectbox_options = [f"{number}: {title}" for number, title in bills_to_select.items()]
    option = st.selectbox(
        'Select a Bill',
        selectbox_options
    )

    # Extracting the bill number from the selected option
    selected_bill_num = option.split(":")[0][1:]
    selected_bill_title = option.split(":")[1]
    
    try:
        selected_bill_text = df.loc[df['BillNumber']== selected_bill_num, 'DocumentText'].values[0]
    except Exception as e:
        selected_bill_text = "blank"
        st.error(f"Cannot find such bill from the source {e}")

    return selected_bill_num, selected_bill_title, selected_bill_text

def get_committee_info(committee_file_name:str, bill_number:str):
    """
    Extracts committee information specific to the bill number provided

    Args:
        committee_file_name (str): name of the file containing commmittee info
        bill_number (str): bill number

    Returns:
        str: committee name and information concatenated together
    """
    if os.path.isfile(committee_file_name):
        df = pd.read_parquet(committee_file_name)
        try:
            committee_name, description = df.loc[df['BillNumber']== bill_number, ['CommitteeName', 'Description']].values[0]
            committee_info = f"{committee_name}: {description}"
        except Exception as e:
            print("Committee Info Not Available")
        return committee_info
    else: 
        return 'None: None'

def get_chap_sec_names(df: pd.DataFrame, bill_number: str, mgl_names_file_path: str) -> str:
    """
    Fetches chapter and section names for a given bill number from a DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame containing bill and chapter-section information.
        bill_number (str): The number of the bill for which chapter and section names are requested.
        mgl_names_file_path (str): path for the file containing chapter and section names
    
    Returns:
        str: All chapter and section names pairs concatenated together.
        
    The function assumes the DataFrame has the necessary columns.
    """
    names_df = pd.read_parquet(mgl_names_file_path)
    chap_sec_lists = df[df['BillNumber'] == bill_number]['Chapter-Section List']  

    names = []

    for lists in chap_sec_lists:
        for tup in lists:
            chap, sec = tup
            try:
                chapter_name = names_df[(names_df["Chapter_Number"] == chap) & (names_df["Section_Number"] == sec)]['Chapter'].values[0]
                section_name = names_df[(names_df["Chapter_Number"] == chap) & (names_df["Section_Number"] == sec)]['Section Name'].values[0]
                names.append({chap: chapter_name, sec: section_name})
            except Exception as e:
                continue
        mgl_names = ", ".join([f"{key}: {value}" for dct in names for key, value in dct.items()])

        return mgl_names

def get_chap_sec_names_internal(chap_sec_lists: list, mgl_names_file_path: str = "./chapter_section_names.pq") -> str:
    """
    Fetches chapter and section names for a given bill number from a local parquet file.
    
    TODO delete this function after we setup a robust database backend with the MGL data.
    
    Args:
        chap_sec_lists (list): list of tuples containing chapter number and section numbers.
        mgl_names_file_path (str): path for the file containing chapter and section names
    
    Returns:
        str: All chapter and section names pairs concatenated together.
        
    The function assumes the DataFrame has the necessary columns.

    """
    names_df = pd.read_parquet(mgl_names_file_path)
    names = {}

    # for lists in chap_sec_lists:
    for tup in chap_sec_lists:
        chap, sec = tup
        try:
            chapter_name = names_df[(names_df["Chapter_Number"] == chap) & (names_df["Section_Number"] == sec)]['Chapter'].values[0]
            section_name = names_df[(names_df["Chapter_Number"] == chap) & (names_df["Section_Number"] == sec)]['Section Name'].values[0]
            names[chapter_name] = section_name
        except Exception as e:
            continue

    return ", ".join([f"{key}: {value}" for key, value in names.items()])

def get_MGL_for_bill(df:pd.DataFrame, bill_number:str) -> str:
    """
    Outputs MGL section text from the dataframe referenced in the selected bill
    Args:
        df (pd.DataFrame): DataFrame containing bill and MGL texts
        bill_number: str:Bill number of the selected bill

    Returns:
        str: _description_
    """
    mgl_ref = df.loc[df['BillNumber']== bill_number, 'Combined_MGL'].values[0]
    return mgl_ref

def count_tokens(bill_title:str, bill_text:str, mgl_ref:str, mgl_names:str, committee_info:str):
    """
    Outputs the number of tokens for the given documents

    Args:
        bill_title (str): Title of the bill.
        bill_text (str): Text content of the bill
        mgl_ref (str):  Mass General Law Section text, if any.
        mgl_names (str): List of Mass General Law chapter and section names
        committee_info (str): Committee name and description

    Returns:
        int: token_count
    """
    
    encoding = tiktoken.encoding_for_model(GPT_MDOEL_VERSION)

    text = str(bill_title) + str(bill_text) + str(mgl_ref) + str(mgl_names) + str(committee_info)
    token_count = len(encoding.encode(text))
    print(f"The text contains {token_count} tokens.")
    return token_count

def generate_response_large_documents(bill_title: str, bill_text: str, mgl_ref: str, mgl_names: str, committee_info: str, llm) -> str:
    """
    Generates a detailed response for large documents related to MA bills.
    
    Args:
        bill_title (str): Title of the bill.
        bill_text (str): Text content of the bill.
        mgl_ref (str): Mass General Law Section text, if any.
        mgl_names (str): List of Mass General Law chapter and section names.
        llm: Language model for generating responses.
    
    Returns:
        str: Detailed response generated by the language model.
    """
    #Splitting the text and using vector embeddings when document size is > 90000 words for large document

    text_splitter = CharacterTextSplitter(chunk_size = 90000, chunk_overlap = 1000)
    documents = [Document(page_content=x) for x in text_splitter.split_text(mgl_ref)]
    vectorstore = Chroma.from_documents(documents, OpenAIEmbeddings())
    mgl_documents = vectorstore.as_retriever()

    template = """You are a trustworthy assistant for question-answering tasks.
        Use the following pieces of retrieved context to answer the question.
        Question: {question}
        Context: {context}
        Answer:
        """

    prompt = PromptTemplate.from_template(template)

    rag_chain = (
            {"context": mgl_documents, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
    
    query = f""" Can you please explain what the following MA bill means to a regular resident without specialized knowledge? 
            Please provide a one paragraph summary in 4 sentences. Please be simple, direct and concise for the busy reader. Make bullet points if possible.
            Note that the bill refers to specific existing chapters and sections of the Mass General Laws. Use the information from those chapters and sections in your context to construct your summary
            Summarize the bill that reads as follows:\n{bill_text}\n and is titled: {bill_title}\n
            
            After generating summary, classify the bill according to this list of {category_for_bill} to the closest relevant category. Do not output categories outside of the list. \n
            Then, output top 3 tags in this specific category from the list of tags {tags_for_bill} that are relevant to this bill. \n
            Do not output the tags outside from the list. \n
            
            Use the information from those chapters and sections in your context as well as the corresponding {mgl_names} of those chapter and sections to construct your summary, generating categories and tags.\n 
            
            Also use committee information {committee_info} if available.\n

            Response Format:\nSummary:\nSUMMARY\nCategory:\nCATEGORY\nTags:\nTAGS")
            """
    with get_openai_callback() as cb:
        response = rag_chain.invoke(query)
        st.write(cb.total_tokens, cb.prompt_tokens, cb.completion_tokens, cb.total_cost)
    return query, response
       
def generate_response_small_documents(bill_title: str, bill_text: str, mgl_ref: str, mgl_names: str, committee_info: str, llm: object) -> str:
    """
    Generates a response for small documents by summarizing a bill and its relevance to Mass General Laws.
    
    Args:
        bill_title (str): Title of the bill.
        bill_text (List[str]): List of text segments from the bill.
        mgl_ref (List[str]): List of Mass General Law references.
        mgl_names (List[Dict[str, str]]): List of dictionaries containing names of the chapters and sections of Mass General Laws.
        llm (Any): Language model used for generating responses.

    Returns:
        Any: The response from the language model based on the generated prompt.
    """

    mgl_documents = [Document(page_content=x) for x in mgl_ref]
    print("using demo_app_copy file")
    try:
        bill_text = [Document(page_content=x) for x in bill_text]
    except Exception as e:
        print("No Bill Text")

    query = """
            Can you please explain what the following MA bill means to a regular resident without specialized knowledge? Please provide a one paragraph summary in 4 sentences. 
            Please be simple, direct and concise for the busy reader. Make bullet points if possible.
            After generating summary, classify the bill according to the list of categories below to the closest relevant category. Do not output categories outside of the list. 
            Then, output the top 3 tags in this specific category from the list of tags that are relevant to this bill. Do not output the tags outside from the list. 
            Note that the bill refers to specific existing chapters and sections of the Mass General Laws. Use the corresponding names of Mass General Law chapter and sections for constructing your summary, generating categories and tags.\n

            Response Format:\n
            Summary: [SUMMARY]\n
            Category: [CATEGORY]\n
            Tags: [TAGS]\n

            The bill title is: \"{title}"\

            The bill text is: \"{context}"\

            The relevant section names are: \"{names}"\

            The relevant section text is: \"{mgl_sections}"\
            
            The relevant committee information: \"{committee_info}"\

            The set of categories is: \"{categories}"\

            The set of tags is: \"{tags}"\
            """

    prompt = PromptTemplate.from_template(query)
    
    chain = create_stuff_documents_chain(llm, prompt)

    with get_openai_callback() as cb:
        response = chain.invoke({"context": bill_text, "mgl_sections": mgl_documents, "title": bill_title, "names": mgl_names, "categories": category_for_bill, "tags": tags_for_bill, "committee_info": committee_info})
        st.write(cb.total_tokens, cb.prompt_tokens, cb.completion_tokens, cb.total_cost)
    return query, response
    
def generate_response(df: pd.DataFrame, bill_number: str, bill_title: str, bill_text: str, mgl_ref: str, committee_file_name:str, mgl_names_file_name:str) -> str:
    """
    Determines document size and generates an appropriate response for the bill.
    
    Args:
        df (pd.DataFrame): DataFrame containing bill information.
        bill_number (str): Number of the bill.
        bill_title (str): Title of the bill.
        bill_text (str): Text content of the bill.
    
    Returns:
        str: Generated response for the bill.
    """
    API_KEY = st.session_state["OPENAI_API_KEY"]
    os.environ['OPENAI_API_KEY'] = API_KEY

    llm = ChatOpenAI(openai_api_key=API_KEY, temperature=0, model=GPT_MDOEL_VERSION, model_kwargs={'seed': 42})  

    committee_info = get_committee_info(committee_file_name, bill_number)
    mgl_names = get_chap_sec_names(df, bill_number, mgl_names_file_name)
    num_tokens = count_tokens(bill_title, bill_text, mgl_ref, mgl_names, committee_info)
    #Context size for 'gpt-4o' is 128K; Need to break up the documents and use vector embenddings when the word length of the documents is large
    #Chose 90K cut off for large docs because: 1 token = 0.75 words --> 90000 words = ~120K words + leaving  room for response
    
    if bill_text is None or str(bill_text).strip().lower() =="nan" or bill_text == "":
        response = f"Bill {bill_number} is not available for summarization"
        query = ""
    elif (num_tokens) < 120000:
        query, response = generate_response_small_documents(bill_title, bill_text, mgl_ref, mgl_names, committee_info, llm)
    else:
        query, response = generate_response_large_documents(bill_title, bill_text, mgl_ref, mgl_names, committee_info, llm)

    return query, response

def parse_response(response:str) -> Tuple[str, str, str]:
    """
    Parses a response string to extract the summary, category, and tags.

    The function expects the response string to be in a specific format, with sections 
    delineated by specific markers: "[SUMMARY]" for the summary section, "Category:" for 
    the category section, and "Tags:" for the tags section. Each section is expected to 
    be separated by newline characters. Tags are further processed to be numbered and cleaned 
    of any leading or trailing hyphens and spaces.

    Args:
    - response (str): The response string containing the summary, category, and tags in 
                      the expected format.

    Returns:
    - Tuple[str, str, str]: A tuple containing the summary, category, and formatted tags 
                            as strings.

    Raises:
    - ValueError: If the response does not contain the expected markers for summary, category,
                  or tags, indicating that the input does not conform to the expected format.
    
    Example:
    >>> response = "[SUMMARY] Example summary text\\nCategory: Example Category\\nTags:\\n- Tag1\\n- Tag2"
    >>> parse_response(response)
    ('Example summary text', 'Example Category', '1. Tag1\\n2. Tag2'\\n3. Tag3)

    Note:
    - The function assumes that the input string strictly follows the expected format. 
      Variations in the format may lead to incorrect parsing or extraction of information.
      """
    
    if not response:
        raise ValueError("The response is empty.")
    if response:
            # Extracting summary
            print(response)
            summary_start = response.find("[SUMMARY]") + 1
            summary_end = response.find("\nCategory:")
            summary = response[summary_start:summary_end].strip()

            # Extracting category
            category_start = summary_end+ len("\nCategory:")
            category_end = response.find("\nTags:")
            category = response[category_start:category_end].strip()
            
            # Extracting tags
            tags_start = category_end + len("\nTags:")
            tags = response[tags_start:].strip()
            tags = response[tags_start:].strip().split("\n")  
            tags = [f"{i + 1}. {tag.strip('- ').strip()}" for i, tag in enumerate(tags)]
            tags = '\n'.join(tags)

    return summary, category, tags

def update_csv(bill_num: str, title: str, summarized_bill: str, category: str, tag: List[str], csv_file_path: str) -> None:
    """
    Updates or appends bill information to a CSV file.
    
    Args:
        bill_num (str): Bill number.
        title (str): Bill title.
        summarized_bill (str): Summarized text of the bill.
        category (str): Category of the bill.
        tag (List[str]): Tags associated with the bill.
        csv_file_path (str): Path to the CSV file.
    """
    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        # If the file does not exist, create a new DataFrame
        df = pd.DataFrame(columns=["Bill Number", "Bill Title", "Summarized Bill", "Category", "Tags"])
    
    mask = df["Bill Number"] == bill_num
    if mask.any():
        df.loc[mask, "Bill Title"] = title
        df.loc[mask, "Summarized Bill"] = summarized_bill
        df.loc[mask, "Category"] = category
        df.loc[mask, "Tags"] = tag
    else:
        new_bill = pd.DataFrame([[bill_num, title, summarized_bill, category, tag]], columns=["Bill Number", "Bill Title", "Summarized Bill", "Category", "Tags"])
        df = pd.concat([df, new_bill], ignore_index=True)
    
    df.to_csv(csv_file_path, index=False)
    return df

def set_openai_api_key():
    '''
    Function to add Open AI API Key to environment variable. 

    This function assumes that API Key is already set.
    In other cases please uncomment the below line and replace API_KEY with your token
    '''
    pass
    # os.environ['OPENAI_API_KEY'] = API_KEY

def set_my_llm_cache(cache_file: Path=LLM_CACHE) -> SQLiteCache:
    """
    Set an LLM cache, which allows for previously executed completions to be
    loaded from disk instead of repeatedly queried.
    """
    set_llm_cache(SQLiteCache(database_path = cache_file))

class BillDetails:

    '''
    A class to store all the details pertaining to a bill. 
    '''

    def __init__(self, bill_id: str, bill_title: str, bill_text: str, mgl_ref: str, committee_info: str, mgl_names: str):
        self.bill_id = bill_id
        self.bill_title = bill_title
        self.bill_text = bill_text
        self.mgl_ref = mgl_ref
        self.committee_info = committee_info
        self.mgl_names = mgl_names

class LLMResults: 

    '''
    A class to store the results of the LLM.
    '''

    def __init__(self, query: str, response: str):
        self.query, self.response = query, response

def extract_bill_context(bill_text) -> str: 



    sections = extract_sections(bill_text)
    mgl_list, empty_responses = query_section_text_all_bills(sections)

    combined_mgl = ' '.join(mgl_list) if len(mgl_list) != 0 else "None"
    mgl_names = get_chap_sec_names_internal(sections)
    return combined_mgl, mgl_names

def get_summary_api_function(bill_id: str, bill_title: str, bill_text: str) -> dict:

    '''
    This function takes in bill id, bill title and bill text as inputs and extracts relevant mgl section text and passes all this information
    to an LLM to generate summary of a bill  

    Arguments: 
    
    bill_id (str): ID of the bill
    bill_title (str): Bill title
    bill_text (str): Contents of the bill

    Returns: 
        A dict of status_code and an response 
        status_code can take these following values {1: Success, -1: Necessary details not found}

    '''

    # extract relevant mgl text
    combined_mgl, mgl_names = extract_bill_context(bill_text)
    
    # create bill_details object
    bill_details = BillDetails(
        bill_id = bill_id,
        bill_title = bill_title, 
        bill_text = bill_text, 
        mgl_ref = combined_mgl, 
        committee_info = 'None:None', 
        mgl_names = mgl_names, 
    )

    # call the summary function
    status_code, results = get_summary(bill_details)

    # return response attribute of returned value
    if status_code != 1: 
        return {'status': status_code, 'summary': ''}
    else: 
        return {'status': status_code, 'summary': results.response}

def get_tags_api_function(bill_id: str, bill_title: str, bill_text: str) -> dict:
    '''
    This function takes in bill id, bill title and bill text as inputs and extracts relevant mgl section text and passes all this information
    to an LLM to generate tags for a bill  

    Arguments: 
    
    bill_id (str): ID of the bill
    bill_title (str): Bill title
    bill_text (str): Contents of the bill

    Returns: 
        A dict of status_code and an response
        status_code can take these following values {1: Success, -1: Necessary details not found}

    '''

    # extract relevant mgl text
    combined_mgl, mgl_names = extract_bill_context(bill_text)
    
    # create bill_details object
    bill_details = BillDetails(
        bill_id = bill_id,
        bill_title = bill_title, 
        bill_text = bill_text, 
        mgl_ref = combined_mgl, 
        committee_info = 'None:None', 
        mgl_names = mgl_names, 
    )

    # call the summary function
    status_code, results = get_tags(bill_details)

    # return response attribute of returned value
    if status_code != 1: 
        return {'status': status_code, 'tags': []}
    else: 
        return {'status': status_code, 'tags': results.response}

def get_llm_call_type(bill_details: BillDetails) -> str:
    """
    This function calculates number of tokens and decides on weather to use RAG or not. It reutrns a string output
    that specifies how to call the LLM. 

    Args: 
        bill_details (BillDetails): object consisting of bill_text, bill_title, mgl_ref, commottee_info, mgl_names

    Returns: 
        str: 'large' or 'small' depeneding upon token count

    """

    bill_text = getattr(bill_details, "bill_text")
    bill_title = getattr(bill_details, "bill_title")
    mgl_ref = getattr(bill_details, "mgl_ref")
    committee_info = getattr(bill_details, "committee_info")
    mgl_names = getattr(bill_details, "mgl_names")

    num_tokens = count_tokens(bill_title, bill_text, mgl_ref, mgl_names, committee_info)
    
    return 'small' if num_tokens < MAX_TOKEN_LIMIT - 5000 else 'large'

def get_category_tags(categories: List) -> List:

    """
    This function takes in list of categories and returns tags pertinant to that specifc categories only. 

    Args: 

        categories (List(str)): List of category strings.

    Returns: 
        List of all tags specific to those of categories.  
    """

    tags_tuple = itemgetter(*set.intersection(set(categories), set(new_categories_for_bill_list)))(new_tags_for_bill_dict)
    
    if isinstance(tags_tuple, list): return tags_tuple

    category_tags = []
    for cts in tags_tuple: category_tags += cts
    return category_tags

def get_summary(bill_details: BillDetails) -> tuple[int, LLMResults]:
    '''
    This function takes in bill details object (bill title, bill text and reference mgl section text) and summarizes the bill. 

    Arguments: 

    bill_details (BillDetails): Object containing information about the bill - bill_text, bill_title, mgl_ref, commottee_info, mgl_names

    Returns: 
        A tuple of status_code and an LLMResults object containing query, response from the LLM
        status_code can take these following values {1: Success, -1: Necessary details not found}

    '''

    if not all(hasattr(bill_details, attr) for attr in ("bill_text", 'bill_title', 'mgl_names', 'committee_info')): 
        return -1, LLMResults()

    set_my_llm_cache()
    llm_call_type = get_llm_call_type(bill_details)

    query = get_query_for_summarization(bill_details, llm_call_type)
    return 1, call_llm(bill_details, query, llm_call_type)

def get_tags(bill_details: BillDetails) -> tuple[int, LLMResults]:
    '''
    This function takes in bill details object (bill title, bill text and reference mgl section text) and tags the bill. 

    Arguments: 

    bill_details (BillDetails): Object containing information about the bill - bill_text, bill_title, mgl_ref, commottee_info, mgl_names

    Returns: 
        A tuple of status_code and an LLMResults object containing query, response from the LLM
        status_code can take these following values {1: Success, -1: Necessary details not found}

    '''

    if not all(hasattr(bill_details, attr) for attr in ("bill_text", 'bill_title', 'mgl_names', 'committee_info')): 
        return -1, LLMResults()

    set_my_llm_cache()
    llm_call_type = get_llm_call_type(bill_details)

    query_1 = get_query_for_categorizing(bill_details, llm_call_type)
    category_response = call_llm(bill_details, query_1, llm_call_type)
    categories = extract_categories_tags(category_response.response)
    category_tags = get_category_tags(categories)
    query_2 = get_query_for_tagging(bill_details, category_tags, llm_call_type)
    tag_response = call_llm(bill_details, query_2, llm_call_type)

    # parses the response from LLM and removes hallucinated tags
    tag_response.response = list(set(extract_categories_tags(tag_response.response)) & set(category_tags))

    return 1, tag_response

def extract_categories_tags(response: str):
    response = response.split('#')
    return [i.strip() for i in response]

def prepare_invoke_dict(bill_details: BillDetails) -> dict:

    """
    This function prepares the dict object that is used in chain.invoke function to call the LLM with prompt and 
    required details. 

    Args: 
        bill_details (BillDetails): Object containing information about the bill - bill_text, bill_title, mgl_ref, commottee_info, mgl_names

    Returns: 
        dict object containing all the necessary keys and values required for invoke call. 

    """
    text_splitter = CharacterTextSplitter(chunk_size = 90000, chunk_overlap = 1000)

    return {
                "title": bill_details.bill_title, 
                "context": [Document(page_content = f"```{x}```") for x in text_splitter.split_text(bill_details.bill_text)],
                "names": bill_details.mgl_names, 
                "mgl_sections": [Document(page_content = f"```{x}```") for x in text_splitter.split_text(bill_details.mgl_ref)],
                "committee_info": bill_details.committee_info
            }

def get_query_for_summarization(bill_details: BillDetails, llm_call_type: str) -> str:

    """
        
    This functions prepares a prompt based on the call type (small: No use of RAG, large: Use RAG) for bill summarization

    Args: 
        bill_details (BillDetails): Object containing information about the bill - bill_text, bill_title, mgl_ref, commottee_info, mgl_names
        llm_call_type (str): This argument can take 2 values ("small": No use of RAG, "large": Use RAG) 

    Returns: 
        str: Query that includes task, context and instructions on summary generation

    """

    
    if llm_call_type == 'large':
        query = f""" 
                    Can you please explain what the following MA bill means to a regular resident without specialized knowledge? \n
                    Please provide a one paragraph summary in a maximum of 4 sentences. Please be simple, direct and concise for the busy reader. \n
                    Please use politically neutral language, keeping in mind that readers may have various ideological perspectives. \n
                    Make bullet points if possible. 

                    Note that the bill refers to specific existing chapters and sections of the Mass General Laws (MGL). Use the corresponding names of Mass General Law chapter and sections for constructing your summary.\n

                    The bill title is: {getattr(bill_details, "bill_title")}\n

                    The bill text is: {getattr(bill_details, "bill_text")}\n  

                    The relevant section names are: {getattr(bill_details, "mgl_names")}\n
                    
                    The relevant committee information if available: {getattr(bill_details, "committee_info")}\n

                    INSTRUCTIONS: 

                    1. Only provide Summary, no other details are required. \n
                    2. Do not provide tags or other extraneous text besides the summary. \n    
                    3. Do not cite the title of the bill - the reader will already know that \n
                    4. Do not cite specific section, chapter or title numbers of the MGL - the reader will not know what those sections are. \n
                    5. Do not reference that this is a “MA” or “Massachusetts” bill - the reader will already know that. \n
                    6. If referencing dates or other provisions of the bill, say that "this would happen if the bill is passed" rather than "this will happen". \n

                    RESPONSE FORMAT:\n\n                Summary: [SUMMARY]
                    """    

    else: 
        query = """
                Can you please explain what the following MA bill means to a regular resident without specialized knowledge? \n
                Please provide a one paragraph summary in a maximum of 4 sentences. Please be simple, direct and concise for the busy reader. \n
                Please use politically neutral language, keeping in mind that readers may have various ideological perspectives. \n
                Make bullet points if possible. 

                Note that the bill refers to specific existing chapters and sections of the Mass General Laws (MGL). Use the corresponding names of Mass General Law chapter and sections for constructing your summary.\n

                The bill title is: \"{title}"\

                The bill text is: \"{context}"\

                The relevant section names are: \"{names}"\

                The relevant section text is: \"{mgl_sections}"\
                
                The relevant committee information if available: \"{committee_info}"\

                INSTRUCTIONS: 

                1. Only provide Summary, no other details are required. \n
                2. Do not provide tags or other extraneous text besides the summary. \n    
                3. Do not cite the title of the bill - the reader will already know that \n
                4. Do not cite specific section, chapter or title numbers of the MGL - the reader will not know what those sections are. \n
                5. Do not reference that this is a “MA” or “Massachusetts” bill - the reader will already know that. \n
                6. If referencing dates or other provisions of the bill, say that "this would happen if the bill is passed" rather than "this will happen". \n

                RESPONSE FORMAT:\n\n                Summary: [SUMMARY]
                """
        bill_details.invoke_dict = prepare_invoke_dict(bill_details)

    return query

def get_query_for_categorizing(bill_details: BillDetails, llm_call_type: str) -> str: 

    """
        
    This functions prepares a prompt based on the call type (small: No use of RAG, large: Use RAG) for bill categoruzation

    Args: 
        bill_details (BillDetails): Object containing information about the bill - bill_text, bill_title, mgl_ref, commottee_info, mgl_names, category_list(If new categories things are added)
        llm_call_type (str): This argument can take 2 values ("small": No use of RAG, "large": Use RAG) 

    Returns: 
        str: Query that includes task, context and instructions on bill categorization

    """

    if llm_call_type == 'large':
        query = f""" Your job is to classify the bill according to the list of categories below. 
                    Choose the closest relevant category and do not output categories outside of this list. 
                    Please be politically neutral, keeping in mind that readers may have various ideological perspectives. 
                    Use the information from specified chapters and sections of the Mass General Laws to categorize the bill.

                    List of Categories:
                    {getattr(bill_details, 'categories', new_categories_for_bill_list)}

                    Note that the bill refers to specific existing chapters and sections of the Mass General Laws. Use the corresponding names of Mass General Law chapter and sections for constructing your summary.\n

                    The bill title is: {getattr(bill_details, 'bill_title')} \n

                    The bill text is: \n {getattr(bill_details, 'bill_text')}
                    
                    The relevant committee information: {getattr(bill_details, 'committee_info')}

                    The relevant section names are: {getattr(bill_details, "mgl_names")}


                    INSTRUCTIONS: 
                    1. Choose just 2 categories from the list above.
                    2. Do not provide explanations for the category choices.
                    3. Do not output categories not listed above.
                    4. Do not modify or paraphrase the category names, choose directly from the list provided.
                    5. Respond with # separated categories

                    Categories: """
    else: 
        query = """ 
                Your job is to classify the bill according to the list of categories below. 
                Choose the closest relevant category and do not output categories outside of this list. 
                Please be politically neutral, keeping in mind that readers may have various ideological perspectives. 
                Use the information from specified chapters and sections of the Mass General Laws to categorize the bill.

                List of Categories:
                {categories}

                Note that the bill refers to specific existing chapters and sections of the Mass General Laws. Use the corresponding names of Mass General Law chapter and sections for constructing your summary.\n

                The bill title is: \"{title}"\

                The bill text is: \"{context}"\
                
                The relevant committee information: \"{committee_info}"\

                The relevant section names are: \"{names}"\

                The relevant section text is: \"{mgl_sections}"

                INSTRUCTIONS: 
                1. Choose just 2 categories from the list above.
                2. Do not provide explanations for the category choices.
                3. Do not output categories not listed above.
                4. Do not modify or paraphrase the category names, choose directly from the list provided.
                5. Respond with # separated categories

                Categories: """
        bill_details.invoke_dict = prepare_invoke_dict(bill_details)
        bill_details.invoke_dict['categories'] = getattr(bill_details, 'categories', new_categories_for_bill_list)

    return query

def get_query_for_tagging(bill_details: BillDetails, category_tags: list, llm_call_type: str) -> str: 

    """
        
    This functions prepares a prompt based on the call type (small: No use of RAG, large: Use RAG) for bill tagging

    Args: 
        bill_details (BillDetails): Object containing information about the bill - bill_text, bill_title, mgl_ref, commottee_info, mgl_names
        category_tags (List): List of tags that the model has to filter from. 
        llm_call_type (str): This argument can take 2 values ("small": No use of RAG, "large": Use RAG) 

    Returns: 
        str: Query that includes task, context and instructions on bill tagging. 

    """

    if llm_call_type == 'large':
        query = f""" 
                Your Job here is to identify the tags that can be associated to the following MA Legislative bill. 
                Choose the closest relevant tags and do not output tags outside of the provided tags. 
                Please be politically neutral, keeping in mind that readers may have various ideological perspectives. 
                Note that the bill refers to specific existing chapters and sections of the Mass General Laws. Use the information from those chapters and sections in your context for tagging.

                
                List of tags: 
                - {category_tags}

                The bill title is: {getattr(bill_details, 'bill_title')}

                The bill text is: \n{getattr(bill_details, 'bill_text')}\n
                
                The relevant committee information: {getattr(bill_details, 'committee_info')}

                The relevant section names are: {getattr(bill_details, "mgl_names")}

                INSTRUCTIONS: 
                1. Choose a maximum of up to 5 tags.
                2. Do not provide explanations for the tag choices.
                3. Do not output tags not listed above.
                4. Do not modify or paraphrase the tag names, choose directly from the list provided.
                5. Do not assign tags only for the sake of tagging; tag them only if they are relevant.
                6. Respond with # separated tags.

                Tags: """

    else: 

        query = """ 
                Your Job here is to identify the tags that can be associated to the following MA Legislative bill. 
                Choose the closest relevant tags and do not output tags outside of the provided tags. 
                Please be politically neutral, keeping in mind that readers may have various ideological perspectives. 
                Note that the bill refers to specific existing chapters and sections of the Mass General Laws. Use the information from those chapters and sections in your context for tagging.

                
                List of tags: 
                - {category_tags}

                The bill title is: \"{title}"\

                The bill text is: \"{context}"\
                
                The relevant committee information: \"{committee_info}"\

                The relevant section names are: \"{names}"\

                The relevant section text is: \"{mgl_sections}"

                INSTRUCTIONS: 
                1. Choose minimum of 3 tags and no more than 5.
                2. Do not provide explanations for the tag choices.
                3. Do not output tags not listed above.
                4. Do not modify or paraphrase the tag names, choose directly from the list provided.
                5. Do not assign tags only for the sake of tagging; tag them only if they are relevant.
                6. Respond with # separated tags.
                
                Tags: """

        bill_details.invoke_dict = prepare_invoke_dict(bill_details)
        bill_details.invoke_dict['category_tags'] = category_tags

    return query

def call_llm(bill_details: BillDetails, query: str, llm_call_type: str = 'small') -> LLMResults: 

    """
        
    This is a generic function that calls the LLM with given query

    Args: 
        bill_details (BillDetails): Object containing information about the bill - bill_text, bill_title, mgl_ref, commottee_info, mgl_names
        query (str): Query string containing details on what model has to do. 
        llm_call_type (str): This argument can take 2 values ("small": No use of RAG, "large": Use RAG) 

    Returns: 
        LLMResults: Object containing query, response (Raw unformatted response from model) and metrics (If requested)

    """
    set_openai_api_key()

    llm = ChatOpenAI(temperature = 0, model = GPT_MDOEL_VERSION, model_kwargs = {'seed': 42})

    if llm_call_type == 'small': 
        response = small_docs(bill_details, query, llm)
    else: 
        response = large_docs(bill_details, query, llm)

    return_obj = LLMResults(query = query, response = response)

    return return_obj

def small_docs(bill_details: BillDetails, query: str, llm: ChatOpenAI) -> str:

    """
        
    This function calls the LLM without using RAG - Generally used if token count is less than 128k

    Args: 
        bill_details (BillDetails): Object containing information about the bill - bill_text, bill_title, mgl_ref, commottee_info, mgl_names
        query (str): Query string containing details on what model has to do. 
        llm (ChatOpenAI): LLM call object

    Returns: 
        (str): Raw response of the LLM. 

    """

    prompt = PromptTemplate.from_template(query)
    chain = create_stuff_documents_chain(llm, prompt)

    with get_openai_callback() as cb:
        response = chain.invoke(bill_details.invoke_dict)

    return response

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_or_create_embeddings(bill_details: BillDetails, emb_api: OpenAIEmbeddings): 

    bill_id = bill_details.bill_id
    client = chromadb.PersistentClient(CHROMA_DB_PATH)
    collection = client.get_or_create_collection(name = "bills_collection")
    
    existing_docs = collection.get(where={"bill_id": bill_id})

    if not existing_docs["ids"]:
        # print(f"Creating new embeddings for bill {bill_id}")

        text_splitter = TokenTextSplitter.from_tiktoken_encoder(
            chunk_size = 2000,
            chunk_overlap = 200
        )
        documents = text_splitter.split_text(bill_details.mgl_ref)

        embeddings = emb_api.embed_documents(documents)

        collection.add(
            documents = documents,
            embeddings = embeddings,
            metadatas = [{"bill_id": bill_id} for _ in documents],
            ids = [f"{bill_id}_{i}" for i in range(len(documents))]
        )
    return client

def large_docs(bill_details: BillDetails, query: str, llm: ChatOpenAI) -> str:

    """
        
    This function calls the LLM using RAG - Generally used if token count is greater than 128k

    Args: 
        bill_details (BillDetails): Object containing information about the bill - bill_text, bill_title, mgl_ref, commottee_info, mgl_names
        query (str): Query string containing details on what model has to do. 
        llm (ChatOpenAI): LLM call object

    Returns: 
        (str): Raw response of the LLM. 

    """

    emb_api = OpenAIEmbeddings()
    chroma_client = get_or_create_embeddings(bill_details, emb_api)
    vectorstore = Chroma(
        client=chroma_client,
        collection_name="bills_collection",
        embedding_function=emb_api
    )
    retrieval_doc_count = min((MAX_TOKEN_LIMIT - count_tokens('', bill_details.bill_text, '', '', ''))//2000 - 2, 7)
    
    retriever = vectorstore.as_retriever(
        search_type="similarity", 
        search_kwargs={
            "k": retrieval_doc_count,
            "filter": {"bill_id": bill_details.bill_id}
        }
    )
    

    # retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 12})

    template = """You are a trustworthy assistant for this task. Use the following pieces of retrieved context to accomplish the job.
        
        Relevant Massachussets General Law section text for context: {context}

        Bill Details: {question}

        """

    prompt = PromptTemplate.from_template(template)
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    try: 
        with get_openai_callback() as cb:
            response = rag_chain.invoke(query)
    except Exception as e: 
        print(e)

    return response