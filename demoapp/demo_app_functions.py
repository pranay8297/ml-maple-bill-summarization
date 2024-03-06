"""
This code implements text summarization, category selection and tagging bills using a single prompt
MGL sections in 'all_bills_with_mgl.pq' files were generated using extract_mgl_section.py 
For token size > 120K,  the code impelemtns OpenAIEmbeddings and Vectorstore to split the MGL text into chunks of len = 90K for vector embeddings
For token size < 120K, all the documents including, bill text, MGL, chapter and section names and bill title are passed into a single prompt. 
"""
import streamlit as st
import pandas as pd
import os
import numpy as np
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_community.callbacks import get_openai_callback
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
import urllib.request
from langchain.chains.combine_documents import create_stuff_documents_chain
from typing import Tuple, List, Dict
import tiktoken
from sidebar import *
from tagging import *

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
    
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

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
    #Splitting the text and using vector embeddings when document size is > 90000 words for large documents

    text_splitter = CharacterTextSplitter(chunk_size=90000, chunk_overlap=1000)
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
            Category:[CATEGORY]\n
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
    llm = ChatOpenAI(openai_api_key=API_KEY, temperature=0, model='gpt-4-1106-preview', model_kwargs={'seed': 42})  
    committee_info = get_committee_info(committee_file_name, bill_number)
    mgl_names = get_chap_sec_names(df, bill_number, mgl_names_file_name)
    num_tokens = count_tokens(bill_title, bill_text, mgl_ref, mgl_names, committee_info)
    #Context size for 'gpt-4-1106-preview' is 128K; Need to break up the documents and use vector embenddings when the word length of the documents is large
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
    