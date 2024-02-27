""" 
This code only implements text summarization, category selection and tagging for 12 bills
Uses OpenAIEmbeddings and Vectorstore to split the MGL text into chunks, storing it and performing vector search
Changed the code to use selected bill number from the app to access the MGL text for that bill (original code used combined MGL text from all the extracted sections for all 12 bills)
For len(bills + MGL) > 100,000, the code impelemtns OpenAIEmbeddings and Vectorstore to split the MGL text into chunks of len = 100,000 for vector embeddings
For len(bills + MGL) <= 100,000, all the documents including, bill text, MGL, chapter and section names and bill title are passed into a single prompt. 
"""

import streamlit as st
import pandas as pd
import os
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_community.callbacks import get_openai_callback
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer
from sentence_transformers import CrossEncoder
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.docstore.document import Document
from typing import Tuple, List, Dict
from sidebar import *
from tagging import *




def set_page_config()-> None:
    """
    Configures the Streamlit page with a custom title and layout, and displays a custom title for the app.
    """
    st.set_page_config(page_title="Summarize and Tagging MA Bills", layout='wide')
    st.title('Summarize Bills')
    sbar()

# Displaying the selectbox
def get_selected_bill_info(df: pd.DataFrame) -> Tuple[str, str, str]:
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

def get_chapter_section_names(df: pd.DataFrame, bill_number: str) -> List[Dict[str, str]]:
    """
    Fetches chapter and section names for a given bill number from a DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame containing bill and chapter-section information.
        bill_number (str): The number of the bill for which chapter and section names are requested.
    
    Returns:
        List[Dict[str, str]]: A list of dictionaries with chapter and section names.
    
    The function assumes the DataFrame has the necessary columns.
    """
    names_df = pd.read_parquet("chapter_section_names.pq")
    chap_sec_lists = df[df['BillNumber'] == bill_number]['Chapter-Section List']
    names = []

    for lists in chap_sec_lists:
        for tup in lists:
            chap, sec = tup
            # chapter_name, section_name = names_df[(names_df["Chapter_Number"] == chap) & (names_df["Section_Number"] == sec)][['Chapter', 'Section Name']].values
            try:
                chapter_name = names_df[(names_df["Chapter_Number"] == chap) & (names_df["Section_Number"] == sec)]['Chapter'].values[0]
                section_name = names_df[(names_df["Chapter_Number"] == chap) & (names_df["Section_Number"] == sec)]['Section Name'].values[0]
                names.append({chap: chapter_name, sec: section_name})
            except Exception as e:
                print("Missing Chapter or Section Name {e}")
                continue

        return names

def get_MGL_for_bill(df:pd.DataFrame, bill_number) -> str:
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
    
def generate_response_large_documents(bill_title: str, bill_text: str, mgl_ref: str, mgl_names: List[Dict[str, str]], llm:object) -> str:
    """
    Generates a detailed response for large documents related to MA bills.
    
    Args:
        bill_title (str): Title of the bill.
        bill_text (str): Text content of the bill.
        mgl_ref (Union[str, None]): Reference Mass General Law text, if any.
        mgl_names (List[Dict[str, str]]): List of Mass General Law chapter and section names.
        llm: Language model for generating responses.
    
    Returns:
        str: Detailed response generated by the language model.
    """
    #Splitting the text and using vector embeddings when document size is > 90000 words
    
    #Context size for 'gpt-4-1106-preview' is 128K; Need to break up the documents and use vector embenddings when the word length of the documents is large
    #Chose 90K cut off for large docs because: 1 token = 0.75 words --> 90000 words = ~120K words + leaving  room for response

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
            Response Format:\nSummary:\n[SUMMARY]\nCategory:\n[CATEGORY]\nTags:\n[TAGS]")
            """         
    with get_openai_callback() as cb:
        response = rag_chain.invoke(query)
        st.write(cb.total_tokens, cb.prompt_tokens, cb.completion_tokens, cb.total_cost)
        return response
        
def generate_response_small_documents(bill_title: str, bill_text: str, mgl_ref: str, mgl_names: List[Dict[str, str]], llm: object) -> str:
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
    bill_text = [Document(page_content=x) for x in bill_text]
    
   
    template =  """Can you please explain what the following MA bill means to a regular resident without specialized knowledge? 
            Please provide a one paragraph summary in 4 sentences. \n Please be simple, direct and concise for the busy reader. Make bullet points if possible.\n
            Note that the bill refers to specific existing chapters and sections of the Mass General Laws. \n 
            Summarize the bill that reads as follows {context} and has a title {title}
            
            After generating summary, classify the bill according to this list of categories {categories} to the closest relevant category. Do not output categories outside of the list. \n
            Then, output top 3 tags in this specific category from the list of tags {tags} that are relevant to this bill. \n
            Do not output the tags outside from the list. \n

            Use the information from those chapters and sections in provided in {mgl_sections} as well as the corresponding {names} of those chapter and sections to construct your summary, generating categories and tags.\n 
            
            Response Format:\nSummary:\n[SUMMARY]\nCategory:\n[CATEGORY]\nTags:\n[TAGS]")
            """

    prompt = PromptTemplate.from_template(template)
    
    chain = create_stuff_documents_chain(llm, prompt)

    with get_openai_callback() as cb:
        response = chain.invoke({"context": bill_text, "mgl_sections": mgl_documents, "title": bill_title, "names": mgl_names, "categories": category_for_bill, "tags": tags_for_bill})
        st.write(cb.total_tokens, cb.prompt_tokens, cb.completion_tokens, cb.total_cost)
        
    return response

def generate_response(df: pd.DataFrame, bill_number: str, bill_title: str, bill_text: str, mgl_ref: str) -> str:
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

    mgl_names = get_chapter_section_names(df, bill_number)
    
    #context size for 'gpt-4-1106-preview' is 128K; Need to break up the documents and use vector embenddings when the word length of the documents is large
    #Chose 90K cut off for large docs because: 1 token = 0.75 words --> 90000 words = ~120K words + leaving  room for response
    if (len(str(mgl_ref).split()) + len (str(bill_text).split())) < 90000:
        response = generate_response_small_documents(bill_title, bill_text, mgl_ref, mgl_names, llm)
    else:
        response = generate_response_large_documents(bill_title, bill_text, mgl_ref, mgl_names, llm)
    return response

def combine_bill_text_mgl_and_names(bill_title:str, bill_text:str, mgl_ref:str):
    """ 
    Combines bill_title, bill_text and MGL_ref in one document for evaluation matrics
    
    Args:
    bill_title: str
    bill_text: str
    mgl_ref: str
    
    Returns:
    Combined text: str
    """
    combined_text = bill_title + bill_text + mgl_ref
    return combined_text

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

# Function to update or append to CSV
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


def render_response():
    """
    Renders the user interface for summarizing bill information, including the generation of
    a summary, evaluation metrics, and the ability to download the summarized text.

    This function performs the following steps:
    1. Sets the page configuration for the Streamlit UI.
    2. Initializes a CrossEncoder model for hallucination evaluation.
    3. Loads a dataset containing bills and their associated information from a parquet file.
    4. Retrieves selected bill information (number, title, and content) based on user input.
    5. Displays a button for the user to initiate the summarization process.
    6. Upon clicking the 'Summarize' button, generates a response including the summary, category,
       and tags for the selected bill.
    7. Updates a CSV file with the new summary information.
    8. Allows the user to download the updated CSV file.
    9. Calculates and displays evaluation metrics for the generated summary, including:
       - ROUGE-1, ROUGE-2, and ROUGE-L scores for assessing the overlap between the generated summary and the original bill content.
       - Cosine similarity score for measuring the similarity between the original bill content and the generated summary.
       - A factual consistency score using the initialized CrossEncoder model to evaluate the factual alignment of the generated summary with the original bill content.


    The function does not return any values but instead directly interacts with the Streamlit interface to render content.
    """
    set_page_config()
    model = CrossEncoder('vectara/hallucination_evaluation_model')

    # load the dataset containing bills and MGL information
    df = pd.read_parquet("demoapp/12_bills_with_more_sections.pq")

    csv_file_path = "demoapp/generated_bills.csv"
    bill_number, bill_title, bill_text = get_selected_bill_info(df)
    mgl_ref = get_MGL_for_bill(df, bill_number)
    combined_text = combine_bill_text_mgl_and_names(bill_title, bill_text, mgl_ref)
    answer_container = st.container()
    with answer_container:
        submit_button = st.button(label='Summarize')
        col1, col2, col3 = st.columns([1.5, 1.5, 1])

        if submit_button:
            with st.spinner("Working hard..."):
                response = generate_response(df, bill_number, bill_title, bill_text, mgl_ref)
        
                with col1:
                    st.subheader(f"Original Bill: #{bill_number}")
                    st.write(bill_title)
                    st.write(bill_text)

                with col2:
                    st.subheader("Generated Text")
                    st.write(response)
                    st.write("###")
                    summary, category, tags = parse_response(response)
                    update_csv(bill_number, bill_title, summary, category, tags, csv_file_path)
                    st.download_button(
                                label="Download Text",
                                data=pd.read_csv("demoapp/generated_bills.csv").to_csv(index=False).encode('utf-8'),
                                file_name='Bills_Summarization.csv',
                                mime='text/csv',)
                        
                with col3:
                    st.subheader("Evaluation Metrics")
                    # rouge score addition
                    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
                    rouge_scores = scorer.score(combined_text, response)
                    st.write(f"ROUGE-1 Score: {rouge_scores['rouge1'].fmeasure:.2f}")
                    st.write(f"ROUGE-2 Score: {rouge_scores['rouge2'].fmeasure:.2f}")
                    st.write(f"ROUGE-L Score: {rouge_scores['rougeL'].fmeasure:.2f}")
                        
                    # calc cosine similarity
                    vectorizer = TfidfVectorizer()
                    tfidf_matrix = vectorizer.fit_transform([combined_text, response])
                    cosine_sim = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])
                    st.write(f"Cosine Similarity Score: {cosine_sim[0][0]:.2f}")

                    # test hallucination
                    scores = model.predict([
                            [combined_text, response]
                        ])
                    score_result = float(scores[0])
                    st.write(f"Factual Consistency Score: {round(score_result, 2)}")
                    
                
                                
                        # st.write("###")
                        # st.subheader("Token Usage")
                        # st.write(f"Response Tokens: {response_tokens + tag_tokens + cate_tokens}")
                        # st.write(f"Prompt Response: {prompt_tokens + tag_tokens + cate_prompt}")
                        # st.write(f"Response Complete:{completion_tokens +  tag_completion + cate_completion}")
                        # st.write(f"Response Cost: $ {response_cost + tag_cost + cate_cost}")              
                        # st.write(f"Cost: response $ {response_cost + tag_cost + cate_cost}")  

render_response()


