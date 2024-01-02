import streamlit as st
import pandas as pd
import os
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.callbacks import get_openai_callback
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer
from sentence_transformers import CrossEncoder
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.docstore.document import Document

from sidebar import *
from tagging import *


st.set_page_config(page_title="Summarize and Tagging MA Bills", layout='wide')
st.title('Summarize Bills')

sbar()

model = CrossEncoder('vectara/hallucination_evaluation_model')

df = pd.read_csv("demoapp/two-thirds_bills_mgl.csv")

def find_bills(bill_number, bill_title):
    """input:
    args: bill_number: (str), Use the number of the bill to find its title and content
    """
    bill = df[df['BillNumber'] == bill_number]['DocumentText']

    try:
         # Locate the index of the bill
        idx = bill.index.tolist()[0]
        # Locate the content and bill title of bill based on idx
        content = df['DocumentText'].iloc[idx]
        #bill_title = df['Title'].iloc[idx]
        bill_number = df['BillNumber'].iloc[idx]
        # laws
        # law = df['combined_MGL'].iloc[idx]

        return content, bill_title, bill_number
    
    except Exception as e:
        content = "blank"
        st.error("Cannot find such bill from the source")
        

# bills_to_select = {
#     '#H3121': 'An Act relative to the open meeting law',
#     '#S2064': 'An Act extending the public records law to the Governor and the Legislature',
#     '#H711': 'An Act providing a local option for ranked choice voting in municipal elections',
#     '#S1979': 'An Act establishing a jail and prison construction moratorium',
#     '#H489': 'An Act providing affordable and accessible high-quality early education and care to promote child development and well-being and support the economy in the Commonwealth',
#     '#S2014': 'An Act relative to collective bargaining rights for legislative employees',
#     '#S301': 'An Act providing affordable and accessible high quality early education and care to promote child development and well-being and support the economy in the Commonwealth',
#     '#H3069': 'An Act relative to collective bargaining rights for legislative employees',
#     '#S433': 'An Act providing a local option for ranked choice voting in municipal elections',
#     '#H400': 'An Act relative to vehicle recalls',
#     '#H538': 'An Act to Improve access, opportunity, and capacity in Massachusetts vocational-technical education',
#     '#S257': 'An Act to end discriminatory outcomes in vocational school admissions'
# }


bill_info = df["Bill Info"].values

bills_to_select = {s.split(": ")[0]: s.split(": ")[1] for s in bill_info}
print(bills_to_select)

selectbox_options = [f"{number}: {title}" for number, title in bills_to_select.items()]
option = st.selectbox(
    'Select a Bill',
    selectbox_options
)

# Extracting the bill number from the selected option
selected_num = option.split(":")[0]
print(selected_num)
selected_title = option.split(":")[1]
print(selected_title)

# bill_content, bill_title, bill_number, masslaw = find_bills(selected_num, selected_title)
bill_content, bill_title, bill_number = find_bills(selected_num, selected_title)


def generate_categories(text):
    """
    generate tags and categories
    parameters:
        text: (string)
    """
    try:
        API_KEY = st.session_state["OPENAI_API_KEY"]
    except Exception as e:
         return st.error("Invalid [OpenAI API key](https://beta.openai.com/account/api-keys) or not found")
    
    # LLM
    category_prompt = """According to this list of category {category}.

        classify this bill {context} into a closest relevant category.

        Do not output a category outside from the list
    """

    prompt = PromptTemplate(template=category_prompt, input_variables=["context", "category"])

    
    llm = LLMChain(
            llm = ChatOpenAI(openai_api_key=API_KEY, temperature=0, model='gpt-4'), prompt=prompt)
    
    response = llm.predict(context = text, category = category_for_bill) # grab from tagging.py
    return response


def generate_response(bill_number, text, category):
    """Function to generate response"""

    API_KEY = st.session_state["OPENAI_API_KEY"]
    os.environ['OPENAI_API_KEY'] = API_KEY
    # loader = TextLoader("demoapp/extracted_mgl.txt").load()
    # bills_12_with_mgl.loc[bills_12_with_mgl['BillNumber'] == "H489", 'combined_MGL_y']
    mgl_ref = df.loc[df['BillNumber']== bill_number, 'combined_MGL']
    
    mgl_ref = mgl_ref.values[0]
    
    
    text_splitter = CharacterTextSplitter(chunk_size=4000, chunk_overlap=0)
    documents = [Document(page_content=x) for x in text_splitter.split_text(mgl_ref)]
    print(len(documents))
    # text_splitter = CharacterTextSplitter(chunk_size=4000, chunk_overlap=0)
    # documents = text_splitter.split_documents(loader)
    
    vectorstore = Chroma.from_documents(documents, OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()

        
    template = """You are a trustworthy assistant for question-answering tasks.
        Use the following pieces of retrieved context to answer the question.
        Question: {question}
        Context: {context}
        Answer:
        """

    prompt = PromptTemplate.from_template(template)
    llm = ChatOpenAI(openai_api_key=API_KEY, temperature=0, model='gpt-4-1106-preview', model_kwargs={'seed': 42})  
        
    rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
    query = f""" Can you please explain what the following MA bill means to a regular resident without specialized knowledge? 
            Please provide a one paragraph summary in 4 sentences. Please be simple, direct and concise for the busy reader. Make bullet points if possible.
            Note that the bill refers to specific existing chapters and sections of the Mass General Laws. Use the information from those chapters and sections in your context to construct your summary
            Summarize the bill that reads as follows:\n{text}\n\n
            
            After generating summary, output Category: {category}.
            Then, output top 2 tags in this specific category from the list of tags {tags_for_bill} that are relevant to this bill. \n"""
            # Do not output the tags outside from the list. \n
            # """
    with get_openai_callback() as cb:
        response = rag_chain.invoke(query)
        st.write(cb.total_tokens, cb.prompt_tokens, cb.completion_tokens, cb.total_cost)
        print("Total Tokens: ", cb.total_tokens, " Prompt Tokens: ", cb.prompt_tokens, "Completion Tokens" , cb.completion_tokens, "Total Cost", cb.total_cost)
    return response, mgl_ref


#Function to update or append to CSV
def update_csv(bill_num, title, summarized_bill, category, tag, csv_file_path, rouge_scorer, cosine_sim_score, fact_const_score ):
    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        # If the file does not exist, create a new DataFrame
        df = pd.DataFrame(columns=["Bill Number", "Bill Title", "Summarized Bill", "Category", "Rouge-1", "Rouge-2", "Rouge-L, Cosine Similarity, Fact Consistency"])
    mask = df["Bill Number"] == bill_num
    if mask.any():
        df.loc[mask, "Bill Title"] = title
        df.loc[mask, "Summarized Bill"] = summarized_bill
        df.loc[mask, "Category"] = category
        # df.loc[mask, "Tags"] = tag
        df.loc[mask, "Rouge-1"] = f"{rouge_scorer['rouge1'].fmeasure:.2f}"
        df.loc[mask, "Rouge-2"] = f"{rouge_scorer['rouge2'].fmeasure:.2f}"
        df.loc[mask, "Rouge-L"] = f"{rouge_scorer['rougeL'].fmeasure:.2f}"
        df.loc[mask, "Cosine Similarity"] = cosine_sim_score
        df.loc[mask, "Fact Consistency"] = fact_const_score
        
    else:
        new_bill = pd.DataFrame(columns=["Bill Number", "Bill Title", "Summarized Bill", "Category", "Rouge-1", "Rouge-2", "Rouge-L, Cosine Similarity, Fact Consistency"])
        df = pd.concat([df, new_bill], ignore_index=True)
    
    df.to_csv(csv_file_path, index=False)
    return df

csv_file_path = "demoapp/new_bills1.csv"


answer_container = st.container()
with answer_container:
    submit_button = st.button(label='Summarize')
    col1, col2, col3 = st.columns([1.5, 1.5, 1])

    if submit_button:
        with st.spinner("Working hard..."):

            category_response = generate_categories(bill_content)
            response, mgl_ref = generate_response(bill_number,bill_content, category_response)
            #tag_response = generate_tags(category_response, bill_content)
    
            with col1:
                st.subheader(f"Original Bill: #{bill_number}")
                st.write(bill_title)
                st.write(bill_content)

            with col2:
                st.subheader("Generated Text")
                st.write(response)
                st.write("###")
                    
            with col3:
                st.subheader("Evaluation Metrics")
                # rouge score addition
                scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
                rouge_scores = scorer.score(bill_content, response)
                st.write(f"ROUGE-1 Score: {rouge_scores['rouge1'].fmeasure:.2f}")
                st.write(f"ROUGE-2 Score: {rouge_scores['rouge2'].fmeasure:.2f}")
                st.write(f"ROUGE-L Score: {rouge_scores['rougeL'].fmeasure:.2f}")
                    
                # calc cosine similarity
                vectorizer = TfidfVectorizer()
                tfidf_matrix = vectorizer.fit_transform([bill_content, response])
                cosine_sim = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])
                st.write(f"Cosine Similarity Score: {cosine_sim[0][0]:.2f}")

                # test hallucination
                scores = model.predict([
                        [bill_content, response]
                    ])
                score_result = float(scores[0])
                st.write(f"Factual Consistency Score: {round(score_result, 2)}")
                
                # update_csv(bill_number, bill_title, response, category_response,csv_file_path, scorer, cosine_sim, score_result)
                # st.download_button(
                #             label="Download Text",
                #             data=pd.read_csv("demoapp/generated_bills1.csv").to_csv(index=False).encode('utf-8'),
                #             file_name='Bills_Summarization.csv',
                #             mime='text/csv',)
                             
                    # st.write("###")
                    # st.subheader("Token Usage")
                    # st.write(f"Response Tokens: {response_tokens + tag_tokens + cate_tokens}")
                    # st.write(f"Prompt Response: {prompt_tokens + tag_tokens + cate_prompt}")
                    # st.write(f"Response Complete:{completion_tokens +  tag_completion + cate_completion}")
                    # st.write(f"Response Cost: $ {response_cost + tag_cost + cate_cost}")              
                    # st.write(f"Cost: response $ {response_cost + tag_cost + cate_cost}")  