import pandas as pd
import re
import numpy as np
import requests
import urllib3
import requests
from requests.exceptions import RequestException
from requests.packages.urllib3.exceptions import InsecureRequestWarning
import logging

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

def csv_to_df_cleaned(bills_csv_file_path: str, bill_title_col_name:str, bill_num_col_name:str, docket_num_col_name:str, bill_text_col_name:str) -> pd.DataFrame: 
    """
      Parameters:
    - bills_csv_file_path (str): Path to the CSV file containing columns "Title", "BillNumber", "DocketNumber", "DocumentText".
    - bill_title_col_name (str): Name of the column containing bill titles.
    - bill_num_col_name (str): Name of the column containing bill numbers.
    - docket_num_col_name (str): Name of the column containing docket numbers.
    - bill_text_col_name (str): Name of the column containing bill text.

    Returns:
    - cleaned_df (pd.DataFrame): Modified Pandas DataFrame containing selected columns and an additional "Bill Info" column
      with the combined Bill Number and Title for demo_app.py.
    """
    try: 
        df = pd.read_csv(bills_csv_file_path)
        # logging.info('CSV file read successfully')
        
        cleaned_df = df[[bill_title_col_name, bill_num_col_name, docket_num_col_name, bill_text_col_name]]
        
        #Adding a column with the Bill Number and Title combined to be used for the demo_app.py
        bill_info = df[bill_num_col_name] + ": " + df[bill_title_col_name]
        cleaned_df.insert(0, 'Bill Info', bill_info)
        
        return cleaned_df
    
    except FileNotFoundError:
        logging.error('Error: The file %s was not found', bills_csv_file_path)
        return None 

def extract_sections(bill_text: str) -> list: 
    """
    Extracts chapters and sections from a bill using regular expressions.

    Parameters:
    - bill_text (str): The text of the bill containing chapters and sections.

    Returns:
    - lists_with_both: A list of lists containing pairs of chapters and sections extracted from the bill.
    """
    regex = ""
    chapter = ""
    section = ""
    
    
    check_list = ["section", "chapter"]
    #Regex to extract strings containing "section# of chapter#, 'section# of said chapter#' , 'section#' or 'chapter#' in a list
    regex = re.findall(r'(section)(\s+\d+[a-zA-Z]+|\s+\d+)\s+(of|of\ssaid)\s(chapter)(\s+\d+[a-zA-Z]+|\s\d+)|(section|chapter)(\s+\d+[a-zA-Z]+|\s\d+)',str(bill_text), re.IGNORECASE)
            
    lists_with_both =[]
    current_chapter = ""

    #iterate over regex list that contains both chapters and sections
    for x in regex:  
        items = []
        #iterate over extracted lists
        for item in x:
            item = item.casefold()
            items.append(item) 
        if all(name in items for name in check_list):
            for i, j in enumerate(x):
        
                if j.casefold() == "section":
                    section = x[i+1].strip()
        
                if j.casefold() == "chapter":
                    chapter = x[i+1].strip()

                    #save current chapter in order to use it for pairing with sections mentioned later
                    current_chapter = chapter 
                    
                    #add chapter and section to the list
                    list_with_both = [chapter, section]
                    
                    #only keep new/unique chapter and section pairs
                    if list_with_both not in lists_with_both:
                        lists_with_both.append(list_with_both)

        else:
            #iterate over list that contains only sections or chapters
            for i, j in enumerate(x):
                #ignore SECTION with caps as it indicates sections from the bills and not the MGL
                if j == "SECTION": 
                    continue
                if j == "":
                    continue
                
                else:
                    if j.casefold() == "chapter":
                        current_chapter = x[i+1].strip() #keep track of current chapter
                    
                    if j.casefold() == "section":
                        if x[i+1] == "":
                            continue
                        else:
                            section = x[i+1].strip()
                        
                        list_with_both = [current_chapter, section]
                        if list_with_both not in lists_with_both:
                            lists_with_both.append(list_with_both)
                            
    return lists_with_both

def apply_extract_sections(bills_df: pd.DataFrame, bill_text_col_name: str, chap_sec_lists_col_name:str) -> pd.DataFrame:
    
    """
    Applies the 'extract_sections' function to a DataFrame containing the bills text

    This function adds a new column 'Reference List' to the input DataFrame 'bills_df'. The 'DocumentText'
    column of 'bills_df' is processed using the 'extract_sections' function, and the results are stored in
    the newly created 'Reference List' column.

    Parameters:
    - bills_df (pd.DataFrame): The DataFrame containing bill information with a 'DocumentText' column.

    Returns:
    - bills_df (pd.DataFrame): The updated DataFrame with a new 'Reference List' column.
    """

    
    bills_df [chap_sec_lists_col_name] = bills_df[bill_text_col_name].apply(extract_sections)
    
    return bills_df

def query_section_text(chapter_section_list: tuple[str, str]) -> str | float:
    """
    Makes an API call to retrieve text data based on the provided chapter and section.

    Parameters:
    - chapter_section_list (list): A list containing two elements - chapter and section, e.g., ['2', '15D'].

    Returns:
    - result (str): The text data retrieved from the API.

    Note:
    - This function uses the malegislature.gov API to fetch text data for a specific chapter and section.
    
    """
    
    result = """"""
  
    try:
        # unpack section and chapter for example: ['2', '15D']
        chapter, section = chapter_section_list
        link = f'https://malegislature.gov/api/Chapters/{chapter}/Sections/{section}'
        r = requests.get(link, verify=False)
        r = r.json()

        # fields to extract
        result =  r.get("Text", np.nan)
        # logging.info('API call successful')
        return result
    except RequestException as e:
        # logging.error('API Request failed:section not found %s',chapter_section_list)
        pass
    

def query_section_text_all_bills(chapter_section_lists: list[tuple[str, str]]) -> tuple[list[str], list[tuple[str, str]]]:
    """
    Retrieves text data for each chapter-section pair in the given sample; prints chapter-section numbers to keep track of the progress.

    Parameters:
    - sample (list): A list of chapter-section pairs.

    Returns:
    - formatted_data(list): A list containing formatted text data for each non-empty chapter-section pair in the sample.
    - empty_responses(list): A list containing chapter section pairs where API doesn't return anything 

    Note:
    - This function prints the provided chapter-section pairs and retrieves text data for each pair using the `make_api_call` function.
    - The function skips empty or None pairs and ignores pairs with empty or NaN text data.
    - The formatted text data for each non-empty pair is stored in a list, which is then returned.
    """
    # print(df.index)
    # print("Chapter-Sections", chapter_section_lists)
    
# Storing and printing each pair
    requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
    formatted_data = []
    empty_responses = []
    result = """"""
    if len(str(chapter_section_lists)) == 0:
        return

    if str(chapter_section_lists) == "nan":
        return
       
    # Iterate through each pair in the chapter_section_lists
    for pair in chapter_section_lists:
        
        if len(pair) == 0:
            continue
        else:
            string = query_section_text(pair)
            if string in {None, np.nan, "", "nan"}:
                empty_responses.append(pair) #get a list of chapter-section pair where the API call returns an empty list
                continue
            else:
                result += string
        formatted_data.append(result)
        
    return formatted_data, empty_responses
    
def combine_mgl_sections(df: pd.DataFrame, chap_sec_lists_col_name: str) -> pd.DataFrame:
    """
    Modifies the provided DataFrame by applying the get_mgl function to a specified column, creating a new column 'MGL Ref'.
    Also prints the progress of getting MGL from the API
    
    Parameters:
    - df (pd.DataFrame): The Pandas DataFrame containing the data.
    - chap_sec_lists_col_name (str): The name of the column in the DataFrame that contains lists of chapter-section pairs.
    - mgl_col_name (str): The name of the column in the DataFrame that has the text from combined MGL sections

    Returns:
    - pd.DataFrame: The modified DataFrame with an additional column 'MGL Ref' containing formatted text data
      for each chapter-section pair in the specified column.
    """
    # Initialize the new columns with default values
    df['MGL Sections List'] = None
    df["Combined_MGL"] = "None"
    df["No MGL Found"] = None
    total_rows = len(df)
    
    # Iterate over the DataFrame rows using iterrows()
    for index, row in df.iterrows():
        # Calculate and print the percentage progress
        progress = (index + 1) / total_rows * 100
        print(f"\rProgress: {progress:.2f}%", end='')

        # Get MGL sections from the API as well as chapter-section pairs with empty API call
        mgl_list, empty_responses = query_section_text_all_bills(row[chap_sec_lists_col_name])
        df.at[index, 'MGL Sections List'] = mgl_list
        df.at[index, 'No MGL Found'] = empty_responses
        
        # combine the MGL sections for one bill into one string
        df.at[index, "Combined_MGL"] = ' '.join(mgl_list) if len(mgl_list) != 0 else "None"

    return df
    

def word_count(text: str) -> int:
    """
    Calculates the number of words in the provided text.

    Parameters:
    - text (str): The input text for which the word count is to be calculated.

    Returns:
    - int: The number of words in the text. Returns 0 if the input is NaN.
    """
    
    if pd.notna(text):
        words = text.split()  # split by spaces
        return len(words)
    else:
        return 0


def get_mgl_word_count(df: pd.DataFrame, word_count_col_name: str) -> pd.DataFrame:
    """
    Applies the word_count function to a specified column in a DataFrame, creating a new column 'NumWords'.

    Parameters:
    - df (pd.DataFrame): The Pandas DataFrame containing the data.
    - combined_mgl_col_name (str): The name of the column in the DataFrame that contains combined MGL references.

    Returns:
    - df: pd.DataFrame: The modified DataFrame with an additional column 'NumWords' containing the number of words
      for each MGL reference in the specified column.

    Note:
    - This function utilizes the get_mgl_length function to calculate the number of words in each MGL reference
      in the specified column and creates a new column 'NumWords' to store the results.
    """
    df[word_count_col_name] = df["Combined_MGL"].apply(word_count) # Column name hardcoded for now as it will be used in the demo_app.py
    return df

def get_df_with_mgl(bills_csv_file_path: str, bill_title_col_name: str, 
                    bill_num_col_name: str, docket_num_col_name: str, 
                    bill_text_col_name: str, chap_sec_lists_col_name: str, 
                    word_count_col_name: str) -> pd.DataFrame:
    
    """
    Processes a CSV file containing bill data and returns a DataFrame with additional columns for Massachusetts General Laws (MGL) sections and word counts.

    This function performs several steps:
    1. Converts the CSV file to a cleaned DataFrame.
    2. Extracts chapter-section lists from the bill text.
    3. Combines MGL sections into a single column.
    4. Calculates the word count for the combined MGL text.

    Parameters:
    - bills_csv_file_path (str): Path to the CSV file containing the bills data.
    - bill_title_col_name (str): Column name for bill titles.
    - bill_num_col_name (str): Column name for bill numbers.
    - docket_num_col_name (str): Column name for docket numbers.
    - bill_text_col_name (str): Column name for the text of the bills.
    - chap_sec_lists_col_name (str): Column name for the lists of chapter-section pairs.
    - mgl_col_name (str): Column name for the combined text of MGL sections.

    Returns:
    - pd.DataFrame: A DataFrame containing the original bill data along with additional columns for MGL sections and their word counts.
    """
    
    df = csv_to_df_cleaned(bills_csv_file_path, bill_title_col_name, bill_num_col_name, docket_num_col_name, bill_text_col_name)
    df = apply_extract_sections(df, bill_text_col_name, chap_sec_lists_col_name)
    df = combine_mgl_sections(df, chap_sec_lists_col_name)
    df = get_mgl_word_count(df, word_count_col_name)
    
    return df
        