"""
This file contains a function to get chapter and section names for Mass General Laws from the MAlegislature APIs, 
'https://malegislature.gov/api/Parts/part#/Chapters' for all chapters for each part, 
'https://malegislature.gov/api/Chapters/chapter#' for chapter details
'https://malegislature.gov/api/Chapters/chapter#/Sections/Section# for section details
"""
import pandas as pd
from requests.sessions import Session

def get_chapter_section_names() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetches chapter and section names from the  specified URLs from MAlegislature API, constructs a DataFrame with chapter and section names,
    and returns the names DataFrame along with a DataFrame containing error URLs.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the data DataFrame and error DataFrame.
    """
    session = Session()
    names = []
    error_section_urls =[]
    
    parts_dict = {"I": "ADMINISTRATION OF THE GOVERNMENT", 
                 "II": "REAL AND PERSONAL PROPERTY AND DOMESTIC RELATIONS", 
                 "III": "COURTS, JUDICIAL OFFICERS AND PROCEEDINGS IN CIVIL CASES",
                 "IV": "CRIMES, PUNISHMENTS AND PROCEEDINGS IN CRIMINAL CASES", 
                 "V": "THE GENERAL LAWS, AND EXPRESS REPEAL OF CERTAIN ACTS AND RESOLVES"}

    for part_numb, part_name in parts_dict.items():
        chapters_response = session.get(f'https://malegislature.gov/api/Parts/{part_numb}/Chapters', verify=False)
        chapters_response.raise_for_status()
        chapters = chapters_response.json()
      
        part_number = f"Part {part_numb}"
         
       #len of chapter and index to define progress
        num_chapters = len(chapters)
        index = 0
        for chapter in chapters:
            
            progress = (index + 1) / num_chapters * 100
            index += 1
            print(f"\rPercent of Chapters Completed for {part_number} of V: {progress:.2f}%", end='')
            chapter_number = chapter['Code'] #chapter number
            
            chapter_details = chapter['Details'] #Chapter URLs
            chapter_name = session.get(chapter_details, verify=False).json()['Name']

            sections_response = session.get(chapter_details, verify=False)
            sections_response.raise_for_status()
            sections = sections_response.json()['Sections']
    
            for section in sections:
                section_number = section['Code'] #section numbers
                section_details = section['Details'] #section URLs
                try:
                    section_name = session.get(section_details, verify=False).json()['Name']
                except Exception as e:
                    error_section_urls.append(f"Chapter {chapter_number}: Section {section_number}: {section_details}") #logging errorred URLs

                names.append({'Part Number': part_number, 'Part Name': part_name, 'Chapter_Number': chapter_number, 'Chapter': chapter_name, 
                            'Section_Number': section_number,  'Section Name': section_name})

    names_df = pd.DataFrame(names)
    error_df = pd.DataFrame(error_section_urls)
    return names_df, error_df

names_df, error_df = get_chapter_section_names()

names_df.to_parquet("chapter_section_names.pq")
error_df.to_csv("Error_URLs_Chapter_Section_Names.csv")