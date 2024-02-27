import pandas as pd
from requests.sessions import Session
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def fetch_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetches data from specified URLs, constructs a DataFrame with chapter and section details,
    and returns the data DataFrame along with a DataFrame containing error URLs.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the data DataFrame and error DataFrame.
    """
    
    session = Session()
    parts = ["I", "II", "III", "IV", "V"]
    names = []
    error_section_urls =[]

    for part in parts:
        chapters_response = session.get(f'https://malegislature.gov/api/Parts/{part}/Chapters', verify=False)
        chapters_response.raise_for_status()
        chapters = chapters_response.json()

        if part == "I":
            part_name = "ADMINISTRATION OF THE GOVERNMENT"
        elif part == "II":
            part_name = "REAL AND PERSONAL PROPERTY AND DOMESTIC RELATIONS"
        elif part == "III":
            part_name = "COURTS, JUDICIAL OFFICERS AND PROCEEDINGS IN CIVIL CASES"
        elif part == "IV":
            part_name = "CRIMES, PUNISHMENTS AND PROCEEDINGS IN CRIMINAL CASES"
        else:
            part_name = "THE GENERAL LAWS, AND EXPRESS REPEAL OF CERTAIN ACTS AND RESOLVES"
      
        part_num = f"Part {part}"
         
       #len of chapter and index to define progress
        num_chapters = len(chapters)
        index = 0
        for chapter in chapters:
            progress = (index + 1) / num_chapters * 100
            index += 1
            print(f"\rPercent of Chapters Completed for Part {part} of V: {progress:.2f}%", end='')

            chapter_number = chapter['Code'] #chapter number
        
            chapter_details = chapter['Details'] #Chapter URLs
            chapter_name = session.get(f'{chapter_details}', verify=False).json()['Name']

            sections_response = session.get(f'{chapter_details}', verify=False)
            sections_response.raise_for_status()
            sections = sections_response.json()['Sections']
    
            for section in sections:
                section_number = section['Code'] #section numbers
                section_details = section['Details'] #section URLs
                try:
                    section_name = session.get(f'{section_details}', verify=False).json()['Name']
                except Exception as e:
                    error_section_urls.append(f"Chapter {chapter_number}: Section {section_number}: {section_details}") #logging errorred URLs

                names.append({'Part Number': part_num, 'Part Name': part_name, 'Chapter_Number': chapter_number, 'Chapter': chapter_name, 
                            'Section_Number': section_number,  'Section Name': section_name})

    df = pd.DataFrame(names)
    error_df = pd.DataFrame(error_section_urls)
    return df, error_df

data_df, error_df = fetch_data()

data_df.to_parquet("chapter_section_names.pq")
error_df.to_csv("Error_URLs_Chapter_Section_Names.csv")