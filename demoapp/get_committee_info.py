"""
This file contains code to extract committee information from the MALegislature API.
CommitteeCode is extracted from the bill API: 'https://malegislature.gov/api/documents/bill_number#'
CommitteeName and Description are extracted using the CommitteeCode from the committee API: 'https://malegislature.gov/api/committees/committee_code'
"""

import requests
import pandas as pd

def get_committee_info(file_path, output_file_name_pq):
    """
    Extracts and compiles information about committees related to each bill listed in a CSV file.

    This function reads CSV file containing bill numbers, fetches CommitteeCode for each bill from an API. 
    It then uses the committee code to fetch information about the committee name and description from the API.
    If the process fails for any bill, it records None for that bill's committee information.
    It then saves the compiled information to a Parquet file named 'committee_info.pq'. 

    Args:
    - file_path (str): The path to the CSV file containing the list of bills. The CSV file is expected to have a column named 'BillNumber' which contains the bill numbers.

    Returns:
    - pandas.DataFrame: A DataFrame containing the bill number, committee code, committee name, and committee description for each bill. If information for a bill cannot be fetched, the respective fields are filled with None.

    Raises:
    - Exception: Propagates any exceptions raised during file reading, API requests, or file writing back to the caller.

    Note:
    - The function prints the progress of fetching committee information in terms of the percentage of bills processed.
    - API requests are made with SSL verification disabled (verify=False). This is not recommended for production code due to security concerns.
    """
    df = pd.read_csv(file_path)
    num_bills = len(df)
    committees_info = []
    
    for idx, bill_number in df["BillNumber"].items():
        progress = (idx + 1) / num_bills * 100
        idx += 1
        print(f"\rPercent of Bills Completed: {progress:.2f}%", end='')
        try:
            #Bill API
            response = requests.get(f"https://malegislature.gov/api/documents/{bill_number}", verify = False)
            response = response.json()
            #Grab CommitteeCode from the bill API
            committee_code = response['CommitteeRecommendations'][0]['Committee']['CommitteeCode']
            
            #Committee API
            committee = requests.get(f"https://malegislature.gov/api/committees/{committee_code}", verify=False)
            
            #Grab committee name and description from the committee API
            committee_name = committee.json()["FullName"]
            committee_description = committee.json()["Description"]
            committees_info.append({"BillNumber": bill_number, "CommitteeCode": committee_code, "CommitteeName": committee_name, "Description": committee_description})

        except Exception as e:
            committees_info.append({"BillNumber": bill_number, "CommitteeCode": None, "CommitteeName": None, "Description": None})
            pass

    committees_df = pd.DataFrame(committees_info)
    committees_df.to_parquet(output_file_name_pq)
    return committees_df


def main():
    all_bill_file_path = "demoapp/all_bills.csv"
    output_file_path = "demoapp/committee_info.pq"
    get_committee_info(all_bill_file_path, output_file_path)

if __name__ == '__main__':
    main()
