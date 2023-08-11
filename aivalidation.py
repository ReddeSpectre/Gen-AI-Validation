#Imports and Openai Key
import streamlit as st
import pandas as pd
import numpy as np
import openai


st.title("Data Validation with Generative AI Assistant")

#create a dataframe for the raw data
df = pd.read_csv(r'googleplaystore.csv')

df_sample_art = df[df['Category'] == 'ART_AND_DESIGN'][:25]
df_sample_games = df[df['Category'] == 'GAME'][:25]
df_sample_shopping = df[df['Category'] == 'SHOPPING'][:25]
df_sample_business = df[df['Category'] == 'BUSINESS'][:25]

df_bal_sample = pd.concat([df_sample_art,df_sample_games,df_sample_shopping,df_sample_business])

trimmed_df = df_bal_sample.iloc[:, [0, 5]]

#create a smaller sample for analysis
trimmed_df_sample = trimmed_df.sample(10)

#pass the data to streamlit
st.write("Our initial trimmed dataset for use in the test:")
st.write(trimmed_df_sample)


#function for checking data for formatting errors
def check_data_validity(df):
    incorrect_format_data = []

    for index, row in df.iterrows():
        app = row['App']
        installs = row['Installs']

        if pd.isnull(app) or pd.isnull(installs):
            missing_data.append(index)
        else:
            # Check "App" column for any missing text or NaN/NA
            if not isinstance(app, str) or len(app.strip()) == 0 or app.lower() == 'nan' or app.lower() == 'na' or app.lower() == 'invalid':
                incorrect_format_data.append((index, 'App'))

            # Check "Installs" column for incorrect format or NaN/NA
            if not isinstance(installs, str) or not installs.endswith('+') or not installs[:-1].replace(',', '').isdigit() or installs.lower() == 'nan' or installs.lower() == 'na' or installs.lower() == 'invalid':
                incorrect_format_data.append((index, 'Installs'))

    # Return "none" if no missing or incorrect format data is found
    if not incorrect_format_data:
        incorrect_format_data = "none"

    return incorrect_format_data

# Assuming df is your DataFrame with the "App" and "Installs" columns
incorrect_format_data = check_data_validity(trimmed_df_sample)

#Pass the data to streamlit
st.write("When Checking this dataset for errors, the program returns the following:")
st.caption(incorrect_format_data)

keyinput = 0
while (keyinput == 0):
    #Get API Key from user
    st.write("In order to begin the demonstration, please input a valid OpenAI API Key:")
    key = st.text_input("API Key", "[Insert API Key Here]")
    keyinput = 1
    
openai.api_key = key

#Function for generating responses with GPT-3
def generate_gpt3_response(incorrect_format_data):
    # Generate a concise prompt based on the results
    prompt = f"""The data validity check has been performed. Here are the results and analysis of the results:
Incorrect Format Data: {incorrect_format_data}"""
    
    # GPT-3 API call to generate a response
    response = openai.Completion.create(
        engine="davinci",
        prompt=prompt,
        temperature=0.4,
        max_tokens=100
    )

    # Extract the response from GPT-3
    gpt3_response = response['choices'][0]['text'].strip()

    return gpt3_response

gpt3_response = generate_gpt3_response(incorrect_format_data)

#Pass GPT3 Response to streamlit
st.write("When passed to GPT for evaluation, GPT returns the following:")
st.caption(gpt3_response)

# Create 5 invalid entries
invalid_entries = [
    {'App': '', 'Installs': '10,000+'},
    {'App': 'WhatsApp', 'Installs': 'Invalid'},
    {'App': 'Facebook', 'Installs': '100,000'},
    {'App': 'Twitter', 'Installs': '500,000'},
    {'App': 'Invalid', 'Installs': '200,000+'},
    {'App': 'N/A', 'Installs': 'NaN'},
    {'App': 'NaN', 'Installs': 'xaew'},
    {'App': 'Jims Crazy World Business', 'Installs': 'NaN+'}
]

# Create a new dataframe with the original data and invalid entries
test_df = pd.concat([trimmed_df_sample, pd.DataFrame(invalid_entries)])

st.write("Now we'll add some invalid data to the dataset and evaluate it:")
st.write(test_df)

#check our new dataframe with invalid entries with the format checking method
incorrect_format_data = check_data_validity(test_df)

#send the results to GPT3 for analysis
gpt3_response = generate_gpt3_response(incorrect_format_data)

#pass GPT3 Response to Streamlit
st.write("When passed to GPT for evaluation GPT returns the following for our new dataset:")
st.caption(gpt3_response)

#function to remove invalid entries from the dataset
def remove_invalid_entries(df, incorrect_format_data):
    # Extract the index labels from the list of (index, column) pairs
    indices_to_drop = [index for index, _ in incorrect_format_data]

    # Create a new dataframe by excluding rows with indices present in indices_to_drop
    new_df = df.drop(indices_to_drop)

    return new_df

#Remove invalid data from the dataframe
test_df = remove_invalid_entries(test_df, incorrect_format_data)

#check dataframe
test_incorrect_format_data = check_data_validity(test_df)

st.write("After running a removal function, what we have left Matches our original valid set of data:")
st.write(test_df)

st.write("""Observations: 

1: GPT Responses are fairly erratic. Additionaly, every time the program is run, the results returned by GPT are different. This can result in errors. This can be solved by training a GPT model on what kind of responses are wanted and then deploying it.""")

st.write("2: Due to its nature as a Language Model, GPT was not able to handle validation tasks by itself. The Validation needed to be split away from the GPT Call.")
