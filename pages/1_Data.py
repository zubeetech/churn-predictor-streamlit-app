import streamlit as st
import os
import pyodbc
import pandas as pd

st.title('VODAFONE CUSTOMER CHURN PREDICTOR')

@st.cache_resource(show_spinner='connecting to database...')
def initialize_connection():
    server = "dap-projects-database.database.windows.net"
    database = "dapDB"
    uid = "LP2_project"
    pwd = "Stat$AndD@t@Rul3"

    connection_string = (
        "DRIVER={ODBC Driver 17 for SQL Server};"
        f"SERVER={server};"
        f"DATABASE={database};"
        f"UID={uid};"
        f"PWD={pwd};"
        "Encrypt=yes;"
        "TrustServerCertificate=no;"
    )

    try:
        connection = pyodbc.connect(connection_string)
        print("Connected successfully!")
        return connection
    except pyodbc.Error as e:
        print("Error connecting to SQL Server:", e)
        return None

conn = initialize_connection()

def query_database(query):
    with conn.cursor() as cur:
        cur.execute(query)
        rows = cur.fetchall()
        df = pd.DataFrame. from_records(data=rows, columns=[ column[0] for column in cur.description])
    return df

@st.cache_data()
def select_all_features():
    query = "SELECT * FROM LP2_Telco_churn_first_3000"
    df = query_database(query)
    return df

@st.cache_data()
def select_numeric_features():
    query = "SELECT * FROM LP2_Telco_churn_first_3000"
    df = query_database(query)
    numeric_df = df.select_dtypes(include=['number'])
    return numeric_df

if __name__ == "__main__":
    col1, col2 = st.columns(2)

    with col1:
        selected_option = st.selectbox("Select type of features", options=['All features', 'Numeric features'], key="selected_columns")

    with col2:
        pass

    if selected_option == "All features":
        data = select_all_features()
    elif selected_option == "Numeric features":
        data = select_numeric_features()

    st.dataframe(data)
