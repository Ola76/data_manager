import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import base64
import os
import matplotlib.pyplot as plt
import base64
import io
from datetime import datetime
from tabula import read_pdf
from scipy.stats import mstats
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer
from fpdf import FPDF
from io import BytesIO


# Function to check the provided credentials.
def check_credentials(username, password):
    if username == "admin" and password == "password123":
        return True
    return False

@st.cache(allow_output_mutation=True)
def load_data(file):
    file_extension = os.path.splitext(file.name)[1].lower()
    if file_extension == ".csv":
        return pd.read_csv(file)
    elif file_extension in [".xlsx", ".xls"]:
        return pd.read_excel(file)
    elif file_extension == ".pdf":
        # Using tabula to read tables from PDF
        dfs = read_pdf(file, pages='all', multiple_tables=True)
        if dfs:
            # If there's only one table, return it
            if len(dfs) == 1:
                return dfs[0]
            # If there are multiple tables, ask the user which one(s) they want
            else:
                options = [f"Table {i}" for i in range(1, len(dfs) + 1)] + ["All"]
                selected_option = st.selectbox(
                    "Multiple tables detected. Please select the table you want to load:",
                    options=options
                )                
                if selected_option == "All":
                    # Concatenate all tables vertically
                    return pd.concat(dfs, ignore_index=True)
                else:
                    table_index = options.index(selected_option)
                    return dfs[table_index]
        else:
            st.error("No tables found in PDF!")
            return None
    else:
        st.error("Unsupported file format!")
        return None

def clean_categorical_data(df):
    # Select only the categorical columns
    cat_columns = df.select_dtypes(['object']).columns

    # Clean the categorical columns
    for col in cat_columns:
        # Strip white spaces
        df[col] = df[col].str.strip()
        # Convert to lowercase
        df[col] = df[col].str.lower()
        # Remove special characters
        df[col] = df[col].str.replace('[^a-zA-Z0-9]', '', regex=True)
    return df

def handle_outliers(df, method="winsorize", limits=(0.01, 0.01)):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if method == "winsorize":
        for col in numeric_cols:
            df[col] = mstats.winsorize(df[col], limits=limits)
        
        # Provide the Winsorize explanation
        st.markdown('<span title="Winsorizing is a method to limit extreme values in the data. Values more extreme than a specified range will be replaced with the nearest boundary value.">Winsorize 🛈</span>', unsafe_allow_html=True)
    
    return df

def advanced_imputation(df, method="KNN"):
    numeric_df = df.select_dtypes(include=[np.number])  # Extract numeric columns
    st.markdown('<span title="KNN (K-Nearest Neighbors) imputer replaces missing values using the mean value of k nearest neighbors.">KNN 🛈</span>', unsafe_allow_html=True)
    st.markdown('<span title="Iterative imputation estimates missing values by modeling each feature as a function of other features in a round-robin fashion. Its a more complex and potentially more accurate method of imputation.">Iterative 🛈</span>', unsafe_allow_html=True)
    
    if method == "KNN":
        imputer = KNNImputer(n_neighbors=5)
        imputed_data = imputer.fit_transform(numeric_df)
        df[numeric_df.columns] = imputed_data  # Update only numeric columns with imputed data
    elif method == "Iterative":
        imputer = IterativeImputer(max_iter=10, random_state=0)
        imputed_data = imputer.fit_transform(numeric_df)
        df[numeric_df.columns] = imputed_data  # Update only numeric columns with imputed data
    return df

def clean_data(df, method):
    if method == "Median":
        return df.fillna(df.median(numeric_only=True))
    elif method == "Mean":
        return df.fillna(df.mean(numeric_only=True))
    elif method == "Most Frequent":
        return df.apply(lambda x: x.fillna(x.value_counts().index[0]) if x.dtype == "O" else x.fillna(x.mode()[0]), axis=0)
    return df  # if method is "None", return the original data

def dataframe_to_pdf(df):
    pdf = FPDF()
    pdf.add_page()
    
    # Set font for the PDF
    pdf.set_font("Arial", size=8)
    page_width = pdf.w - 2 * pdf.l_margin
    col_width = page_width/len(df.columns)
    row_height = 8 * 1.5

    for col in df.columns:
        pdf.cell(col_width, row_height, col, border=1)

    pdf.ln(row_height)

    for index, row in df.iterrows():
        for item in row:
            pdf.cell(col_width, row_height, str(item), border=1)
        pdf.ln(row_height)

    # Convert PDF into bytes string and return
    return pdf.output(dest='S').encode('latin1')

def create_download_link(df, download_format="csv"):
    if download_format == "csv":
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()  
        href = f'data:file/csv;base64,{b64}'
    elif download_format == "pdf":
        pdf_data = dataframe_to_pdf(df)
        b64 = base64.b64encode(pdf_data).decode()
        href = f'data:file/pdf;base64,{b64}'
    return href
    
def main():
    st.title("Advanced Data Cleaning & Visualization Application")

    st.write("""
        Welcome to our Advanced Data Cleaning & Visualization Hub! Here's a curated journey of the features we've built for you:

        1. **Dataset Uploader**: Begin by uploading your dataset. We accept CSV, Excel, and even PDF formats. Dive into the smooth integration of your data with our state-of-the-art uploader.

        2. **Tackle Outliers (For Numeric Data)**: Outliers can skew results. Especially built for numerical data, our tools, including the 'Winsorize' method, strategically adjust these data points to ensure they don't misrepresent statistical analyses.

        3. **Masterclass Imputation (Again, for Numeric Data)**: Absence makes the data grow fonder... of errors. Handle missing values with our cutting-edge techniques like KNN and Iterative imputation, exclusively for numeric columns. We're constantly innovating to bring you even more advanced methods.

        4. **Simple Yet Powerful Imputation**: Dive into foundational methods like Median, Mean, or Mode imputation. Depending on the nuances of your dataset and your analytical aspirations, sometimes simplicity is the key.

        Once you've set everything, sit back and let us refine your data. But that's not all – our integrated visualization suite provides instant insights into the heart of your cleaned dataset. Embark on a data cleaning journey like never before! 
             
        Application by Oburoh
        """)

    # Load data
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file:
        df = load_data(uploaded_file)

        # Call the cleaning function
        df = clean_categorical_data(df)
        if 'columns_to_drop' not in st.session_state:
            st.session_state.columns_to_drop = []

        # Display multiselect for dropping columns
        columns_to_drop = st.multiselect("Select columns to drop", df.columns, st.session_state.columns_to_drop)

        # Update the session state
        st.session_state.columns_to_drop = columns_to_drop

        # Drop the selected columns
        df = df.drop(columns=st.session_state.columns_to_drop)

        # Display the dataframe
        st.write(df)
        # Calculate the number of missing values for each feature
        missing_values = df.isnull().sum()

        # Filter out features without missing values
        missing_values = missing_values[missing_values > 0]

        # Display missing values
        if not missing_values.empty:
            st.write("Number of missing values for each feature:")
            st.write(missing_values)
        else:
            st.write("No missing values detected in the dataset!")

            # Outlier Handling
        outlier_method = st.selectbox("Choose a method to handle outliers", ["None", "Winsorize"])
        if outlier_method == "Winsorize":
            df = handle_outliers(df, method="winsorize")

        # Advanced Imputation
        advanced_impute_method = st.selectbox("Choose an advanced imputation method", ["None", "KNN", "Iterative"])
        if advanced_impute_method != "None":
            df = advanced_imputation(df, method=advanced_impute_method)
        else:
            # Your existing cleaning methods
            cleaning_method = st.selectbox("Choose a method to clean the data", ["None", "Median", "Mean", "Most Frequent"])
            df = clean_data(df, cleaning_method)
        # Displaying unique values for each column
        unique_counts = df.nunique()
        unique_df = pd.DataFrame({
            'Feature': unique_counts.index,
            'Unique Values': unique_counts.values
        })

        st.write("Number of Unique Values for Each Feature:")
        st.table(unique_df)

        # Create buttons using the correct links
        csv_href = create_download_link(df, "csv")
        csv_button = f'<a href="{csv_href}" download="cleaned_data.csv" class="btn btn-outline-primary btn-sm">Download Cleaned Data as CSV</a>'

        pdf_href = create_download_link(df, "pdf")
        pdf_button = f'<a href="{pdf_href}" download="cleaned_data.pdf" class="btn btn-outline-secondary btn-sm">Download Cleaned Data as PDF</a>'

        st.markdown(csv_button, unsafe_allow_html=True)
        st.markdown(pdf_button, unsafe_allow_html=True)

        st.write("The shape of the data: Column 1, Row 0")
        st.table(df.shape)

        # ... Categorical Data Visualization
        st.title("Categorical Data")
        st.markdown("""
        Categorical data refers to variables that can be divided into multiple categories but do not have any order or priority. They are often non-numeric and represent characteristics such as gender, ethnicity, or regions. Unlike numeric data, they usually take on limited, and typically fixed, numbers of possible values (referred to as 'categories' or 'levels'). Understanding these variables is crucial as they can provide qualitative insights and can be encoded to numeric form for various analyses and machine learning tasks.
        """)

        # Filter categorical columns
        categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
        if categorical_columns:
            selected_cat_column = st.selectbox("Select a categorical column", categorical_columns)
        # Plot bar graph for the selected column using Plotly
            fig = px.bar(df, x=selected_cat_column, title=f'Bar Chart of {selected_cat_column}')
            st.plotly_chart(fig)
        else:
            st.write("No categorical columns found in the dataset.")
        
        # ... Numeric Data Visualization
        st.title("Numeric Data")
        st.markdown("""
        Numeric data refers to variables that contain numerical values. These are typically represented by **integers** (`int`) and **floating-point numbers** (`float`). The distinction is important because numerical operations, analyses, or visualizations can be applied to these data types. Their numeric nature aids in obtaining meaningful insights and conducting quantitative analysis.
        """)

        # Filter numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_columns:
            selected_num_column = st.selectbox("Select a numeric column", numeric_columns)
            # Plot frequency distribution for the selected numeric column using Plotly
            fig = px.histogram(df, x=selected_num_column, marginal='box', title=f'Distribution of {selected_num_column}')
            st.plotly_chart(fig)
        else:
            st.write("No numeric columns found in the dataset.")

        st.title("FacetGrid Visualization")
        st.markdown("""
            FacetGrid visualization provides a way to visualize the distribution or relationship of data separately within subsets. This can be particularly useful when you want to explore the variation of data across different categories or levels of a certain feature.
        """)
        # Choose the variable for faceting
        facet_variable = st.selectbox("Select a variable for faceting", df.columns)
        numeric_columns_without_facet = [col for col in numeric_columns if col != facet_variable]

        # Ensure that the chosen facet variable has more than one unique value and less than a certain threshold (e.g., 20 for performance reasons)
        if 1 < df[facet_variable].nunique() <= 20:
            # Choose a variable for the x-axis
            x_variable = st.selectbox("Select a variable for the x-axis", numeric_columns_without_facet)
            # Generate the FacetGrid-like plot
            try:
                facet_fig = px.histogram(df, 
                                x=x_variable,
                                facet_col=facet_variable,
                                title=f"Distribution of {x_variable} by {facet_variable}"
                                    )
                st.plotly_chart(facet_fig)
            except Exception as e:
                st.warning(f"An error occurred: {e}. The selected features cannot be plotted.")
        else:
            st.warning("The selected variable for faceting is not suitable. Please choose another one.")

        st.title("Custom Visualization")
        st.markdown("""
            This section allows you to visualize the dataset based on your chart preferences. You can explore different chart types based on the feature selected.
        """)

        # Select feature
        feature_to_plot = st.selectbox("Select a feature for visualization", df.columns)

        # Define a threshold for unique value counts
        unique_threshold = 20  

        # Check if the selected feature is numeric or categorical
        if df[feature_to_plot].dtype in [np.number]:
            # Check if numeric feature has limited unique values
            if df[feature_to_plot].nunique() <= unique_threshold:
                st.warning(f"The feature '{feature_to_plot}' has a limited number of unique values and might not be ideal for scatter plots.")
                chart_type = st.selectbox("Select chart type", ["Histogram", "Box Plot", "Line Chart", "Pie Chart"])
            else:
                chart_type = st.selectbox("Select chart type", ["Histogram", "Box Plot", "Line Chart", "Scatter Plot", "Pie Chart"])

            if chart_type == "Histogram":
                fig = px.histogram(df, x=feature_to_plot)
            elif chart_type == "Box Plot":
                fig = px.box(df, y=feature_to_plot)
            elif chart_type == "Line Chart":
                fig = px.line(df, y=feature_to_plot)
            elif chart_type == "Scatter Plot":
                secondary_feature = st.selectbox("Select feature for y-axis", [col for col in df.columns if col != feature_to_plot])
                fig = px.scatter(df, x=feature_to_plot, y=secondary_feature)
            elif chart_type == "Pie Chart":
                st.warning("Pie charts are not suitable for numeric features.")
                fig = None
            if fig:
                st.plotly_chart(fig)
        else:
            # List of chart types for categorical data
            chart_type = st.selectbox("Select chart type", ["Bar Chart", "Pie Chart"])
            if chart_type == "Bar Chart":
                fig = px.bar(df, x=feature_to_plot)
            elif chart_type == "Pie Chart":
                fig = px.pie(df, names=feature_to_plot)
            if fig:
                st.plotly_chart(fig)

        st.title("Correlation Heatmap")
        st.markdown("""
            A correlation heatmap visualizes the correlation coefficients for different variables in a matrix format. This can be especially useful to understand the relationships between different features in the dataset.
        """)

        # Calculate correlations
        correlation_matrix = df[numeric_columns].corr()

        # Plot heatmap
        heatmap_fig = px.imshow(correlation_matrix, 
                                x=correlation_matrix.columns, 
                                y=correlation_matrix.columns,
                                labels=dict(color="Correlation coefficient"),
                                title="Feature Correlation Heatmap"
                                )
        st.plotly_chart(heatmap_fig)

        # Checking if there's a datetime column in the dataset
        date_cols = df.select_dtypes(include=[np.datetime64]).columns.tolist()

        if date_cols:
            if st.button("Time Series Analysis"):
                    
                # If the datetime column is not of type datetime64, convert it
                for col in date_cols:
                    df[col] = pd.to_datetime(df[col])

                # Extract day, month, month-year, and year features
                selected_date_col = st.selectbox("Select a datetime column for analysis", date_cols)
                df['Day'] = df[selected_date_col].dt.day
                df['Month'] = df[selected_date_col].dt.month
                df['Month-Year'] = df[selected_date_col].dt.to_period('M')
                df['Year'] = df[selected_date_col].dt.year

                time_feature = st.selectbox("Select a time feature for visualization", ["Day", "Month", "Month-Year", "Year"])
        
                # Line graph visualization
                fig = px.line(df, x=selected_date_col, y=time_feature, title=f"Time Series Analysis of {time_feature}")
                st.plotly_chart(fig)

        else:
            st.warning("No datetime column found in the dataset for time series analysis.")

        # Feedback section
        st.subheader("Provide Feedback")
        feedback = st.text_area("Feedback:")
        email = st.text_input("Email:")

        if st.button("Submit Feedback"):
        # Get the current timestamp
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Create a DataFrame with feedback data
            feedback_data = pd.DataFrame({
                "timestamp": [current_time],
                "email": [email],
                "feedback": [feedback]
            })  

        # If feedbacks.csv exists, append the data, otherwise create a new file
            try:
                existing_feedback = pd.read_csv("feedbacks.csv")
                all_feedback = pd.concat([existing_feedback, feedback_data])
                all_feedback.to_csv("feedbacks.csv", index=False)
            except FileNotFoundError:
                feedback_data.to_csv("feedbacks.csv", index=False)

            st.success("Thank you for your feedback!")

# Initialize session state for login status.
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

st.sidebar.title("Authentication")

# If user is not logged in, display the login form.
if not st.session_state['logged_in']:
    entered_username = st.sidebar.text_input("Username")
    entered_password = st.sidebar.text_input("Password", type="password")

    if st.sidebar.button("Login"):
        if check_credentials(entered_username, entered_password):
            st.session_state['logged_in'] = True
            st.experimental_rerun()  # Rerun the app to refresh the sidebar
        else:
            st.sidebar.error("Invalid credentials")
else:
    # If user is logged in, display the main content and a Logout button.
    main()
    
    if st.sidebar.button("Logout"):
        st.session_state['logged_in'] = False
        st.experimental_rerun()  # Rerun the app to refresh the sidebar