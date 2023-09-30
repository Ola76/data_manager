import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import os
import matplotlib.pyplot as plt
import base64
import io
import json
from streamlit_lottie import st_lottie
from datetime import datetime
from tabula import read_pdf
from scipy.stats import mstats
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer
from fpdf import FPDF
from io import BytesIO
import yagmail
import keyring
from keyring.backends import null

# Set the keyring backend to a dummy backend that does nothing
keyring.set_keyring(null.Keyring())


def load_lottiefile(filepath: str):
    with open (filepath, "r") as f:
        return json.load(f)

# Function to check the provided credentials.
def check_credentials(username, password):
    if username == "admin" and password == "password123":
        return True
    return False

email_password = st.secrets["general"]["email_password"]
yag = yagmail.SMTP('danieljjj32@gmail.com', email_password)


def send_email(feedback_data):
    subject = "New Feedback Received"
    content = f"""
    Timestamp: {feedback_data['timestamp'][0]}
    Email: {feedback_data['email'][0]}
    Feedback: {feedback_data['feedback'][0]}
    """
    yag.send(
    to='danieljjj32@gmail.com',
    subject=subject,
    contents=content,
    headers={'Reply-To': feedback_data['email'][0]}
)

@st.cache_data
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

@st.cache_data
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

@st.cache_data
def handle_outliers(df, method="winsorize", limits=(0.01, 0.01)):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if method == "winsorize":
        for col in numeric_cols:
            df[col] = mstats.winsorize(df[col], limits=limits)
        
        # Provide the Winsorize explanation
        st.markdown('<span title="Winsorizing is a method to limit extreme values in the data. Values more extreme than a specified range will be replaced with the nearest boundary value.">Winsorize 🛈</span>', unsafe_allow_html=True)

    return df

@st.cache_data
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

@st.cache_data
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
        
        Application by Oburoh.
        """)
    
    # Display the image
    st.sidebar.image("data_b.png", caption="Version 1.0.5", use_column_width=True)

    st.sidebar.subheader("New Update")
    st.sidebar.write("""
    1. App optimization
    2. Dynamic correlation matrix
    3. Time series bug fixed
    4. Dynamic feedback loop
    """)

    st.sidebar.subheader("Version 1.2 coming soon")
    st.sidebar.write("""
    1. Authentication development
    2. Machine learning models
    3. Prediction and Validation
    """)

    st.sidebar.write("Many more to come.")

    # Load data
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file:
        df = load_data(uploaded_file)

        # Call the cleaning function
        df = clean_categorical_data(df)

        # Initialize the session state if it doesn't exist
        if 'columns_to_drop' not in st.session_state:
            st.session_state.columns_to_drop = []

        # Use multiselect for dropping columns and store the value in a temporary variable
        selected_columns_to_drop = st.multiselect("Select columns to drop", df.columns, st.session_state.columns_to_drop)

        # If there's a change in selection, update the session_state and rerun the app
        if set(selected_columns_to_drop) != set(st.session_state.columns_to_drop):
            st.session_state.columns_to_drop = selected_columns_to_drop
            st.experimental_rerun()

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
        # If 'advanced_impute_method' and 'cleaning_method' are not in session_state, initialize them
        if 'advanced_impute_method' not in st.session_state:
            st.session_state['advanced_impute_method'] = "None"

        if 'cleaning_method' not in st.session_state:
            st.session_state['cleaning_method'] = "None"

        # Use selectbox for 'advanced_impute_method' and store the value in a temporary variable
        selected_advanced_impute = st.selectbox("Choose an advanced imputation method", ["None", "KNN", "Iterative"], index=["None", "KNN", "Iterative"].index(st.session_state['advanced_impute_method']))

        # If there's a change in selection for 'advanced_impute_method', update the session_state and rerun the app
        if selected_advanced_impute != st.session_state['advanced_impute_method']:
            st.session_state['advanced_impute_method'] = selected_advanced_impute
            st.experimental_rerun()

        if st.session_state['advanced_impute_method'] != "None":
            df = advanced_imputation(df, method=st.session_state['advanced_impute_method'])
        else:
            # Use selectbox for 'cleaning_method' and store the value in a temporary variable
            selected_cleaning_method = st.selectbox("Choose a method to clean the data", ["None", "Median", "Mean", "Most Frequent"], index=["None", "Median", "Mean", "Most Frequent"].index(st.session_state['cleaning_method']))

            # If there's a change in selection for 'cleaning_method', update the session_state and rerun the app
            if selected_cleaning_method != st.session_state['cleaning_method']:
                st.session_state['cleaning_method'] = selected_cleaning_method
                st.experimental_rerun()

            df = clean_data(df, st.session_state['cleaning_method'])

        # Displaying unique values for each column
        unique_counts = df.nunique()
        unique_df = pd.DataFrame({
            'Feature': unique_counts.index,
            'Unique Values': unique_counts.values
        })

        
        # Create buttons using the correct links
        csv_href = create_download_link(df, "csv")
        csv_button = f'<a href="{csv_href}" download="cleaned_data.csv" class="btn btn-outline-primary btn-sm">Download Cleaned Data as CSV</a>'

        st.markdown(csv_button, unsafe_allow_html=True)

        pdf_href = create_download_link(df, "pdf")
        pdf_button = f'<a href="{pdf_href}" download="cleaned_data.pdf" class="btn btn-outline-secondary btn-sm">Download Cleaned Data as PDF</a>'
        
        st.markdown(pdf_button, unsafe_allow_html=True)

        shape_data = pd.DataFrame({
            'Description': ['features', 'rows'],
            'Count': [df.shape[1], df.shape[0]]
        })

        st.table(shape_data)

        st.markdown("**Let's Begin Data Visualization!**")
        lottie_coding = load_lottiefile("amine.json")
        st_lottie(lottie_coding, speed=1, loop=True, quality="low")

        @st.cache_data
        def get_categorical_columns(df):
            return df.select_dtypes(include=['object']).columns.tolist()

        with st.expander("Categorical Data"):
            # ... Categorical Data Visualization
            st.title("Categorical Data")
            st.markdown("""
            Categorical data refers to variables that can be divided into multiple categories but do not have any order or priority. They are often non-numeric and represent characteristics such as gender, ethnicity, or regions. Unlike numeric data, they usually take on limited, and typically fixed, numbers of possible values (referred to as 'categories' or 'levels'). Understanding these variables is crucial as they can provide qualitative insights and can be encoded to numeric form for various analyses and machine learning tasks.
            """)

            # Use the cached function to get categorical columns
            categorical_columns = get_categorical_columns(df)
    
            if categorical_columns:
                selected_cat_column = st.selectbox("Select a categorical column", categorical_columns)
                fig = px.bar(df, x=selected_cat_column, title=f'Bar Chart of {selected_cat_column}')
                st.plotly_chart(fig)
            else:
                st.write("No categorical columns found in the dataset.")
    
        @st.cache_data
        def get_numeric_columns(df):
            return df.select_dtypes(include=[np.number]).columns.tolist()

        with st.expander("Numeric Data"):
            # Numeric Data Visualization
            st.title("Numeric Data")
            st.markdown("""
            Numeric data refers to variables that contain numerical values. These are typically represented by **integers** (`int`) and **floating-point numbers** (`float`). The distinction is important because numerical operations, analyses, or visualizations can be applied to these data types. Their numeric nature aids in obtaining meaningful insights and conducting quantitative analysis.
            """)

            # Use the cached function to get numeric columns
            numeric_columns = get_numeric_columns(df)
    
            if numeric_columns:
                selected_num_column = st.selectbox("Select a numeric column", numeric_columns)
                fig = px.histogram(df, x=selected_num_column, marginal='box', title=f'Distribution of {selected_num_column}')
                st.plotly_chart(fig)
            else:
                st.write("No numeric columns found in the dataset.")

        @st.cache_data
        def get_numeric_columns_without_facet(df, numeric_columns, facet_variable):
            return [col for col in numeric_columns if col != facet_variable]

        with st.expander("FacetGrid Data Visualization"):
            st.title("FacetGrid Visualization")
            st.markdown("""
                        FacetGrid visualization provides a way to visualize the distribution or relationship of data separately within subsets. This can be particularly useful when you want to explore the variation of data across different categories or levels of a certain feature.
                    """)
    
            # Choose the variable for faceting
            facet_variable = st.selectbox("Select a variable for faceting", df.columns)
    
            # Use cached function
            numeric_columns_without_facet = get_numeric_columns_without_facet(df, numeric_columns, facet_variable)

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

        @st.cache_data
        def is_feature_numeric(df, feature_to_plot):
            return df[feature_to_plot].dtype in [np.number]

        @st.cache_data
        def create_numeric_chart(df, feature_to_plot, chart_type, secondary_feature=None):
            if chart_type == "Histogram":
                return px.histogram(df, x=feature_to_plot)
            elif chart_type == "Box Plot":
                return px.box(df, y=feature_to_plot)
            elif chart_type == "Line Chart":
                return px.line(df, y=feature_to_plot)
            elif chart_type == "Scatter Plot":
                return px.scatter(df, x=feature_to_plot, y=secondary_feature)
            else:
                return None

        @st.cache_data
        def create_categorical_chart(df, feature_to_plot, chart_type):
            if chart_type == "Bar Chart":
                return px.bar(df, x=feature_to_plot)
            elif chart_type == "Pie Chart":
                return px.pie(df, names=feature_to_plot)
            else:
                return None

        with st.expander("Custom Data Visualization"):
            st.title("Custom Visualization")
            st.markdown("""
                        This section allows you to visualize the dataset based on your chart preferences. You can explore different chart types based on the feature selected.
                    """)
            feature_to_plot = st.selectbox("Select a feature for visualization", df.columns)
            unique_threshold = 20  

            if is_feature_numeric(df, feature_to_plot):
                if df[feature_to_plot].nunique() <= unique_threshold:
                    st.warning(f"The feature '{feature_to_plot}' has a limited number of unique values and might not be ideal for scatter plots.")
                    chart_type = st.selectbox("Select chart type", ["Histogram", "Box Plot", "Line Chart", "Pie Chart"])
                else:
                    chart_type = st.selectbox("Select chart type", ["Histogram", "Box Plot", "Line Chart", "Scatter Plot", "Pie Chart"])

                if chart_type == "Scatter Plot":
                    secondary_feature = st.selectbox("Select feature for y-axis", [col for col in df.columns if col != feature_to_plot])
                else:
                    secondary_feature = None

                fig = create_numeric_chart(df, feature_to_plot, chart_type, secondary_feature)
            else:
                chart_type = st.selectbox("Select chart type", ["Bar Chart", "Pie Chart"])
                fig = create_categorical_chart(df, feature_to_plot, chart_type)
    
            if fig:
                st.plotly_chart(fig)

        @st.cache_data
        def compute_correlation_matrix(df, numeric_columns, method='pearson'):
            return df[numeric_columns].corr(method=method)

        with st.expander("Correlation Matrix"):
            st.title("Correlation Heatmap")
            st.markdown("""
                A correlation heatmap visualizes the correlation coefficients for different variables in a matrix format. This can be especially useful to understand the relationships between different features in the dataset.
            """)

            # Tooltips for each correlation method
            tooltips = {
                "pearson": "Measures the linear relationship between two datasets. Suitable for continuous, normally distributed data.",
                "kendall": "A measure of rank correlation. Calculates the difference between the concordant and discordant pairs of data.",
                "spearman": "Measures the strength and direction of the monotonic relationship between two paired datasets. It's based on the ranked values of the data."
            }

            correlation_methods = ["pearson", "kendall", "spearman"]

            # Create dropdown with tooltips
            options = [f"{method} 🛈" for method in correlation_methods]
            selected_option = st.selectbox("Select Correlation Method", options, index=0)
            selected_method = selected_option.split()[0]  # Extracting the actual method without the icon

            # Displaying the tooltip for the selected method
            st.markdown(f'<span title="{tooltips[selected_method]}">{selected_method.capitalize()}</span>', unsafe_allow_html=True)

            # Slider for threshold
            threshold = st.slider("Set correlation threshold to highlight", 0.0, 1.0, 0.7)

            # Use the cached function to get the correlation matrix
            correlation_matrix = compute_correlation_matrix(df, numeric_columns, method=selected_method)

            # Highlight only correlations above or below the threshold
            highlighted_corr = correlation_matrix.mask((correlation_matrix < threshold) & (correlation_matrix > -threshold))

            # Plot heatmap with annotations for the highlighted correlations
            heatmap_fig = px.imshow(
                highlighted_corr,
                x=correlation_matrix.columns,
                y=correlation_matrix.columns,
                labels=dict(color="Correlation coefficient"),
                title="Feature Correlation Heatmap",
                color_continuous_scale="RdBu_r",  # Blue to red color scale
                zmin=-1,  # Minimum correlation value
                zmax=1   # Maximum correlation value
            )

            st.plotly_chart(heatmap_fig)

        @st.cache_data
        def try_convert_to_datetime(col):
            try:
                return pd.to_datetime(col, format='%d%m%Y')
            except:
                return col

        @st.cache_data
        def process_time_features(df, selected_date_col):
            df = df.copy()  # Clone the dataframe to avoid in-place modifications
            df['Day'] = df[selected_date_col].dt.day
            df['Month'] = df[selected_date_col].dt.month
            df['Month-Year'] = df[selected_date_col].dt.strftime('%b-%Y')  # Converts to "Mon-YYYY" format
            df['Year'] = df[selected_date_col].dt.year.astype(str)  # Convert year to string
            return df

        # Attempt to convert potential date columns to datetime format
        df = df.apply(try_convert_to_datetime)

        with st.expander("Time series analysis"):
            # Checking if there's a datetime column in the dataset
            date_cols = df.select_dtypes(include=[np.datetime64]).columns.tolist()

            if date_cols:
                if not 'time_feature' in st.session_state:
                    st.session_state['time_feature'] = "Day"  # Default value

                # Extract day, month, month-year, and year features using the cached function
                selected_date_col = st.selectbox("Select a column to analyze time series", date_cols)
                df = process_time_features(df, selected_date_col)

                # Using radio button for time feature selection and updating session state
                chosen_time_feature = st.radio("Select a time feature for visualization", ["Day", "Month", "Month-Year", "Year"])
        
                # If the chosen time feature changes, update the session state
                if st.session_state['time_feature'] != chosen_time_feature:
                    st.session_state['time_feature'] = chosen_time_feature
                    st.experimental_rerun()

                if st.session_state['time_feature'] != 'Month-Year':
                    aggregated_data = df.groupby(st.session_state['time_feature']).size().reset_index(name='Count')
                else:
                    # Special handling for "Month-Year"
                    aggregated_data = df.groupby([st.session_state['time_feature'], selected_date_col]).size().reset_index(name='Count').sort_values(by=selected_date_col)
                    aggregated_data = aggregated_data.drop(columns=selected_date_col)  # Drop the original datetime column after sorting

                fig = px.line(aggregated_data, x=st.session_state['time_feature'], y='Count', title=f"Time Series Analysis of {st.session_state['time_feature']}")
                st.plotly_chart(fig)

            else:
                st.warning("No potential datetime column found in the dataset for time series analysis.")

        # Introduce a new session state variable for tracking feedback submission
        if 'feedback_submitted' not in st.session_state:
            st.session_state.feedback_submitted = False

        if 'feedback' not in st.session_state:
            st.session_state.feedback = ''
        if 'email' not in st.session_state:
            st.session_state.email = ''

        with st.expander("Feedback"):
            st.subheader("Provide Feedback")

            # Check if feedback has been submitted
            if st.session_state.feedback_submitted:
                st.write("Thank you for your feedback!")
            else:
                # Use session_state for the input fields
                feedback = st.text_area("Feedback:", value=st.session_state.feedback)
                st.session_state.feedback = feedback  # Update session_state with the current input
        
                email = st.text_input("Email:", value=st.session_state.email)
                st.session_state.email = email  # Update session_state with the current input

                if st.button("Submit Feedback"):
                    # Get the current timestamp
                    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    # Create a DataFrame with feedback data
                    feedback_data = pd.DataFrame({
                        "timestamp": [current_time],
                        "email": [st.session_state.email],
                        "feedback": [st.session_state.feedback]
                    })

                    # If feedbacks.csv exists, append the data, otherwise create a new file
                    try:
                        existing_feedback = pd.read_csv("feedbacks.csv")
                        all_feedback = pd.concat([existing_feedback, feedback_data])
                        all_feedback.to_csv("feedbacks.csv", index=False)
                    except FileNotFoundError:
                        feedback_data.to_csv("feedbacks.csv", index=False)

                    # Send the feedback data via email
                    send_email(feedback_data)

                    st.success("Thank you for your feedback!")

                    # Set feedback_submitted to True after successful feedback processing
                    st.session_state.feedback_submitted = True

                    # Reset session state variables
                    st.session_state.feedback = ''
                    st.session_state.email = ''
                
                    # Rerun the app to reset the input widgets
                    st.experimental_rerun()

# Initialize session state for login status.
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

st.sidebar.title("Authentication")

# If user is not logged in, display the login form.
if not st.session_state['logged_in']:
    # Display the image
    st.image("data.png", caption="Data Manager 1.0.5", use_column_width=True)
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