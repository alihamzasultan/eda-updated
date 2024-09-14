import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import time
import google.generativeai as genai

st.markdown('<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css">', unsafe_allow_html=True)

  # Replace with your actual API key
api_key = "AIzaSyCQYNe42dTLNZrS9Wdi6dYqEW19zN1s_8M"

    # Check if the API key is provided
if api_key == "YOUR_API_KEY_HERE":
    st.error("Please replace 'YOUR_API_KEY_HERE' with your actual API key.")
    st.stop()
else:
    genai.configure(api_key=api_key)

HARD_CODED_PROMPT = (
        "You are a friendly and knowledgeable chatbot designed to assist users with their data."
        "You will tell the user all the problems that the dataset may have (e.g., nullvalues, normalization etc )"
        "You will provide insights from the data, suggest appropriate methods for preprocessing, and recommend algorithms or techniques for further analysis."
        "make some tables if necessary for better understanding"
        "Suggest the most appropriate algorithms based on the data type (e.g., classification, regression)."
        "For classification problems, recommend the best classification algorithms (e.g., decision trees, random forests, SVMs) and explain why they are suitable."
        "For regression or prediction tasks, suggest appropriate algorithms (e.g., linear regression, neural networks) and explain their relevance."
        "Make sure to guide the user through each step and provide clear, concise recommendations, ensuring that your suggestions align with the type and structure of the data."
)

state = {'show_result': False, 'knn_inputs': {}, 'user_inputs': {}}

def read_uploaded_file(file):
    if file.name.endswith('.csv'):
        return pd.read_csv(file)
    elif file.name.endswith('.xlsx'):
        return pd.read_excel(file)
    return None

def upload_file():
    st.header("Upload your CSV data file")
    data_file = st.file_uploader("Upload CSV", type=["csv"])

    if data_file is not None:
        data = pd.read_csv(data_file)
        return data
    return None

    # Function to get response from Gemini API with retry logic
def get_gemini_response(input_text, context, retries=3, delay=2):
        attempt = 0
        while attempt < retries:
            try:
                model = genai.GenerativeModel('gemini-1.5-flash')
                response = model.generate_content([context, input_text])
                return response.text.strip()
            except Exception as e:
                if "quota" in str(e).lower() or "500" in str(e):
                    st.warning(f"Oops! Slow down. Trying again in {delay} seconds...")
                    attempt += 1
                    time.sleep(delay)
                else:
                    st.error("Something went wrong. Please try again later.")
                    return None
        st.error("We're unable to process your request right now. Please try again later.")
        return None

 
def main():

    st.title("Hello, Analyst!")
    st.write("Scroll down to use the Smart AI-Assistant")
    st.sidebar.title("Visualize & Preprocess")

    uploaded_file = st.file_uploader("Click to Upload", type=["csv", "xlsx"])

    # Read data from the uploaded file
    data = None
    if uploaded_file is not None:
        data = read_uploaded_file(uploaded_file)

    if data is not None:
        max_num_rows = len(data)
        st.write("Data overview:")
        st.write(data.head())  # Display the entire dataset
        
        # Slider for selecting the number of rows
        num_rows = st.sidebar.slider("Select number of rows to use", min_value=100, max_value=max_num_rows, value=1000)
        
        st.write("Data overview (first {} rows):".format(num_rows))
        st.write(data.head(num_rows))  # Display the selected number of rows

        st.header("Model Selection")

        model_type = st.selectbox("Select Model", ["KNN Classification", "Naive Bayes", "Linear Regression", "Logistic Regression", "Decision Tree", "Random Forest"])

         

        if model_type == "KNN Classification":
            apply_knn_classification(data, state)
        elif model_type == "Naive Bayes":
            apply_naive_bayes(data)
        elif model_type == "Linear Regression":
            apply_linear_regression(data)
        elif model_type == "Logistic Regression":
            apply_logistic_regression(data)
        elif model_type == "Decision Tree":
            apply_decision_tree(data)
        elif model_type == "Random Forest":
            apply_random_forest(data)

        st.sidebar.header("Visualizations")
        plot_options = ["Bar plot", "Scatter plot", "Histogram", "Box plot", "Heatmap", "Line plot", "Pie chart", "Area plot", "Violin plot", "Density contour plot","Map plot"]
        selected_plot = st.sidebar.selectbox("Choose a plot type", plot_options)
        selected_columns = []

        if selected_plot in ["Bar plot", "Scatter plot", "Line plot"]:
            selected_columns.append(st.sidebar.selectbox("Select x-axis", data.columns, key=f"{selected_plot}_x_axis"))
            selected_columns.append(st.sidebar.selectbox("Select y-axis", data.columns, key=f"{selected_plot}_y_axis"))

        if selected_plot == "Histogram":
            selected_columns.append(st.sidebar.selectbox("Select a column", data.columns, key="histogram_column"))

        if selected_plot == "Box plot":
            selected_columns.append(st.sidebar.selectbox("Select a column", data.columns, key="boxplot_column"))

        if selected_plot == "Pie chart":
            selected_columns.append(st.sidebar.selectbox("Select a column", data.columns, key="pie_column"))

        if selected_plot == "Area plot":
            selected_columns.append(st.sidebar.selectbox("Select x-axis", data.columns, key="area_x_axis"))
            selected_columns.append(st.sidebar.selectbox("Select y-axis", data.columns, key="area_y_axis"))

        if selected_plot == "Violin plot":
            selected_columns.append(st.sidebar.selectbox("Select x-axis", data.columns, key="violin_x_axis"))
            selected_columns.append(st.sidebar.selectbox("Select y-axis", data.columns, key="violin_y_axis"))

        if selected_plot == "Density contour plot":
            selected_columns.append(st.sidebar.selectbox("Select x-axis", data.columns, key="density_x_axis"))
            selected_columns.append(st.sidebar.selectbox("Select y-axis", data.columns, key="density_y_axis"))

        if selected_plot == "Bar plot":
            st.write("Bar plot:")
            fig = px.bar(data, x=selected_columns[0], y=selected_columns[1])
            st.plotly_chart(fig)


        elif selected_plot == "Scatter plot":
            st.write("Scatter plot:")
            fig = px.scatter(data, x=selected_columns[0], y=selected_columns[1])
            st.plotly_chart(fig)

        elif selected_plot == "Histogram":
            st.write("Histogram:")
            fig = px.histogram(data, x=selected_columns[0])
            st.plotly_chart(fig)

        elif selected_plot == "Box plot":
            st.write("Box plot:")
            fig = px.box(data, x=selected_columns[0])
            st.plotly_chart(fig)

        elif selected_plot == "Heatmap":
            st.write("Heatmap:")
            fig = px.imshow(data.corr(), color_continuous_scale='thermal')
            st.plotly_chart(fig)

        elif selected_plot == "Line plot":
            st.write("Line plot:")
            fig = px.line(data, x=selected_columns[0], y=selected_columns[1])
            st.plotly_chart(fig)

        elif selected_plot == "Pie chart":
            st.write("Pie chart:")
            fig = px.pie(data, names=selected_columns[0])
            st.plotly_chart(fig)

        elif selected_plot == "Area plot":
            st.write("Area plot:")
            fig = px.area(data, x=selected_columns[0], y=selected_columns[1])
            st.plotly_chart(fig)

        elif selected_plot == "Violin plot":
            st.write("Violin plot:")
            fig = px.violin(data, x=selected_columns[0], y=selected_columns[1])
            st.plotly_chart(fig)

        elif selected_plot == "Density contour plot":
            st.write("Density contour plot:")
            fig = px.density_contour(data, x=selected_columns[0], y=selected_columns[1])
            st.plotly_chart(fig)



        display_null_values_and_datatype(data)

        fill_null_values_section(data)
        drop_null_values_section(data)
        rename_column_section(data)
        change_data_type_section(data)
        normalization_section(data)
        replace_column_value(data)


        
    # Limit the number of rows to summarize for the Gemini API to avoid large inputs
    def summarize_data(data, max_rows=10):
        try:
            # Get a summary with limited rows and describe the data
            summary = data.head(max_rows).to_string()
            description = data.describe(include='all').to_string()
            return f"Data Summary (first {max_rows} rows):\n{summary}\n\nData Description:\n{description}"
        except Exception as e:
            st.error(f"Error summarizing data: {e}")
            return ""



    with st.expander("AI-Chat Assistant"):
                    # Initialize session state for conversation history
                if 'conversation' not in st.session_state:
                    st.session_state.conversation = []

                
                st.title("AI-Chat Section")


                user_prompt = st.text_area("Ask the AI about your dataset:", placeholder="Type your question related to the dataset...")

                    # Section for uploading CSV or Excel in the sidebar

                if st.button("🔎"):
                    if user_prompt:
                        with st.spinner("Generating response..."):
                            # Ensure data is uploaded
                            if data is not None:
                                try:
                                    data_summary = summarize_data(data, max_rows=10)
                                    st.subheader("Response")
                                except Exception as e:
                                    st.error(f"Error summarizing data: {e}")
                                    data_summary = ""
                            else:
                                data_summary = ""  # No data to summarize

                            # Combine user prompt with hardcoded prompt and file content summary
                            combined_prompt = f"{HARD_CODED_PROMPT}\n\nUser Prompt:\n{user_prompt}\n\nFile Data Summary:\n{data_summary}"
                            
                            # Get response from Gemini API
                            response = get_gemini_response(user_prompt, combined_prompt)

                            if response:
                                # Append user prompt and response to chat history
                                st.session_state.conversation.append({"user": user_prompt, "bot": response})
                            else:
                                st.error("Failed to generate response.")
                    else:
                        st.error("Please enter a prompt.")


                # Display the latest message first, and keep the history in an expander
                if st.session_state.conversation:
                    # Get the latest chat (last element of the conversation)
                    latest_chat = st.session_state.conversation[-1]
                    
                    # Style for the latest response
                    user_style = "background-color: #FFffff; padding: 8px; border-radius: 10px; color: black; max-width: 80%;"
                    bot_style = "background-color: #FFFfff; padding: 8px; border-radius: 10px; color: black; max-width: 80%;"

                    # Display user's latest message without an icon
                    st.markdown(f"""
                    <div style="display: flex; justify-content: flex-end; align-items: center; margin: 10px 0;">
                        <div style="{user_style}">
                            <strong>👤  :</strong> {latest_chat['user']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display bot's latest message with an icon
                    st.markdown(f"""
                    <div style="display: flex; justify-content: flex-start; align-items: center; margin: 10px 0;">
                        <div style="margin-right: 10px;">
                            <img src="https://img.icons8.com/?size=100&id=L3uh0mNuxBXw&format=png&color=000000" width="50">
                        </div>
                        <div style="{bot_style}">
                            <strong></strong> {latest_chat['bot']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                        # Add a dropdown (expander) to show previous conversation history
                    
                            # Display the conversation history in reverse order
                    conversation_reversed = reversed(st.session_state.conversation[:-1])  # Exclude the latest chat
                    for i, chat in enumerate(conversation_reversed):
                                # Style for other responses
                                user_style = "background-color: black; padding: 8px; border-radius: 10px; color: white; max-width: 80%;"
                                bot_style = "background-color: black; padding: 8px; border-radius: 10px; color: white; max-width: 80%;"

                                # Display user's previous message
                                st.markdown(f"""
                                <div style="display: flex; justify-content: flex-end; align-items: center; margin: 10px 0;">
                                    <div style="{user_style}">
                                        <strong></strong> {chat['user']}
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Display bot's previous message with an icon
                                st.markdown(f"""
                                <div style="display: flex; justify-content: flex-start; align-items: center; margin: 10px 0;">
                                    <div style="margin-right: 10px;">
                                        <img src="https://img.icons8.com/?size=100&id=L3uh0mNuxBXw&format=png&color=000000" width="50">
                                    </div>
                                    <div style="{bot_style}">
                                        <strong></strong> {chat['bot']}
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)


            


  


def display_null_values_and_datatype(data):
    null_counts = data.isnull().sum()
    datatypes = data.dtypes
    null_table = pd.DataFrame({'Null Values': null_counts, 'Data Type': datatypes})
    st.sidebar.write("Number of null values and datatype in each column:")
    st.sidebar.table(null_table)


def fill_null_values_section(data):
    st.sidebar.header("Fill Null Values")
    fill_methods = ["Mean", "Median", "Mode", "Most Frequent", "Custom Value"]
    selected_fill_method = st.sidebar.selectbox("Choose a fill method", fill_methods)
    if selected_fill_method == "Custom Value":
        custom_value = st.sidebar.text_input("Enter custom value")
    else:
        custom_value = None

    if st.sidebar.button("Fill Null Values"):
        data = fill_null_values(data, selected_fill_method, custom_value)
        st.sidebar.success("Null values filled successfully.")
        display_null_values_and_datatype(data)


def drop_null_values_section(data):
    st.sidebar.header("Drop Null Values")
    if st.sidebar.button("Drop Null Values"):
        data = drop_null_values(data)
        st.sidebar.success("Null values dropped successfully.")
        display_null_values_and_datatype(data)


def rename_column_section(data):
    data2 = None
    st.sidebar.header("Change Column Name")
    old_column_name = st.sidebar.selectbox("Select a column", data.columns, key="rename_column_select")
    new_column_name = st.sidebar.text_input("Enter new column name", key="rename_new_column_name")
    if st.sidebar.button("Change Column Name"):
        data = rename_column(data, old_column_name, new_column_name)
        st.sidebar.success(f"Column '{old_column_name}' renamed to '{new_column_name}' successfully.")
        st.write("Data overview (after renaming column):")
        st.write(data.head())


def change_data_type_section(data):
    st.sidebar.header("Change Data Type")
    column_to_change = st.sidebar.selectbox("Select a column", data.columns, key="change_datatype_select")
    new_data_type = st.sidebar.selectbox("Select new data type", ["int64", "float64", "object", "datetime64", "bool", "string"], key="change_datatype_new")
    if st.sidebar.button("Change Data Type"):
        data = change_data_type(data, column_to_change, new_data_type)
        st.sidebar.success(f"Data type of column '{column_to_change}' changed to '{new_data_type}' successfully.")
        display_null_values_and_datatype(data)





def normalization_section(data):
    st.sidebar.header("Normalization")
    normalization_options = ["Standard Scaler", "Min-Max Scaler", "Robust Scaler"]
    selected_normalization = st.sidebar.selectbox("Select normalization technique", normalization_options)

    if selected_normalization:
        selected_columns = st.sidebar.multiselect("Select columns to normalize", data.columns)

        if st.sidebar.button("Normalize Data"):
            if not selected_columns:
                st.sidebar.error("Please select at least one column to normalize.")
            else:
                if selected_normalization == "Standard Scaler":
                    data = standard_scaler_normalization(data, selected_columns)
                    st.sidebar.success("Data normalized using Standard Scaler.")
                elif selected_normalization == "Min-Max Scaler":
                    data = min_max_scaler_normalization(data, selected_columns)
                    st.sidebar.success("Data normalized using Min-Max Scaler.")
                elif selected_normalization == "Robust Scaler":
                    data = robust_scaler_normalization(data, selected_columns)
                    st.sidebar.success("Data normalized using Robust Scaler.")

                st.write("Data overview (after normalization):")
                st.write(data.head())


def standard_scaler_normalization(data, column):
    scaler = StandardScaler()
    data[column] = scaler.fit_transform(data[[column]])
    return data


def min_max_scaler_normalization(data, column):
    scaler = MinMaxScaler()
    data[column] = scaler.fit_transform(data[[column]])
    return data


def robust_scaler_normalization(data, column):
    scaler = RobustScaler()
    data[column] = scaler.fit_transform(data[[column]])
    return data


def fill_null_values(data, method, custom_value=None):
    for column in data.columns:
        if data[column].dtype == 'object':
            if method == "Custom Value" and custom_value is not None:
                data[column].fillna(custom_value, inplace=True)
            elif method == "Most Frequent":
                most_frequent_value = data[column].mode().iloc[0]
                data[column].fillna(most_frequent_value, inplace=True)
        else:
            if method == "Mean":
                data[column].fillna(data[column].mean(), inplace=True)
            elif method == "Median":
                data[column].fillna(data[column].median(), inplace=True)
            elif method == "Mode":
                data[column].fillna(data[column].mode().iloc[0], inplace=True)
            elif method == "Custom Value" and custom_value is not None:
                data[column].fillna(custom_value, inplace=True)
    return data


def drop_null_values(data):
    return data.dropna()


def rename_column(data, old_column_name, new_column_name):
    data.rename(columns={old_column_name: new_column_name}, inplace=True)
    return data


def change_data_type(data, column, new_type):
    if new_type == "datetime64":
        new_type = "datetime64[ns]"  # Correct format for datetime
    elif new_type == "string":
        new_type = "object"  # pandas dtype for string is 'object'

    data[column] = data[column].astype(new_type)
    return data
def replace_column_value(data):
    st.sidebar.header("Replace Value in Column")
    column_to_replace = st.sidebar.selectbox("Select a column", data.columns, key="replace_column_select")
    value_to_replace = st.sidebar.text_input("Enter value to replace", key="replace_value_input")
    replace_with = st.sidebar.text_input("Replace with", key="replace_with_input")

    if st.sidebar.button("Replace Value"):
        # Convert the value to the same data type as in the dataset
        dtype = data[column_to_replace].dtype
        try:
            if dtype == 'object':
                value_to_replace = str(value_to_replace)
            elif dtype == 'int64':
                value_to_replace = int(value_to_replace)
            elif dtype == 'float64':
                value_to_replace = float(value_to_replace)
            elif dtype == 'bool':
                value_to_replace = bool(value_to_replace)
            elif dtype == 'datetime64':
                value_to_replace = pd.to_datetime(value_to_replace)
        except ValueError:
            st.sidebar.error("Failed to convert the value to the correct data type.")
            return

        # Check if the value to be replaced exists in the selected column
        if value_to_replace not in data[column_to_replace].values:
            st.sidebar.error(f"The value '{value_to_replace}' does not exist in column '{column_to_replace}'.")
            return

        # Replace the value in the selected column
        data[column_to_replace].replace(value_to_replace, replace_with, inplace=True)

        # Display success message
        st.sidebar.success(f"Value '{value_to_replace}' in column '{column_to_replace}' replaced with '{replace_with}' successfully.")

        # Display the updated DataFrame
        st.write("Data overview (after replacing value):")
        st.write(data.head())  # Display the updated DataFrame


def apply_knn_classification(data, state):
    if 'show_result' not in state:
        state['show_result'] = False
    if 'knn_inputs' not in state:
        state['knn_inputs'] = {}

    st.header("KNN Classification")

    target_column = st.selectbox("Select the target column", data.columns, key="knn_target_column")
    feature_columns = st.multiselect("Select the feature columns", data.columns, key="knn_feature_columns")
    test_size = st.slider("Select test size", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
    random_state = st.slider("Select random state", min_value=0, max_value=100, value=42, step=1)
    enable_user_inputs = st.radio("Enable user inputs", ["Yes", "No"])

    if enable_user_inputs == "Yes":
        k_value = st.slider("Select the value of k", min_value=1, max_value=10, value=5, step=1)

        # Collect user inputs for each feature column
        user_inputs = {}
        for column in feature_columns:
            user_input = st.number_input(f"Enter value for '{column}'", key=f"knn_input_{column}")
            user_inputs[column] = user_input

        if st.button("Apply KNN Classifier"):
            if len(feature_columns) < 1:
                st.error("Please select at least one feature column.")
                return

            # Store user inputs in state
            state['user_inputs'] = user_inputs

            # Create a DataFrame with user inputs
            user_df = pd.DataFrame([user_inputs])

            # Load the dataset again to ensure consistency in column ordering
            X = data[feature_columns]
            y = data[target_column]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

            # Train KNN model
            knn_model = KNeighborsClassifier(n_neighbors=k_value)
            knn_model.fit(X_train, y_train)

            # Predict the target value for user inputs
            predicted_value = knn_model.predict(user_df)

            # Display the predicted value along with user inputs
            st.write("Predicted Target Value and User Inputs:")
            user_inputs_df = pd.DataFrame([user_inputs])
            predicted_value_df = pd.DataFrame({target_column: predicted_value})
            combined_df = pd.concat([user_inputs_df, predicted_value_df], axis=1)
            st.table(combined_df)

            # Predict the target values for test data
            y_pred = knn_model.predict(X_test)

            # Calculate accuracy on test data if available
            accuracy = accuracy_score(y_test, y_pred)
            st.write(f"Accuracy: {accuracy:.2f}")

            # Display classification report for test data
            st.write("Classification Report:")
            classification_rep = classification_report(y_test, y_pred, output_dict=True)
            df_classification_rep = pd.DataFrame(classification_rep).transpose()
            st.table(df_classification_rep)

            # Update state to show the result
            state['show_result'] = True

    elif enable_user_inputs == "No":
        if st.button("Apply KNN Classifier"):
            if len(feature_columns) < 1:
                st.error("Please select at least one feature column.")
                return

            # Load the dataset again to ensure consistency in column ordering
            X = data[feature_columns]
            y = data[target_column]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

            # Train KNN model
            knn_model = KNeighborsClassifier()
            knn_model.fit(X_train, y_train)

            # Predict the target values for test data
            y_pred = knn_model.predict(X_test)

            # Calculate accuracy on test data if available
            accuracy = accuracy_score(y_test, y_pred)
            st.write(f"Accuracy: {accuracy:.2f}")

            # Display classification report for test data
            st.write("Classification Report:")
            classification_rep = classification_report(y_test, y_pred, output_dict=True)
            df_classification_rep = pd.DataFrame(classification_rep).transpose()
            st.table(df_classification_rep)

            # Update state to show the result
            state['show_result'] = True





def apply_naive_bayes(data):
    st.header("Naive Bayes")
    
    target_column = st.selectbox("Select the target column", data.columns, key="nb_target_column")
    feature_columns = st.multiselect("Select the feature columns", data.columns, key="nb_feature_columns")
    test_size = st.slider("Select test size", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
    random_state = st.slider("Select random state", min_value=0, max_value=100, value=42, step=1)
    enable_user_inputs = st.radio("Enable user inputs", ["Yes", "No"])

    if enable_user_inputs == "Yes":
        # Collect user inputs for each feature column
        user_inputs = {}
        for column in feature_columns:
            user_input = st.number_input(f"Enter value for '{column}'", key=f"nb_input_{column}")
            user_inputs[column] = user_input

        if st.button("Apply Naive Bayes"):
            if len(feature_columns) < 1:
                st.error("Please select at least one feature column.")
                return

            # Create a DataFrame with user inputs
            user_df = pd.DataFrame([user_inputs])

            X = data[feature_columns]
            y = data[target_column]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

            nb_model = GaussianNB()
            nb_model.fit(X_train, y_train)

            y_pred = nb_model.predict(user_df)

            # Display the predicted value along with user inputs
            st.write("Predicted Target Value and User Inputs:")
            user_inputs_df = pd.DataFrame([user_inputs])
            predicted_value_df = pd.DataFrame({target_column: y_pred})
            combined_df = pd.concat([user_inputs_df, predicted_value_df], axis=1)
            st.table(combined_df)

            # Predict the target values for test data
            y_pred_test = nb_model.predict(X_test)

            # Calculate accuracy on test data if available
            accuracy = accuracy_score(y_test, y_pred_test)
            st.write(f"Accuracy on Test Data: {accuracy:.2f}")

            # Display classification report for test data
            st.write("Classification Report on Test Data:")
            classification_rep = classification_report(y_test, y_pred_test, output_dict=True)
            df_classification_rep = pd.DataFrame(classification_rep).transpose()
            st.table(df_classification_rep)

    elif enable_user_inputs == "No":
        if st.button("Apply Naive Bayes"):
            if len(feature_columns) < 1:
                st.error("Please select at least one feature column.")
                return

            X = data[feature_columns]
            y = data[target_column]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

            nb_model = GaussianNB()
            nb_model.fit(X_train, y_train)

            y_pred = nb_model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            st.write(f"Accuracy: {accuracy:.2f}")

            classification_rep = classification_report(y_test, y_pred, output_dict=True)
            df_classification_rep = pd.DataFrame(classification_rep).transpose()
            st.write("Classification Report:")
            st.table(df_classification_rep)


def apply_linear_regression(data):
    st.header("Linear Regression")

    target_column = st.selectbox("Select the target column", data.columns, key="linear_target_column")
    feature_columns = st.multiselect("Select the feature columns", data.columns, key="linear_feature_columns")
    test_size = st.slider("Select test size", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
    random_state = st.slider("Select random state", min_value=0, max_value=100, value=42, step=1)
    enable_user_inputs = st.radio("Enable user inputs", ["Yes", "No"])

    if enable_user_inputs == "Yes":
        # Collect user inputs for each feature column
        user_inputs = {}
        for column in feature_columns:
            user_input = st.number_input(f"Enter value for '{column}'", key=f"linear_input_{column}")
            user_inputs[column] = user_input

        if st.button("Apply Linear Regression"):
            if len(feature_columns) < 1:
                st.error("Please select at least one feature column.")
                return

            # Create a DataFrame with user inputs
            user_df = pd.DataFrame([user_inputs])

            X = data[feature_columns]
            y = data[target_column]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

            linear_model = LinearRegression()
            linear_model.fit(X_train, y_train)

            y_pred = linear_model.predict(user_df)

            st.write("Predicted Target Value based on User Inputs:")
            st.write(y_pred)

            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            st.write(f"Mean Squared Error on Test Data: {mse:.2f}")
            st.write(f"R-squared on Test Data: {r2:.2f}")

    elif enable_user_inputs == "No":
        if st.button("Apply Linear Regression"):
            if len(feature_columns) < 1:
                st.error("Please select at least one feature column.")
                return

            X = data[feature_columns]
            y = data[target_column]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

            linear_model = LinearRegression()
            linear_model.fit(X_train, y_train)

            y_pred = linear_model.predict(X_test)

            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            st.write(f"Mean Squared Error on Test Data: {mse:.2f}")
            st.write(f"R-squared on Test Data: {r2:.2f}")



def apply_logistic_regression(data):
    st.header("Logistic Regression")

    target_column = st.selectbox("Select the target column", data.columns, key="logistic_target_column")
    feature_columns = st.multiselect("Select the feature columns", data.columns, key="logistic_feature_columns")
    test_size = st.slider("Select test size", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
    random_state = st.slider("Select random state", min_value=0, max_value=100, value=42, step=1)
    enable_user_inputs = st.radio("Enable user inputs", ["Yes", "No"])

    if enable_user_inputs == "Yes":
        # Collect user inputs for each feature column
        user_inputs = {}
        for column in feature_columns:
            user_input = st.number_input(f"Enter value for '{column}'", key=f"logistic_input_{column}")
            user_inputs[column] = user_input

        if st.button("Apply Logistic Regression"):
            if len(feature_columns) < 1:
                st.error("Please select at least one feature column.")
                return

            # Create a DataFrame with user inputs
            user_df = pd.DataFrame([user_inputs])

            X = data[feature_columns]
            y = data[target_column]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

            logistic_model = LogisticRegression()
            logistic_model.fit(X_train, y_train)

            y_pred = logistic_model.predict(user_df)

            st.write("Predicted Target Value based on User Inputs:")
            st.write(y_pred)

            accuracy = accuracy_score(y_test, y_pred)
            st.write(f"Accuracy on Test Data: {accuracy:.2f}")

            classification_rep = classification_report(y_test, y_pred, output_dict=True)
            df_classification_rep = pd.DataFrame(classification_rep).transpose()
            st.write("Classification Report on Test Data:")
            st.table(df_classification_rep)

    elif enable_user_inputs == "No":
        if st.button("Apply Logistic Regression"):
            if len(feature_columns) < 1:
                st.error("Please select at least one feature column.")
                return

            X = data[feature_columns]
            y = data[target_column]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

            logistic_model = LogisticRegression()
            logistic_model.fit(X_train, y_train)

            y_pred = logistic_model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            st.write(f"Accuracy on Test Data: {accuracy:.2f}")

            classification_rep = classification_report(y_test, y_pred, output_dict=True)
            df_classification_rep = pd.DataFrame(classification_rep).transpose()
            st.write("Classification Report on Test Data:")
            st.table(df_classification_rep)


def apply_decision_tree(data):
    st.header("Decision Tree")

    target_column = st.selectbox("Select the target column", data.columns, key="dt_target_column")
    feature_columns = st.multiselect("Select the feature columns", data.columns, key="dt_feature_columns")
    test_size = st.slider("Select test size", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
    random_state = st.slider("Select random state", min_value=0, max_value=100, value=42, step=1)
    enable_user_inputs = st.radio("Enable user inputs", ["Yes", "No"])

    if enable_user_inputs == "Yes":
        # Collect user inputs for each feature column
        user_inputs = {}
        for column in feature_columns:
            user_input = st.number_input(f"Enter value for '{column}'", key=f"dt_input_{column}")
            user_inputs[column] = user_input

        if st.button("Apply Decision Tree"):
            if len(feature_columns) < 1:
                st.error("Please select at least one feature column.")
                return

            # Create a DataFrame with user inputs
            user_df = pd.DataFrame([user_inputs])

            X = data[feature_columns]
            y = data[target_column]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

            dt_model = DecisionTreeClassifier()
            dt_model.fit(X_train, y_train)

            y_pred = dt_model.predict(user_df)

            st.write("Predicted Target Value based on User Inputs:")
            st.write(y_pred)

            accuracy = accuracy_score(y_test, y_pred)
            st.write(f"Accuracy on Test Data: {accuracy:.2f}")

            classification_rep = classification_report(y_test, y_pred, output_dict=True)
            df_classification_rep = pd.DataFrame(classification_rep).transpose()
            st.write("Classification Report on Test Data:")
            st.table(df_classification_rep)

    elif enable_user_inputs == "No":
        if st.button("Apply Decision Tree"):
            if len(feature_columns) < 1:
                st.error("Please select at least one feature column.")
                return

            X = data[feature_columns]
            y = data[target_column]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

            dt_model = DecisionTreeClassifier()
            dt_model.fit(X_train, y_train)

            y_pred = dt_model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            st.write(f"Accuracy on Test Data: {accuracy:.2f}")

            classification_rep = classification_report(y_test, y_pred, output_dict=True)
            df_classification_rep = pd.DataFrame(classification_rep).transpose()
            st.write("Classification Report on Test Data:")
            st.table(df_classification_rep)



def apply_random_forest(data):
    st.header("Random Forest")

    target_column = st.selectbox("Select the target column", data.columns, key="rf_target_column")
    feature_columns = st.multiselect("Select the feature columns", data.columns, key="rf_feature_columns")
    test_size = st.slider("Select test size", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
    random_state = st.slider("Select random state", min_value=0, max_value=100, value=42, step=1)
    n_estimators = st.slider("Select the number of estimators", min_value=1, max_value=100, value=10, step=1)
    enable_user_inputs = st.radio("Enable user inputs", ["Yes", "No"])

    if enable_user_inputs == "Yes":
        # Collect user inputs for each feature column
        user_inputs = {}
        for column in feature_columns:
            user_input = st.number_input(f"Enter value for '{column}'", key=f"rf_input_{column}")
            user_inputs[column] = user_input

        if st.button("Apply Random Forest"):
            if len(feature_columns) < 1:
                st.error("Please select at least one feature column.")
                return

            # Create a DataFrame with user inputs
            user_df = pd.DataFrame([user_inputs])

            X = data[feature_columns]
            y = data[target_column]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

            rf_model = RandomForestClassifier(n_estimators=n_estimators)
            rf_model.fit(X_train, y_train)

            y_pred = rf_model.predict(user_df)

            st.write("Predicted Target Value based on User Inputs:")
            st.write(y_pred)

            accuracy = accuracy_score(y_test, y_pred)
            st.write(f"Accuracy on Test Data: {accuracy:.2f}")

            classification_rep = classification_report(y_test, y_pred, output_dict=True)
            df_classification_rep = pd.DataFrame(classification_rep).transpose()

            # Convert the "support" column to integers
            df_classification_rep["support"] = df_classification_rep["support"].astype(int)

            st.write("Classification Report on Test Data:")
            st.table(df_classification_rep)

    elif enable_user_inputs == "No":
        if st.button("Apply Random Forest"):
            if len(feature_columns) < 1:
                st.error("Please select at least one feature column.")
                return

            X = data[feature_columns]
            y = data[target_column]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

            rf_model = RandomForestClassifier(n_estimators=n_estimators)
            rf_model.fit(X_train, y_train)

            y_pred = rf_model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            st.write(f"Accuracy on Test Data: {accuracy:.2f}")

            classification_rep = classification_report(y_test, y_pred, output_dict=True)
            df_classification_rep = pd.DataFrame(classification_rep).transpose()

            # Convert the "support" column to integers
            df_classification_rep["support"] = df_classification_rep["support"].astype(int)

            st.write("Classification Report on Test Data:")
            st.table(df_classification_rep)


def footer():

  html_temp = """
  <div style="position: fixed; bottom: 50px; width: 100%; text-align: center; font-weight: bold;">
    <p style="margin-bottom: 5px; font-size: 14px;">
      Copyright &copy; Made By <span style="color: #007bff; font-weight: bold;">AliHamzaSultan</span>
      <a href="https://www.linkedin.com/in/ali-hamza-sultan-1ba7ba267/" target="_blank" style="margin-left: 10px;"><i class="fab fa-linkedin" style="font-size: 20px;"></i></a>
      <a href="https://github.com/alihamzasultan" target="_blank"><i class="fab fa-github" style="font-size: 20px; margin-left: 10px;"></i></a>
    </p>
  </div>
  """
  st.markdown(html_temp, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

footer()






