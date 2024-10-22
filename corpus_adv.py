import streamlit as st
import pandas as pd
import re
import spacy
from plotly import graph_objects as go
import time

# Set wide layout for the app
st.set_page_config(layout="wide")

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load CSV data on your side
@st.cache_data
def load_data():
    df = pd.read_csv("pax/pax_corpus_v8.csv")
    return df

# Function to search for terms in the corpus and return matches with context (using spaCy for sentence segmentation)
def search_terms(df, search_terms):
    results = []
    search_regex = '|'.join([re.escape(term.strip()) for term in search_terms])
    
    for idx, row in df.iterrows():
        doc = nlp(row['Agreement text'])  # Use the correct column name for the text
        context_snippets = []
        match_found = False
        term_counts = {term: 0 for term in search_terms}  # Initialize counts for each search term
        
        for sent in doc.sents:  # Segment by sentence
            sentence_text = sent.text
            matches = re.findall(search_regex, sentence_text, re.IGNORECASE)
            if matches:
                match_found = True
                context_snippets.append(sentence_text)
                for term in search_terms:
                    term_counts[term] += len(re.findall(re.escape(term), sentence_text, re.IGNORECASE))  # Count mentions of each term
        
        if match_found:
            results.append({
                'agreement_id': row['AgtId'],  # Assuming agreement has an ID
                'title': row['Agt'],           # Assuming title is a column
                'year': row['year'],           # Assuming year is a column
                'stage_label': row['stage_label'],  # Assuming stage_label is a column
                'snippet': ' | '.join(context_snippets),  # Use ' | ' as a delimiter
                'num_segments': len(context_snippets),
                'matches': ', '.join(set(re.findall(search_regex, row['Agreement text'], re.IGNORECASE))),
                'term_counts': term_counts,    # Count of each term
                'total_mentions': sum(term_counts.values())  # Total number of mentions
            })
    return pd.DataFrame(results)
# Function to plot mentions distribution and agreements by user-selected X-axis
def plot_mentions_distribution(results_df, search_terms, x_axis_field, y_axis_selection):
    # Prepare data for the X-axis
    if x_axis_field in results_df.columns:
        x_data = results_df[x_axis_field]
    else:
        st.error(f"The field {x_axis_field} is not available in the dataset.")
        return None, None

    # Prepare data for the Y-axis
    if y_axis_selection == 'Number of Agreements':
        y_data = results_df.groupby(x_axis_field).size()
    elif y_axis_selection == 'Count of Mentions':
        # Count mentions by search terms
        results_df['total_mentions'] = results_df['term_counts'].apply(lambda x: sum(x.values()))
        y_data = results_df.groupby(x_axis_field)['total_mentions'].sum()
    else:
        st.error("Invalid Y-axis selection.")
        return None, None

    # Plot the user-defined chart
    fig = go.Figure([go.Bar(x=y_data.index, y=y_data.values)])
    fig.update_layout(title=f"{y_axis_selection} by {x_axis_field}", xaxis_title=x_axis_field, yaxis_title=y_axis_selection)

    return fig

# Streamlit App
def main():
    st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&display=swap');
        
        body {
            font-family: 'Montserrat', sans-serif;
            color: #091f40;  /* Set core color for text */
        }
        
        .header-container {
            display: flex;
            align-items: center;
            justify-content: space-between;  /* Distribute space between items */
            margin-bottom: 20px;
        }
        .header-container img {
            width: 200px;  /* Adjust height as needed */
            margin: 0 20px;  /* Space around logos */
        }
        .header-title {
            text-align: center;
            flex-grow: 1;  /* Allow title to grow and take up space between logos */
            font-size: 3em;  /* Adjust title size */
            margin: 0;  /* Remove default margin */
            font-family: 'Montserrat', sans-serif;
            color: #091f40;  /* Set core color for text */
        }
        .sub-title {
            text-align: center;  /* Center the subtitle */
            font-size: 1.5em;  /* Adjust subtitle size */
            margin-top: 10px;  /* Space above subtitle */
            font-family: 'Montserrat', sans-serif;
            color: #091f40;  /* Set core color for text */
        }
    </style>
    <div class="header-container">
        <img src="https://peacerep.github.io/logos/img/PeaceRep_nobg.png" alt="PeaceRep Logo" />
        <h1 class="header-title">PA-X Peace Agreement Corpus Search</h1>
        <img src="https://peacerep.github.io/logos/img/Pax_nobg.png" alt="Logo" />
    </div>
    <div class="sub-title">
        <p><b>Credits: Bell, C., & Badanjak, S. (2019). Introducing PA-X: A new peace agreement database and dataset. Journal of Peace Research, 56(3), 452-466. Available at https://pax.peaceagreements.org/ <br>PeaceRep, University of Edinburgh</b></p>
        <p>This experimental tool allows you to search for multiple keywords and phrases to find in the PA-X corpus. To speed up processing time, filter the agreements by selecting a column and the metadata you would like to keep for the search. The results can be visualised on charts, and exported as a csv file. This is not a replacement for the search interface, rather an app to quickly multi-search the corpus and visualise the word distributions.</p>
    </div>
    """,
    unsafe_allow_html=True
)

   # Load the data on your side
    df = load_data()
    st.write("Corpus Loaded Successfully.")
    
    # Initialize filtered_df as the full dataset in case no filters are applied
    filtered_df = df.copy()
    
    # Allow users to select columns and then filter agreements based on selected columns
    st.subheader("Filter Agreements")
    
    # Step 1: Select Columns
    available_columns = df.columns.tolist()
    selected_columns = st.multiselect("Select columns to filter by", available_columns)

    # Step 2: Show value options for the selected columns
    if selected_columns:
        col1, col2 = st.columns(2)  # Use column layout for a cleaner look
        for i, col in enumerate(selected_columns):
            unique_vals = df[col].unique()
            if i % 2 == 0:
                # Display in first column
                with col1:
                    selected_vals = st.multiselect(f"Select values for {col}", options=unique_vals, default=unique_vals)
                    filtered_df = filtered_df[filtered_df[col].isin(selected_vals)]
            else:
                # Display in second column
                with col2:
                    selected_vals = st.multiselect(f"Select values for {col}", options=unique_vals, default=unique_vals)
                    filtered_df = filtered_df[filtered_df[col].isin(selected_vals)]
    
    # Show a preview of the filtered dataset
    if st.checkbox("Preview Filtered Dataset"):
        st.write(filtered_df.head())
    
    # **Search box for multiple terms with advanced options**
    st.subheader("Search Terms")
    search_terms_input = st.text_input("Enter search terms (comma-separated):", "")
    use_boolean = st.checkbox("Use Boolean Search (AND, OR, NOT)")
    phrase_search = st.checkbox("Search for exact phrases (enclose in quotes)")

    if use_boolean:
        st.info("You can use AND, OR, NOT for Boolean search. Example: 'peace AND conflict OR treaty'")
        
    if phrase_search:
        st.info("To search for exact phrases, enclose them in quotes. Example: \"peace agreement\"")

    # **Run search**
    if search_terms_input:
        # Process the input based on the selected options
        if phrase_search:
            search_term_list = [term.strip().replace('"', '') for term in search_terms_input.split(',')]
        else:
            search_term_list = [term.strip() for term in search_terms_input.split(',')]
        
        # Run search with a progress bar
        with st.spinner("Searching the corpus..."):
            start_time = time.time()
            results_df = search_terms(filtered_df, search_term_list)  # Pass filtered_df and search terms
            search_time = time.time() - start_time
            st.success(f"Search completed in {search_time:.2f} seconds.")
        
        if not results_df.empty:
            st.write(f"Found {len(results_df)} agreements with matches.")
            
            # Expand term counts into separate columns for each search term
            for term in search_term_list:
                results_df[term] = results_df['term_counts'].apply(lambda x: x.get(term, 0))
            
            # Display results with selected columns
            st.write(results_df[['agreement_id', 'title', 'year', 'snippet', 'num_segments', 'total_mentions'] + search_term_list].head(10))
            
            # Download results as CSV
            csv = results_df.to_csv(index=False)
            st.download_button(label="Download CSV", data=csv, mime='text/csv', file_name='search_results.csv')
            
            # **Text Analysis Section**
            st.subheader("Text Analysis")
            
            # Frequency Analysis
            if st.button("Analyze Frequency of Terms"):
                freq_data = Counter()
                for term in search_term_list:
                    freq_data[term] += results_df[term].sum()
                st.write("Frequency of terms:", freq_data)

            # N-gram Analysis
            ngram_size = st.number_input("Enter n-gram size (2 for bigrams, 3 for trigrams, etc.):", min_value=1, max_value=10, value=2)
            if st.button("Generate N-grams"):
                ngrams = []
                for text in results_df['snippet']:
                    tokens = text.split()  # Basic tokenization, consider using nltk or spacy for better tokenization
                    ngrams.extend(zip(*[tokens[i:] for i in range(ngram_size)]))
                ngram_counts = Counter(ngrams)
                st.write("Top N-grams:", ngram_counts.most_common(10))

            # **Plot results**
            st.subheader("Mention Distribution")
            fig, fig_year = plot_mentions_distribution(results_df, search_term_list)
            st.plotly_chart(fig)
            st.plotly_chart(fig_year)
        else:
            st.write("No matches found for the given terms.")
        
if __name__ == '__main__':
    main()