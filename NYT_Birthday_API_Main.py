import csv
import matplotlib.pyplot as plot
import pandas as pd
import requests
import spacy
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords

# Load NLP processing pipeline
nlp = spacy.load("en_core_web_sm")
nltk.download("vader_lexicon")
analyzer = SentimentIntensityAnalyzer()


def get_articles(api_key, year, month):
    base_url = "https://api.nytimes.com/svc/archive/v1"
    month_url = f"{month}.json"
    url = f"{base_url}/{year}/{month_url}"

    query = {"api-key": api_key}

    try:
        response = requests.get(url, params=query)
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.json().get('response', {}).get('docs', [])
    except requests.exceptions.RequestException as e:
        print(f"Error making API request: {e}")
        return []

def filter_articles_by_dob(articles, dob):
    dob_articles = []
    for article in articles:
        publication_date = article.get('pub_date', '')
        if publication_date.startswith(dob):
            dob_articles.append(article)
    return dob_articles

def save_frontpage_to_csv(frontpage_df, dob, filename):
    try:
        frontpage_df.to_csv(filename, index=False)
        print(f"Front page articles saved to {filename}")
    except Exception as e:
        print(f"Error saving to CSV: {e}")

def search_articles_by_keyword(df, keyword):
    keyword = keyword.lower()
    matching_articles = []
    for index, row in df.iterrows():
        headline = row['headline']['main'].lower()
        if keyword in headline:
            matching_articles.append(row)
    return matching_articles

def analyze_sentiment(text):
  sentiment = analyzer.polarity_scores(text)
  return sentiment

for article in dob_articles:
  text = article.get("snippet", "")
  sentiment = analyze_sentiment(text)
  article['sentiment'] = sentiment

def get_named_entities(text):
  doc = nlp(text)
  named_entities = {
    "PERSON" : [],
    "ORG" : [],
    "GPE" : [],
    "DATE" : []
  }
  for ent in doc.ents:
    if ent.label_ in named_entities:
      named_entities[ent.label_].append(ent.text)
  return named_entities

if __name__ == "__main__":
    myapikey = "" # Input API key here
    year = "1990"
    month = "9"
    dob = "1990-09-21"
    
    articles = get_articles(myapikey, year, month)
    dob_articles = filter_articles_by_dob(articles, dob)

    # Create a DataFrame from dob_articles
    dobarticles_df = pd.DataFrame(dob_articles)

    # Save front page articles to CSV
    save_frontpage_to_csv(dobarticles_df[dobarticles_df['print_page'] == '1'], dob, "frontpage.csv")

    # Search for articles by keyword and print them
    matching_articles = search_articles_by_keyword(dobarticles_df, "stooge")
    for article in matching_articles:
        print(article['headline']['main'])

    # Data Analysis
    average_word_count = dobarticles_df['word_count'].mean()
    print(f"Average Word Count: {average_word_count}")

    # Data Visualization
    plot.hist(dobarticles_df['word_count'], bins=20, alpha=0.5, color='b')
    plot.xlabel('Word Count')
    plot.ylabel('Frequency')
    plot.title('Word Count Distribution')
    plot.show()

    # Additional Filtering
    high_word_count_articles = dobarticles_df[dobarticles_df['word_count'] > 1000]
    print(f"Number of Articles with Word Count > 1000: {len(high_word_count_articles)}")

    # Saving Additional CSV Files
    high_word_count_articles.to_csv("high_word_count_articles.csv", index=False)

    # Generating a Report
    with open("report.txt", "w") as report_file:
        report_file.write("Data Analysis Report\n")
        report_file.write("-------------------\n")
        report_file.write(f"Average Word Count: {average_word_count}\n")
        report_file.write(f"Number of Articles with Word Count > 1000: {len(high_word_count_articles)}\n")

    print("Report generated and saved as 'report.txt'.")