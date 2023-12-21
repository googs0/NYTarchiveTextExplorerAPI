import matplotlib.pyplot as plt
import pandas as pd
import requests
import spacy
from textblob import TextBlob
from wordcloud import WordCloud

# NLP pipeline
nlp = spacy.load("en_core_web_md")


def get_articles(api_key, year, month):
    base_url = "https://api.nytimes.com/svc/archive/v1"
    month_url = f"{month}.json"
    url = f"{base_url}/{year}/{month_url}"
    query = {"api-key": api_key}

    try:
        response = requests.get(url, params=query)
        response.raise_for_status()
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


def save_frontpage_to_csv(frontpage_df, filename):
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
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    return sentiment


def get_named_entities(text):
    doc = nlp(text)
    named_entities = {
        "PERSON": [],
        "ORG": [],
        "GPE": [],
        "DATE": []
    }
  
    for ent in doc.ents:
        if ent.label_ in named_entities:
            named_entities[ent.label_].append(ent.text)
    return named_entities


def perform_sentiment_analysis_and_ner(articles):
    for i, article in enumerate(articles, 1):
        text = article.get("snippet", "") or article.get("lead_paragraph", "")
        sentiment = analyze_sentiment(text)
        entities = get_named_entities(text)
        article['sentiment'] = sentiment
        article['snippet'] = text
        article['entities'] = entities


def visualize_sentiment_vs_wordcount(df):
    plt.figure(figsize=(12, 8))
    scat_colormap = plt.colormaps['turbo_r']
    plt.scatter(df['word_count'], df['sentiment'], c=df['sentiment'], cmap=scat_colormap, alpha=0.7)

    plt.colorbar(label='Sentiment')
    plt.xlabel('Word Count', fontsize=14, labelpad=20, fontweight='bold')
    plt.ylabel('Sentiment', fontsize=14, labelpad=20, fontweight='bold')
    plt.title('Sentiment vs. Word Count', fontsize=20, pad=20, fontweight='bold')
    plt.show()


def wordcloud_top_words(df):
    text = ' '.join(df['headline'].apply(lambda x: x['main'] if 'main' in x else ''))
    date = df['pub_date'].iloc[0][:10]
    wordcloud = WordCloud(width=1000, height=800, background_color='white').generate(text)

    plt.figure(figsize=(12, 10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Top Keywords & Topics - {date}', fontsize=20, pad=20, fontweight='bold')
    plt.show()


def main():
    apikey = ""
    year = "1991"
    month = "10"
    dob = "1991-10-21"

    articles = get_articles(apikey, year, month)

    if not articles:
        print("No articles found.")
        return

    dob_articles = filter_articles_by_dob(articles, dob)

    # Analyze sentiment, snippet, and entities for each article
    perform_sentiment_analysis_and_ner(dob_articles)

    # DOB articles dataframe
    dobarticles_df = pd.DataFrame(dob_articles)

    # Keyword Search
    keyword = "stooge"
    matching_articles = search_articles_by_keyword(dobarticles_df, keyword)

    # Average word count
    average_word_count = dobarticles_df['word_count'].mean()

    # Word count thresholds and filtering
    high_threshold = 1000
    low_threshold = 500
    high_word_count_articles = dobarticles_df[dobarticles_df['word_count'] > high_threshold]
    low_word_count_articles = dobarticles_df[dobarticles_df['word_count'] < low_threshold]

    # Save front page articles to CSV
    save_frontpage_to_csv(dobarticles_df[dobarticles_df['print_page'] == '1'], "frontpage.csv")

    # Saving Additional CSV Files
    high_word_count_articles.to_csv("high_word_count_articles.csv", index=False)

    # Visualizations
    visualize_sentiment_vs_wordcount(dobarticles_df)
    wordcloud_top_words(dobarticles_df)

    # Generate report
    with open("report.txt", "w") as report_file:
        report_file.write("Data Analysis Report\n")
        report_file.write("-------------------\n")
        report_file.write(f"Average Word Count: {average_word_count}\n")
        report_file.write(f"Number of Articles with Word Count > 1000: {len(high_word_count_articles)}\n")

        # Sentiment Analysis
        report_file.write("\nSentiment Analysis\n")
        for i, article in enumerate(dob_articles, 1):
            report_file.write(f"\nArticle {i}:\n")
            report_file.write(f"Sentiment: {article.get('sentiment', 'N/A')}\n")
            report_file.write(f"Snippet: {article.get('snippet', 'N/A')}\n")

        # Entity Analysis
        report_file.write("\nNamed Entities Analysis\n")
        for i, article in enumerate(dob_articles, 1):
            report_file.write(f"\nArticle {i}:\n")
            entities = article.get('entities', {})
            for entity_type, entity_list in entities.items():
                report_file.write(f"{entity_type}: {entity_list}\n")

        # Additional Filtering
        report_file.write("\nAdditional Metrics\n")
        report_file.write(f"Number of Articles with Word Count > {high_threshold}: {len(high_word_count_articles)}\n\n")
        report_file.write(f"Average Word Count: {average_word_count}\n")
        report_file.write(f"Highest Word Count Article: {high_word_count_articles['word_count'].max()}\n")
        report_file.write(f"Lowest Word Count Article: {low_word_count_articles['word_count'].min()}\n\n")
        report_file.write(f"Article with Highest Sentiment: {dobarticles_df['sentiment'].max()}\n")
        report_file.write(f"Article with Lowest Sentiment: {dobarticles_df['sentiment'].min()}\n\n")
        report_file.write(f"Returned article with keyword '{keyword}': {matching_articles[0]['snippet']}\n")

    print("Report generated and saved as 'report.txt.'")


if __name__ == "__main__":
    main()
