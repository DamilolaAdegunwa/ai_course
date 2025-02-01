# Install the newspaper3k library if you haven't already
# pip install newspaper3k

from newspaper import Article, build


def gather_articles(urls):
    """
    Gather news articles from the provided list of URLs.

    Args:
    - urls (list): List of URLs of news websites.

    Returns:
    - articles_data (list): List of dictionaries with article details.
    """
    articles_data = []

    for url in urls:
        try:
            # Build a newspaper object for the website
            news_source = build(url, memoize_articles=False)

            # Iterate through articles from the source
            for article in news_source.articles[:10]:  # Limit to the first 10 articles
                try:
                    article.download()
                    article.parse()
                    article.nlp()  # Perform natural language processing
                    articles_data.append({
                        "title": article.title,
                        "authors": article.authors,
                        "publish_date": article.publish_date,
                        "text": article.text,
                        "summary": article.summary,
                        "keywords": article.keywords,
                        "url": article.url
                    })
                except Exception as e:
                    print(f"Error processing article from {article.url}: {e}")

        except Exception as e:
            print(f"Error processing source {url}: {e}")

    return articles_data


# Example usage
if __name__ == "__main__":
    news_sites = [
        "https://www.bbc.com",  # Example news site
        "https://www.cnn.com",
        "https://www.reuters.com"
    ]

    collected_articles = gather_articles(news_sites)

    # Print gathered articles
    for article in collected_articles:
        print(f"Title: {article['title']}")
        print(f"Authors: {', '.join(article['authors']) if article['authors'] else 'Unknown'}")
        print(f"Date: {article['publish_date']}")
        print(f"Summary: {article['summary']}")
        print(f"Keywords: {', '.join(article['keywords'])}")
        print(f"URL: {article['url']}\n")
