import re
from flask import Flask, render_template, request
from llm_search import perplexity_clone
import os
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
sentiment_model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

proxies={
          "http": os.getenv("PROXY"),
          "https": os.getenv("PROXY")
          }


def extract_products(search_result):
    products = {}
    current_product = None
    current_citation = None
    citations = re.findall(r"\[\[\d+\]\]\(.*?\)", search_result)

    for match in re.finditer(r"\[\[\d+\]\]\((.*?)\)(.*?)(?=(\[\[\d+\]\]\((.*?)\))|$)", search_result, re.DOTALL):
        citation, link, content = match.groups()
        content = content.strip() 
        if content:
            if current_product is None:
                current_product = content.split(".")[0]
                products[current_product] = {"citations": [citation], "content": [content], "link": link}
            else:
                if citation is not None:
                    current_citation = citation
                
                next_sentence = content.split(".")[0]
                if next_sentence != current_product and next_sentence in products:
                    current_product = next_sentence
                
                products[current_product]["citations"].append(citation)
                products[current_product]["content"].append(content)
                products[current_product]["link"] = link

    return products


def analyze_and_compare_info(product_info):
    pros = []
    cons = []
    features = []
    overall_sentiment = []

    for content in product_info["content"]:
        inputs = tokenizer(content, return_tensors="pt")
        with torch.no_grad():
            logits = sentiment_model(**inputs).logits
        predicted_class_id = logits.argmax().item()
        sentiment = sentiment_model.config.id2label[predicted_class_id]
        overall_sentiment.append(sentiment)
        if sentiment == "POSITIVE":
            pros.append(content)
        elif sentiment == "NEGATIVE":
            cons.append(content)
        else:
            features.append(content)
    
    positive_count = overall_sentiment.count("POSITIVE")
    negative_count = overall_sentiment.count("NEGATIVE")
    if positive_count > negative_count:
        overall_sentiment = "positive"
    elif positive_count < negative_count:
        overall_sentiment = "negative"
    else:
        overall_sentiment = "neutral"

    return {
        "pros": pros,
        "cons": cons,
        "features": features,
        "link": product_info["link"],
        "overall_sentiment": overall_sentiment
    }


def search_and_compare(query, proxies=proxies):

    search_result = perplexity_clone(query, proxies=proxies, verbose=False) 

    products = extract_products(search_result)
    comparisons = {}

    for product_name, product_info in products.items():
        comparisons[product_name] = analyze_and_compare_info(product_info)

    return comparisons  


app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        search_query = request.form["search"]
        comparison_results = search_and_compare(search_query)
        return render_template("results.html", results=comparison_results)
    else:
        return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True) 
