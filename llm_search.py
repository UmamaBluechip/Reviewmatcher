from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse, parse_qs
from readability import Document
from datetime import datetime
from bs4 import BeautifulSoup
import requests
#import ujson
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

device = "cpu"

model_name = "C:/Users/Lenovo/.cache/huggingface/hub/models--mistralai--Mistral-7B-Instruct-v0.2/snapshots/41b61a33a2483885c981aa79e0df6b32407ed873"

model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

SOURCE_COUNT = 5


def generate_search_query(text: str) -> str:
  """
  Uses the provided model to generate a search query from a given text.

  This function assumes the model can perform instruction following tasks.
  """
  prompt = f"Given a query, respond with the Google search query that would best help to answer the query. Don't use search operators. Respond with only the Google query and nothing else.\nQuery: {text}"
  encoded_prompt = tokenizer(prompt, return_tensors="pt").to(device)
  with torch.no_grad():
      generated_ids = model.generate(encoded_prompt, max_length=64, do_sample=True)
  decoded_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
  return decoded_text.strip()


def get_google_search_links(query: str, source_count: int = SOURCE_COUNT, proxies: dict = None) -> list[str]:
    """
    Scrapes the official Google search page using the `requests` module and returns the first `source_count` links.
    """
    url = f"https://www.google.com/search?q={query}"
    if proxies:
        response = requests.get(url, proxies=proxies)
    else:
        response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    link_tags = soup.find_all("a")
    
    links = []
    for link in link_tags:
        href = link.get("href")
        if href and href.startswith("/url?q="):
            cleaned_href = parse_qs(href)["/url?q"][0]
            if cleaned_href not in links:
                links.append(cleaned_href)

    filtered_links = []
    for link in links:
        domain = urlparse(link).hostname
        exclude_list = ["google", "facebook", "twitter", "instagram", "youtube", "tiktok"]
        if not any(site in domain for site in exclude_list):
            if any(new_url.hostname == domain for new_url in [urlparse(l) for l in filtered_links]) == False:
                filtered_links.append(link)
    
    return filtered_links[:source_count]


def scrape_text_from_links(links: list, proxies: dict = None) -> list[dict]:   
    """
    Uses a `ThreadPoolExecutor` to run `scrape_text_from_links` on each link in `links` concurrently, allowing for lightning-fast scraping.
    """ 
    with ThreadPoolExecutor(max_workers=len(links)) as executor:
        if proxies:
            results = list(executor.map(scrape_text_from_link, links, [proxies] * len(links)))
        else:
            results = list(executor.map(scrape_text_from_link, links))
    
    for i, result in enumerate(results, start=1):
        result["result_number"] = i

    return results
    

def scrape_text_from_link(link: str, proxies: dict = None) -> dict:
    """
    Uses the `requests` module to scrape the text from a given link, and then uses the `readability-lxml` module along with `BeautifulSoup4` to parse the text into a readable format.
    """
    try:
        if proxies:
            response = requests.get(link, proxies=proxies, timeout=15)
        else:
            response = requests.get(link, timeout=15)
    except:
        return {"url": link, "text": "Error: Unable to connect to the website."}

    doc = Document(response.text)
    parsed = doc.summary()
    soup = BeautifulSoup(parsed, "html.parser")
    source_text = soup.get_text()
    return {"url": link, "text": summarize_text(source_text[:50000])}


def summarize_text(text: str) -> str:
  """
  Uses the provided model to summarize a given text.

  This function assumes the model can perform summarization tasks.
  """
  prompt = f"Given text, respond with the summarized text (no more than 100 words) and nothing else.\nText: {text}"
  encoded_prompt = tokenizer(prompt, return_tensors="pt").to(device)
  with torch.no_grad():
      generated_ids = model.generate(encoded_prompt, max_length=100, do_sample=True)
  decoded_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
  return decoded_text.strip()


def search(query: str, proxies: dict = None) -> tuple[list[str], list[dict]]:
    """
    This function takes a query as input, gets top Google search links for the query, and then scrapes the text from the links.
    It returns a tuple containing the list of links and a list of dictionaries. Each dictionary contains the URL and the summarized text from the link.
    """
    links = []

    for _ in range(3):
        if len(links) == 0:
            links = get_google_search_links(query, proxies=proxies)
        else:
            break

    sources = scrape_text_from_links(links, proxies=proxies)

    return links, sources


def perplexity_clone(query: str, proxies: dict = None, verbose=False) -> str:
  """
  A clone of Perplexity AI's "Search" feature using the Mistral model.

  This function takes a query as input and returns Markdown formatted text containing
  a response to the query with cited sources. It leverages retrieved summaries
  for answer generation with the Mistral model.
  """
  formatted_time = datetime.utcnow().strftime("%A, %B %d, %Y %H:%M:%S UTC")

  if verbose:
      print(f"Searching \"{query}\"...")
  links, sources = search(query, proxies=proxies)

  instructions = f"Given a list of web search results (URL and Summary), generate a comprehensive and informative answer for a given question solely based on the provided information. Use an unbiased and journalistic tone. Use this current date and time: {formatted_time}. Combine search results together into a coherent answer. Do not repeat text. Cite search results using [number] notation, and don't link the citations. Only cite the most relevant results that answer the question accurately.\n"

  for i, source in enumerate(sources, start=1):
      instructions += f"Summary {i}: {source['text']}\nURL {i}: {source['url']}\n"
  instructions += f"Question: {query}"

  encoded_prompt = tokenizer(instructions, return_tensors="pt").to(device)
  with torch.no_grad():
      generated_ids = model.generate(encoded_prompt, max_length=1000, do_sample=True)
  decoded_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

  for i, link in enumerate(links, start=1):
      decoded_text = decoded_text.replace(f"[{i}]", f"[[{i}]]({link})")

  return decoded_text
