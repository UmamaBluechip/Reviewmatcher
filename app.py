from flask import Flask, render_template, request

def search_and_compare(query, proxies=None):

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
