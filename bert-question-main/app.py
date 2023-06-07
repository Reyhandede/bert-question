from bs4 import BeautifulSoup
import urllib
import requests
import re
from setup import Selenium,Bert
from flask import Flask,render_template,request
import googletrans
from googletrans import Translator
import torch

def replace_non_alnum_punct(text):
    text = re.sub(r'\s', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def Search(textqr):
    driver = Selenium()
    model,tokenizer = Bert()
    translator = Translator()
    # Query to obtain links
    text = textqr
    result = translator.translate(text, dest='en')
    query=result.text
    links = [] # Initiate empty list to capture final results
    # Specify number of pages on google search, each page contains 10 #links
    n_pages = 2
    for page in range(1, n_pages):
        url = "http://www.google.com/search?q=" + query + "&start=" +      str((page - 1))
        driver.get(url)
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        search = soup.find_all('div', class_="yuRUbf")
        for h in search:
            links.append(h.a.get('href'))

    for url in links[0:5]:
        res=requests.get(url)
        #html_page=res.content
        soup=BeautifulSoup(res.text,features="html.parser")


        #a = " ".join(line.rstrip() for line in a.split(" "))
        #b+=a
        a=replace_non_alnum_punct(soup.get_text())

    paragraph=a[0:2048]

    encoding = tokenizer.encode_plus(text=query,text_pair=paragraph)
    inputs = encoding['input_ids']  #Token embeddings
    sentence_embedding = encoding['token_type_ids']  #Segment embeddings
    tokens = tokenizer.convert_ids_to_tokens(inputs) #input tokens

    start_scores, end_scores = model(input_ids=torch.tensor([inputs]), token_type_ids=torch.tensor([sentence_embedding]),return_dict=False)

    start_index = torch.argmax(start_scores)

    end_index = torch.argmax(end_scores)

    answer = ' '.join(tokens[start_index:end_index+1])



    corrected_answer = ''
    for word in answer.split():
        #If it's a subword token
        if word[0:2] == '##':
            corrected_answer += word[2:]
        else:
            corrected_answer += ' ' + word
    corrected_answer+="."
    result = translator.translate(corrected_answer, dest='tr')
    result.text= result.text.capitalize()
    print("----------------------------------------------------------------------------------------------------")
    print(result)
    return result.text





app = Flask(__name__)
@app.route("/")
def index():
    return render_template("index.html")
@app.route("/result",methods=["GET","POST"])
def result():
    if request.method == "POST":
        textqr=request.form.get("search")
        result = Search(textqr)
        return render_template("result.html",qr=result)
    else:
        return render_template("index.html",qr=result)
    


if __name__ == "__main__":
    app.run(debug = True)
    