
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer

def Selenium():
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument("--headless")
    driver = webdriver.Chrome(ChromeDriverManager().install(), chrome_options=chrome_options)
    return driver
def Bert():
    #Model
    model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    model.config.max_position_embeddings = 4096

    #Tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

    return model,tokenizer



