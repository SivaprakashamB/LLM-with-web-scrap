# from flask import Flask, render_template, request, jsonify
# import requests
# from bs4 import BeautifulSoup
# import os
#
#
# app = Flask(__name__)
#
# # Ensure your API key is stored as a string
# # HUGGING_FACE_API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
# HUGGING_FACE_API_URL = "https://api-inference.huggingface.co/models/serpapi/bert-base-local-results"
#
# HUGGING_FACE_API_KEY = "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"
#
# headers = {
#     "Authorization": f"Bearer {HUGGING_FACE_API_KEY}"
# }
#
# @app.route('/')
# def index():
#     return render_template('index.html')
#
#
# @app.route('/scrape', methods=['POST'])
# def scrape_and_summarize():
#     url = request.json.get('url')
#     if not url:
#         return jsonify({'error': 'No URL provided'}), 400
#
#     # Web scraping
#     response = requests.get(url)
#     print("response", response)
#     if response.status_code != 200:
#         return jsonify({'error': 'Failed to retrieve content'}), 500
#
#     # soup = BeautifulSoup(response.content, 'html.parser')
#     # paragraphs = soup.find_all('p')
#     # text = ' '.join([para.get_text() for para in paragraphs])
#     # html_content = response.text
#     # print("html_contenthtml_content", html_content)
#     # Prepare payload for Hugging Face API
#     payload = {
#         # "inputs": text,
#         "inputs": url,
#         # "parameters": {"max_length": 130, "min_length": 30, "do_sample": False}
#     }
#     print("payloadpayload", payload)
#     # Use Hugging Face API for summarization
#     api_response = requests.post(HUGGING_FACE_API_URL, headers=headers, json=payload)
#     print("api_responseapi_response", api_response)
#     if api_response.status_code != 200:
#         return jsonify({'error': 'Failed to get summary from Hugging Face API'}), 500
#     print("api_response.json()", api_response.json())
#     summary = api_response.json()
#     print("summarysummary", summary)
#     return jsonify({'summary': summary})
#
# if __name__ == '__main__':
#     app.run(debug=True)
# #
# #
# #
# # -------------------------------------------------------------------------------------------------
# # import base64
# #
# # from flask import Flask, request, jsonify, render_template
# # import json
# # import requests
# # import time
# # import cv2
# #
# # app = Flask(__name__)
# #
# # token_access = "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"
# # headers = {"Authorization": f"Bearer {token_access}"}
# #
# # API_URL = "https://api-inference.huggingface.co/models/facebook/detr-resnet-50"
# #
# # def query(filename):
# #     with open(filename, "rb") as f:
# #         data = f.read()
# #
# #     while True:
# #         try:
# #             time.sleep(1)
# #             response = requests.post(API_URL, headers=headers, data=data)
# #             break
# #         except Exception:
# #             continue
# #
# #     return json.loads(response.content.decode("utf-8"))
# #
# # @app.route('/')
# # def index():
# #     return render_template('index.html')
# #
# #
# # @app.route('/detect_objects', methods=['POST'])
# # def detect_objects():
# #     file = request.files['file']
# #     if file:
# #         file_path = 'uploads/' + file.filename
# #         file.save(file_path)
# #         data = query(file_path)
# #
# #         # Open the image file
# #         with open(file_path, "rb") as f:
# #             image_data = f.read()
# #
# #         # Encode the image data to base64
# #         encoded_image = base64.b64encode(image_data).decode('utf-8')
# #
# #         return jsonify({'image': encoded_image, 'detections': data})
# #     else:
# #         return jsonify({'error': 'No file provided'}), 400
# #
# #
# # if __name__ == '__main__':
# #     app.run(debug=True)
# from flask import Flask, request, render_template
# import requests
#
# app = Flask(__name__)
#
# LM_STUDIO_API_URL = "http://localhost:1234/v1/chat/completions"
#
#
# @app.route('/')
# def home():
#     return render_template('index.html')
#
#
# @app.route('/ask', methods=['POST'])
# def ask():
#     user_input = request.form['text']
#     payload = {
#         "messages": [
#             {"role": "system", "content": "You are a helpful coding assistant."},
#             {"role": "user", "content": user_input}
#         ],
#         "temperature": 0.7,
#         "max_tokens": -1,
#         "stream": False
#     }
#
#     headers = {"Content-Type": "application/json"}
#
#     response = requests.post(LM_STUDIO_API_URL, json=payload, headers=headers)
#
#     if response.status_code == 200:
#         data = response.json()
#         # Extract the response content
#         answer = data['choices'][0]['message']['content']
#     else:
#         answer = "Error: Could not get a response from LM Studio."
#
#     return render_template('index.html', question=user_input, answer=answer)
#
#
# if __name__ == "__main__":
#     app.run(debug=True)
# from flask import Flask, request, render_template, session
# import requests
# from bs4 import BeautifulSoup, Comment
# from sentence_transformers import SentenceTransformer, util
# import faiss
# import numpy as np
# import time
# import pickle
# import os
# from urllib.parse import urljoin
# from selenium import webdriver
# from webdriver_manager.chrome import ChromeDriverManager
# from selenium.webdriver.chrome.service import Service
# from selenium.webdriver.common.by import By
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC
#
# from view_pickle_files import index, chunks
#
# app = Flask(__name__)
# app.secret_key = 'supersecretkey'
#
#
#
# def setup_driver():
#     # Setup the Chrome WebDriver using webdriver-manager
#     service = Service(ChromeDriverManager().install())
#     options = webdriver.ChromeOptions()
#     options.add_argument('--headless')
#     options.add_argument('--disable-gpu')
#     options.add_argument('--no-sandbox')
#     driver = webdriver.Chrome(service=service, options=options)
#     return driver
#
#
# base_dir = os.path.abspath(os.path.dirname(__file__))
#
# faiss_index_path = os.path.join(base_dir, 'faiss_index.pkl')
#
# chunks_index_path = os.path.join(base_dir, 'chunks.pkl')
#
# # Initialize the sentence transformer model
# embedder = SentenceTransformer('all-MiniLM-L6-v2')
# LM_STUDIO_API_URL = "http://localhost:1234/v1/chat/completions"
# # LM_STUDIO_API_URL = "http://localhost:1234/v1/embeddings"
#
#
# def crawl_website(url, max_retries=3):
#     pages = set()
#     print("pages", pages)
#     to_visit = [url]
#     print("to_visit", to_visit)
#     visited = set()
#     print("visited", visited)
#
#     while to_visit:
#         current_url = to_visit.pop()
#         print("current_url", current_url)
#         print("visited", visited)
#         if current_url in visited:
#             continue
#
#         retries = 0
#         while retries < max_retries:
#             try:
#                 headers = {
#                     'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'
#                 }
#                 response = requests.get(current_url, headers=headers)
#                 response.raise_for_status()
#                 break  # Exit the retry loop if the request was successful
#             except requests.exceptions.RequestException as e:
#                 print(f"Error crawling {current_url}: {e}")
#                 retries += 1
#                 time.sleep(3)
#                 if retries == max_retries:
#                     print(f"Failed to crawl {current_url} after {max_retries} retries.")
#                     continue  # Move to the next URL in to_visit
#
#         visited.add(current_url)
#         pages.add(current_url)
#         print("visited", visited)
#         print("pagespages", pages)
#         soup = BeautifulSoup(response.text, 'html.parser')
#         for a_tag in soup.find_all('a', href=True):
#             link = a_tag['href']
#             print("linklink", link)
#             full_link = urljoin(url, link)  # Convert relative URLs to absolute
#             print("full_linkfull_link", full_link)
#             if url in full_link and full_link not in visited and full_link not in to_visit:
#                 to_visit.append(full_link)
#                 print("to_visitto_visit", to_visit)
#
#     print("pages:", pages)
#     return pages
#
# def extract_text_from_pages(pages):
#     driver = setup_driver()
#     content = ""
#     print("Extracting text from pages...")
#     for page in pages:
#         print("Processing page:", page)
#         retries = 0
#         success = False
#         print("retries", retries)
#         while retries < 3:
#             try:
#                 print("pagepage", page)
#                 driver.get(page)
#                 print("driverdriver", driver)
#                 # Wait until the content is loaded (adjust the condition as needed)
#                 time.sleep(10)
#                 success = True
#                 break  # Exit the retry loop if the request was successful
#             except Exception as e:
#                 print(f"Error extracting text from {page}: {e}")
#                 retries += 1
#                 time.sleep(12)
#
#         if not success:
#             print(f"Failed to extract text from {page} after 3 retries.")
#             continue
#
#         soup = BeautifulSoup(driver.page_source, 'html.parser')
#         texts = soup.find_all(string=True)
#         visible_texts = filter(tag_visible, texts)
#         print("visible_texts", visible_texts)
#         visible_content = " ".join(t.strip() for t in visible_texts)
#         print("visible_content", visible_content)
#         if visible_content:
#             content += visible_content + " "
#
#     driver.quit()
#     print("Extracted content:", content)
#     return content.strip()
#
# def tag_visible(element):
#     if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
#         return False
#     if isinstance(element, Comment):
#         return False
#     return True
#
#
# def prepare_vector_store(content, local_llm_api):
#     # Split the content into chunks
#     # chunks = [content[i:i + 512] for i in range(0, len(content), 512)]
#     # print("chunkschunks", chunks)
#     # # Create embeddings for the chunks
#     # embeddings = embedder.encode(chunks, convert_to_tensor=False)
#     # print("embeddings", embeddings)
#     # # Create a FAISS index
#     # dimension = embeddings.shape[1]
#     # print("dimension", dimension)
#     # index = faiss.IndexFlatL2(dimension)
#     # print("index", index)
#     # index.add(np.array(embeddings))
#     # # Save the index and chunks
#     # with open(faiss_index_path, 'wb') as f:
#     #     a = pickle.dump(index, f)
#     #     print("AAAAAAAa", a)
#     # with open(chunks_index_path, 'wb') as f:
#     #     b = pickle.dump(chunks, f)
#     #     print("bbbbbbbb", b)
#     payload = {
#         "input": content,
#         "model": "Meta-Llama-3-8B-Instruct-GGUF"
#     }
#     if local_llm_api:
#         local_llm_api = 'http://localhost:1234/v1/embeddings'
#     else:
#         local_llm_api = 'http://localhost:1234/v1/chat/completions'
#
#     response = requests.post(local_llm_api, json=payload)
#     if response.status_code == 200:
#         embeddings = response.json()['data']
#         return [embedding['embedding'] for embedding in embeddings]
#     else:
#         raise Exception("Error generating embeddings")
#
# @app.route('/')
# def home():
#     return render_template('index.html')
#
#
# @app.route('/prepare', methods=['POST'])
# def prepare():
#     url = request.form.get('url')
#     print("urlurl", url)
#     if not url:
#         return render_template('index.html', error="Please provide a URL.")
#
#     try:
#         # Crawl the website and extract text
#         pages = crawl_website(url)
#         print("pages", pages)
#         content = extract_text_from_pages(pages)
#
#         if not content:
#             return render_template('index.html', error="Could not extract content from the provided URL.")
#         local_llm_api = True
#         # Prepare the vector store
#         session['url'] = url  # Store the URL in the session
#         prepare_vector_store(content, local_llm_api)
#
#     except requests.exceptions.RequestException as e:
#         return render_template('index.html', error=f"Error processing the request: {e}")
#
#     return render_template('index.html', message="Content prepared successfully.", url=url)
#
#
# def query_vector_store(question, top_k=1):
#     # Load the index and chunks
#     # with open(faiss_index_path, 'rb') as f:
#     #     print("faiss_index_path F", f)
#     #     index = pickle.load(f)
#     #     print("index", index)
#     # with open(chunks_index_path, 'rb') as f:
#     #     print("chunks_index_path F", f)
#     #     chunks = pickle.load(f)
#     #     print("chunks", chunks)
#     #
#     # # Encode the question
#     # question_embedding = embedder.encode(question, convert_to_tensor=False)
#     # print("question_embedding embedder.encode", question_embedding)
#     # question_embedding = np.expand_dims(question_embedding, axis=0)
#     # print("question_embedding np.expand_dims", question_embedding)
#     #
#     # # Search the FAISS index
#     # distances, indices = index.search(question_embedding, top_k)
#     # print("distances", distances)
#     # print("indices", indices)
#     # results = [chunks[idx] for idx in indices[0]]
#     # print("results", results)
#     question_embedding = prepare_vector_store([question])[0]
#     distances, indices = index.search(np.array([question_embedding]), top_k)
#     results = [chunks[idx] for idx in indices[0]]
#     return results
#
#
# @app.route('/ask', methods=['POST'])
# def ask():
#     user_input = request.form.get('text')
#
#     if not user_input:
#         return render_template('index.html', error="Please ask a question.")
#
#     try:
#         # Query the vector store for relevant content
#         print("user_input", user_input)
#         results = query_vector_store(user_input, top_k=1)
#         print("results", results)
#         # best_chunk = results[0] if results else "No relevant information found."
#         best_chunk = " ".join(results) if results else "No relevant information found."
#         print("best_chunk", best_chunk)
#         # Use the best chunk to generate the answer using LM Studio
#         payload = {
#             "messages": [
#                 {"role": "system",
#                  "content": "You are a helpful assistant. Answer the question based on the provided context."},
#                 {"role": "user", "content": f"Context: {best_chunk} Question: {user_input}"}
#             ],
#             "temperature": 0.7,
#             "max_tokens": -1,
#             "stream": False
#         }
#
#         headers = {"Content-Type": "application/json"}
#         response = requests.post(LM_STUDIO_API_URL, json=payload, headers=headers)
#         print("response", response)
#         if response.status_code == 200:
#             data = response.json()
#             print("data", data)
#             answer = data['choices'][0]['message']['content']
#             print("answer", answer)
#         else:
#             answer = "Error: Could not get a response from LM Studio."
#
#     except requests.exceptions.RequestException as e:
#         return render_template('index.html', error=f"Error processing the request: {e}")
#
#     return render_template('index.html', question=user_input, answer=answer)
#
#
# if __name__ == "__main__":
#     app.run(debug=True)

# from flask import Flask, request, render_template, session
# import requests
# from bs4 import BeautifulSoup, Comment
# from sentence_transformers import SentenceTransformer, util
# import faiss
# import numpy as np
# import time
# import pickle
# import os
# from urllib.parse import urljoin
# from selenium import webdriver
# from webdriver_manager.chrome import ChromeDriverManager
# from selenium.webdriver.chrome.service import Service
# import tiktoken
#
# app = Flask(__name__)
# app.secret_key = 'supersecretkey'
#
# # Setup paths
# base_dir = os.path.abspath(os.path.dirname(__file__))
# faiss_index_path = os.path.join(base_dir, 'faiss_index.pkl')
# chunks_index_path = os.path.join(base_dir, 'chunks.pkl')
#
# # Initialize the sentence transformer model
# embedder = SentenceTransformer('all-MiniLM-L6-v2')
# EMBEDDINGS_API_URL = "http://localhost:1234/v1/embeddings"
# COMPLETIONS_API_URL = "http://localhost:1234/v1/chat/completions"
# OPENAI_API_KEY = 'lm-studio'
# MODEL_IDENTIFIER = "model-identifier" # model-identifier # text-embedding-3-small
#
# faiss_index = None
# chunks = []
# nlist = 100  # Number of clusters
# nprobe = 10  # Number of clusters to search
#
# def setup_driver():
#     service = Service(ChromeDriverManager().install())
#     options = webdriver.ChromeOptions()
#     options.add_argument('--headless')
#     options.add_argument('--disable-gpu')
#     options.add_argument('--no-sandbox')
#     driver = webdriver.Chrome(service=service, options=options)
#     return driver
#
# def crawl_website(url, max_retries=3):
#     print("url", url)
#     pages = set()
#     print("pages", pages)
#     to_visit = [url]
#     print("to_visit", to_visit)
#     visited = set()
#     print("visited", visited)
#     print("to_visit", to_visit)
#     while to_visit:
#         current_url = to_visit.pop()
#         print("current_url", current_url)
#         print("visited", visited)
#         if current_url in visited:
#             continue
#
#         retries = 0
#         while retries < max_retries:
#             try:
#                 headers = {
#                     'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'
#                 }
#                 response = requests.get(current_url, headers=headers)
#                 print("response while", response)
#                 response.raise_for_status()
#                 break
#             except requests.exceptions.RequestException as e:
#                 retries += 1
#                 time.sleep(130)
#                 if retries == max_retries:
#                     continue
#
#         visited.add(current_url)
#         print("visited", visited)
#         pages.add(current_url)
#         print("pages", pages)
#         soup = BeautifulSoup(response.text, 'html.parser')
#         print("soup", soup)
#         for a_tag in soup.find_all('a', href=True):
#             link = a_tag['href']
#             print("link", link)
#             full_link = urljoin(url, link)
#             print("full_link", full_link)
#             if url in full_link and full_link not in visited and full_link not in to_visit:
#                 to_visit.append(full_link)
#                 print("to_visit IF", to_visit)
#     print("pagespages", pages)
#     return pages
#
# def extract_text_from_pages(pages):
#     driver = setup_driver()
#     print("driver", driver)
#     content = ""
#     print("pages", pages)
#     for page in pages:
#         print("page", page)
#         retries = 0
#         success = False
#         while retries < 3:
#             try:
#                 driver.get(page)
#                 time.sleep(10)
#                 success = True
#                 break
#             except Exception as e:
#                 retries += 1
#                 time.sleep(12)
#
#         if not success:
#             continue
#
#         soup = BeautifulSoup(driver.page_source, 'html.parser')
#         print("soup", soup)
#         texts = soup.find_all(string=True)
#         print("texts", texts)
#         visible_texts = filter(tag_visible, texts)
#         print("visible_texts", visible_texts)
#         visible_content = " ".join(t.strip() for t in visible_texts)
#         print("visible_content", visible_content)
#         if visible_content:
#             content += visible_content + " "
#
#     driver.quit()
#     print("content.strip()", content.strip())
#     return content.strip()
#
# def tag_visible(element):
#     if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
#         return False
#     if isinstance(element, Comment):
#         return False
#     return True
#
# def get_embedding(text, model="model-identifier"):
#     payload = {
#         "input": [text],
#         "model": model
#     }
#     headers = {
#         "Content-Type": "application/json",
#         "Authorization": f"Bearer {OPENAI_API_KEY}"
#     }
#     print("payload", payload)
#     response = requests.post(EMBEDDINGS_API_URL, headers=headers, json=payload)
#     print("response", response)
#     if response.status_code == 200:
#         embedding = response.json()['data'][0]['embedding']
#         print(f"Embedding dimension: {len(embedding)}")  # Debug print for embedding dimension
#         return embedding
#         # return response.json()['data'][0]['embedding']
#     else:
#         raise Exception(f"Error: {response.status_code} - {response.text}")
#
#
# def prepare_vector_store(content):
#     global faiss_index, chunks
#     print("faiss_index", faiss_index)
#     print("chunks", chunks)
#     print("content", content)
#     chunks = [content[i:i + 512] for i in range(0, len(content), 512)]
#     print("chunks", chunks)
#
#     embeddings = []
#     for chunk in chunks:
#         embedding = get_embedding(chunk, model="model-identifier")
#         embeddings.append(embedding)
#
#     dimension = len(embeddings[0])
#     # Create and train the FAISS index
#     nlist = 100
#     nlist = min(nlist, len(embeddings))
#     print(f"Adjusted nlist: {nlist}")  # Debug print for nlist
#     quantizer = faiss.IndexFlatL2(dimension)
#     faiss_index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
#     assert not faiss_index.is_trained
#     faiss_index.train(np.array(embeddings).astype(np.float32))
#     assert faiss_index.is_trained
#
#     faiss_index.add(np.array(embeddings).astype(np.float32))
#     faiss_index.nprobe = nprobe
#
#     # faiss_index = faiss.IndexFlatL2(dimension)
#     # faiss_index.add(np.array(embeddings))
#     # return ({"status": "success", "message": "Embeddings added to FAISS index"})
#
#     # payload = {
#     #     "input": chunks,
#     #     "model": "model-identifier" # nomic-embed-text-v1.5 # model-identifier-here # text-embedding-3-small
#     # }  # text-embedding-3-large # bge-large-en-v1.5-gguf
#     # print("payload", payload)
#     # headers = {
#     #     "Content-Type": "application/json",
#     #     "Authorization": f"Bearer {OPENAI_API_KEY}"
#     # }
#     # response = requests.post(EMBEDDINGS_API_URL, headers=headers, json=payload)
#     # print("response", response)
#     # if response.status_code == 200:
#     #     data = response.json()
#     #     print("datadatadata", data)
#     #     if 'data' in data:
#     #         embeddings = [item['embedding'] for item in data['data']]
#     #         print("embeddings", embeddings)
#     #         dimension = len(embeddings[0])
#     #         print("dimension", dimension)
#     #         index = faiss.IndexFlatL2(dimension)
#     #         print("index", index)
#     #         a = index.add(np.array(embeddings))
#     #         print("AAAAA", a)
#     #         faiss_index = faiss.IndexFlatL2(dimension)
#     #         faiss_index.add(np.array(embeddings))
#     #     # else:
#     #     #     raise ValueError("The key 'data' is missing in the response.")
#     #
#     #         # with open(faiss_index_path, 'wb') as f:
#     #         #     pickle.dump(index, f)
#     #         # with open(chunks_index_path, 'wb') as f:
#     #         #     pickle.dump(chunks, f)
#     #     else:
#     #         raise ValueError("The key 'data' is missing in the response.")
#     # else:
#     #     raise Exception(f"Error generating embeddings: {response.status_code} - {response.text}")
#
#
# @app.route('/')
# def home():
#     return render_template('index.html')
#
#
# @app.route('/prepare', methods=['POST'])
# def prepare():
#     url = request.form.get('url')
#     print("url", url)
#     if not url:
#         return render_template('index.html', error="Please provide a URL.")
#
#     try:
#         pages = crawl_website(url)
#         print("pages", pages)
#         content = extract_text_from_pages(pages)
#         print("contentcontent", content)
#
#         if not content:
#             return render_template('index.html', error="Could not extract content from the provided URL.")
#
#         session['url'] = url
#         prepare_vector_store(content)
#         print(">>>>>>")
#     except requests.exceptions.RequestException as e:
#         return render_template('index.html', error=f"Error processing the request: {e}")
#
#     return render_template('index.html', message="Content prepared successfully.", url=url)
#
#
# # def query_vector_store(question, top_k=1):
# #     global faiss_index, chunks
# #
# #     question_embedding = embedder.encode(question, convert_to_tensor=False)
# #     print("question_embedding", question_embedding)
# #     question_embedding = np.expand_dims(question_embedding, axis=0)
# #     print("question_embedding", question_embedding)
# #
# #     distances, indices = faiss_index.search(question_embedding, top_k)
# #
# #     results = [chunks[idx] for idx in indices[0]]
# #     print("resultsresults", results)
# #     return results
#
# def query_vector_store(question, top_k=1):
#     try:
#         global faiss_index, chunks
#         # question = request.json.get('question')
#         # top_k = request.json.get('top_k', 1)
#
#         question_embedding = get_embedding(question, model=MODEL_IDENTIFIER)
#         print(f"Query embedding dimension: {len(question_embedding)}")  # Debug print for query embedding dimension
#
#         # question_embedding = np.expand_dims(np.array(question_embedding), axis=0)
#         question_embedding = np.expand_dims(np.array(question_embedding).astype(np.float32), axis=0)
#         print("question_embedding", question_embedding)
#
#         distances, indices = faiss_index.search(question_embedding, top_k)
#         print("indicesindices", indices)
#         results = [chunks[idx] for idx in indices[0]]
#         print("results", results)
#         return results
#     except Exception as e:
#         print(f"Error: {e}")
#         return e
# # def query_vector_store(question, top_k=1):
# #     with open(faiss_index_path, 'rb') as f:
# #         index = pickle.load(f)
# #     with open(chunks_index_path, 'rb') as f:
# #         chunks = pickle.load(f)
# #
# #     question_embedding = embedder.encode(question, convert_to_tensor=False)
# #     print("question_embedding", question_embedding)
# #     question_embedding = np.expand_dims(question_embedding, axis=0)
# #     print("question_embedding", question_embedding)
# #
# #     distances, indices = index.search(question_embedding, top_k)
# #     results = [chunks[idx] for idx in indices[0]]
# #     print("resultsresults", results)
# #     return results
#
# @app.route('/ask', methods=['POST'])
# def ask():
#     user_input = request.form.get('text')
#     if not user_input:
#         return render_template('index.html', error="Please ask a question.")
#
#     try:
#         results = query_vector_store(user_input, top_k=1)
#         print("resultsresults", results)
#         # best_chunk = " ".join(results) if results else "No relevant information found."
#         best_chunk = results
#         payload = {
#             "messages": [
#                 {"role": "system", "content": "You are a helpful assistant. Answer the question based on the provided context."},
#                 {"role": "user", "content": f"Context: {best_chunk} Question: {user_input}"}
#             ],
#             "temperature": 0.7,
#             "max_tokens": -1,
#             "stream": False
#         }
#
#         headers = {"Content-Type": "application/json"}
#         response = requests.post(COMPLETIONS_API_URL, json=payload, headers=headers)
#         if response.status_code == 200:
#             data = response.json()
#             answer = data['choices'][0]['message']['content']
#         else:
#             answer = "Error: Could not get a response from LM Studio."
#     except requests.exceptions.RequestException as e:
#         return render_template('index.html', error=f"Error processing the request: {e}")
#
#     return render_template('index.html', question=user_input, answer=answer)
#
# if __name__ == "__main__":
#     app.run(debug=True)


# from flask import Flask, request, render_template, session
# import requests
# from bs4 import BeautifulSoup, Comment
# from sentence_transformers import SentenceTransformer, util
# import faiss
# import numpy as np
# import time
# import pickle
# import os
# from urllib.parse import urljoin
# from selenium import webdriver
# from webdriver_manager.chrome import ChromeDriverManager
# from selenium.webdriver.chrome.service import Service
# import tiktoken
#
# app = Flask(__name__)
# app.secret_key = 'supersecretkey'
#
# # Setup paths
# base_dir = os.path.abspath(os.path.dirname(__file__))
# faiss_index_path = os.path.join(base_dir, 'faiss_index.pkl')
# chunks_index_path = os.path.join(base_dir, 'chunks.pkl')
#
# # Initialize the sentence transformer model
# embedder = SentenceTransformer('all-MiniLM-L6-v2')
# EMBEDDINGS_API_URL = "http://localhost:1234/v1/embeddings"
# COMPLETIONS_API_URL = "http://localhost:1234/v1/chat/completions"
# OPENAI_API_KEY = 'lm-studio'
# MODEL_IDENTIFIER = "model-identifier"  # model-identifier # text-embedding-3-small
#
# faiss_index = None
# chunks = []
# nlist = 100  # Number of clusters
# nprobe = 10  # Number of clusters to search
#
#
# def setup_driver():
#     service = Service(ChromeDriverManager().install())
#     options = webdriver.ChromeOptions()
#     options.add_argument('--headless')
#     options.add_argument('--disable-gpu')
#     options.add_argument('--no-sandbox')
#     driver = webdriver.Chrome(service=service, options=options)
#     return driver
#
#
# def crawl_website(url, max_retries=3):
#     print("url", url)
#     pages = set()
#     print("pages", pages)
#     to_visit = [url]
#     print("to_visit", to_visit)
#     visited = set()
#     print("visited", visited)
#     print("to_visit", to_visit)
#     while to_visit:
#         current_url = to_visit.pop()
#         print("current_url", current_url)
#         print("visited", visited)
#         if current_url in visited:
#             continue
#
#         retries = 0
#         while retries < max_retries:
#             try:
#                 headers = {
#                     'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'
#                 }
#                 response = requests.get(current_url, headers=headers)
#                 print("response while", response)
#                 response.raise_for_status()
#                 break
#             except requests.exceptions.RequestException as e:
#                 retries += 1
#                 time.sleep(130)
#                 if retries == max_retries:
#                     continue
#
#         visited.add(current_url)
#         print("visited", visited)
#         pages.add(current_url)
#         print("pages", pages)
#         soup = BeautifulSoup(response.text, 'html.parser')
#         print("soup", soup)
#         for a_tag in soup.find_all('a', href=True):
#             link = a_tag['href']
#             print("link", link)
#             full_link = urljoin(url, link)
#             print("full_link", full_link)
#             if url in full_link and full_link not in visited and full_link not in to_visit:
#                 to_visit.append(full_link)
#                 print("to_visit IF", to_visit)
#     print("pagespages", pages)
#     return pages
#
#
# def extract_text_from_pages(pages):
#     driver = setup_driver()
#     print("driver", driver)
#     content = ""
#     print("pages", pages)
#     for page in pages:
#         print("page", page)
#         retries = 0
#         success = False
#         while retries < 3:
#             try:
#                 driver.get(page)
#                 time.sleep(10)
#                 success = True
#                 break
#             except Exception as e:
#                 retries += 1
#                 time.sleep(12)
#
#         if not success:
#             continue
#
#         soup = BeautifulSoup(driver.page_source, 'html.parser')
#         print("soup", soup)
#         texts = soup.find_all(string=True)
#         print("texts", texts)
#         visible_texts = filter(tag_visible, texts)
#         print("visible_texts", visible_texts)
#         visible_content = " ".join(t.strip() for t in visible_texts)
#         print("visible_content", visible_content)
#         if visible_content:
#             content += visible_content + " "
#
#     driver.quit()
#     print("content.strip()", content.strip())
#     return content.strip()
#
#
# def tag_visible(element):
#     if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
#         return False
#     if isinstance(element, Comment):
#         return False
#     return True
#
#
# def get_embedding(text, model="model-identifier"):
#     payload = {
#         "input": [text],
#         "model": model
#     }
#     headers = {
#         "Content-Type": "application/json",
#         "Authorization": f"Bearer {OPENAI_API_KEY}"
#     }
#     print("payload", payload)
#     response = requests.post(EMBEDDINGS_API_URL, headers=headers, json=payload)
#     print("response", response)
#     if response.status_code == 200:
#         embedding = response.json()['data'][0]['embedding']
#         print(f"Embedding dimension: {len(embedding)}")  # Debug print for embedding dimension
#         return embedding
#         # return response.json()['data'][0]['embedding']
#     else:
#         raise Exception(f"Error: {response.status_code} - {response.text}")
#
#
# def prepare_vector_store(content):
#     global faiss_index, chunks
#     print("faiss_index", faiss_index)
#     print("chunks", chunks)
#     print("content", content)
#     chunks = [content[i:i + 512] for i in range(0, len(content), 512)]
#     print("chunks", chunks)
#
#     embeddings = []
#     for chunk in chunks:
#         embedding = get_embedding(chunk, model="model-identifier")
#         embeddings.append(embedding)
#
#     dimension = len(embeddings[0])
#     # Create and train the FAISS index
#     nlist = 100
#     nlist = min(nlist, len(embeddings))
#     print(f"Adjusted nlist: {nlist}")  # Debug print for nlist
#     quantizer = faiss.IndexFlatL2(dimension)
#     faiss_index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
#     assert not faiss_index.is_trained
#     faiss_index.train(np.array(embeddings).astype(np.float32))
#     assert faiss_index.is_trained
#
#     faiss_index.add(np.array(embeddings).astype(np.float32))
#     faiss_index.nprobe = nprobe
#
#
# @app.route('/')
# def home():
#     return render_template('index.html')
#
#
# @app.route('/view_chunks')
# def view_chunks():
#     global chunks
#     print("chunks", chunks)
#     return "<br>".join(chunks)
#
#
# @app.route('/view_embeddings')
# def view_embeddings():
#     global faiss_index
#     if faiss_index is None:
#         return "FAISS index is not initialized."
#
#     embeddings = faiss_index.reconstruct_n(0, faiss_index.ntotal)
#     embeddings_list = [emb.tolist() for emb in embeddings]
#     print("embeddings_list", embeddings_list)
#     return f"Number of embeddings: {len(embeddings_list)}<br>" + "<br>".join(map(str, embeddings_list))
#
#
# @app.route('/view_text_and_embeddings')
# def view_text_and_embeddings():
#     global faiss_index, chunks
#     if faiss_index is None or not chunks:
#         return "FAISS index or chunks are not initialized."
#
#     num_embeddings = faiss_index.ntotal
#     text_embeddings_pairs = []
#     for i in range(num_embeddings):
#         embedding = faiss_index.reconstruct(i)
#         chunk = chunks[i] if i < len(chunks) else "No corresponding chunk found"
#         text_embeddings_pairs.append((chunk, embedding))
#
#     if not text_embeddings_pairs:
#         return "No text and embeddings available."
#
#     html_content = ""
#     for idx, (text, embedding) in enumerate(text_embeddings_pairs):
#         embedding_str = ', '.join(
#             map(str, embedding[:10])) + '...'  # Display the first 10 values of the embedding for brevity
#         text_snippet = text[:100] + '...'  # Display the first 100 characters of the text for brevity
#         html_content += f"<div><b>Text {idx + 1}:</b> {text_snippet}<br><b>Embedding {idx + 1}:</b> [{embedding_str}]<br><br></div>"
#
#     return html_content
#
#
# @app.route('/prepare', methods=['POST'])
# def prepare():
#     url = request.form.get('url')
#     print("url", url)
#     if not url:
#         return render_template('index.html', error="Please provide a URL.")
#
#     try:
#         pages = crawl_website(url)
#         print("pages", pages)
#         content = extract_text_from_pages(pages)
#         print("contentcontent", content)
#
#         if not content:
#             return render_template('index.html', error="Could not extract content from the provided URL.")
#
#         session['url'] = url
#         prepare_vector_store(content)
#         print(">>>>>>")
#     except requests.exceptions.RequestException as e:
#         return render_template('index.html', error=f"Error processing the request: {e}")
#
#     return render_template('index.html', message="Content prepared successfully.", url=url)
#
#
# def query_vector_store(question, top_k=1):
#     try:
#         global faiss_index, chunks
#         # question = request.json.get('question')
#         # top_k = request.json.get('top_k', 1)
#
#         question_embedding = get_embedding(question, model=MODEL_IDENTIFIER)
#         print(f"Query embedding dimension: {len(question_embedding)}")  # Debug print for query embedding dimension
#
#         # question_embedding = np.expand_dims(np.array(question_embedding), axis=0)
#         question_embedding = np.expand_dims(np.array(question_embedding).astype(np.float32), axis=0)
#         print("question_embedding", question_embedding)
#
#         distances, indices = faiss_index.search(question_embedding, top_k)
#         print("indicesindices", indices)
#         results = [chunks[idx] for idx in indices[0]]
#         print("results", results)
#         return results
#     except Exception as e:
#         print(f"Error: {e}")
#         return e
#
#
# @app.route('/ask', methods=['POST'])
# def ask():
#     user_input = request.form.get('text')
#     if not user_input:
#         return render_template('index.html', error="Please ask a question.")
#
#     try:
#         results = query_vector_store(user_input, top_k=1)
#         print("resultsresults", results)
#         # best_chunk = " ".join(results) if results else "No relevant information found."
#         best_chunk = results
#         payload = {
#             "messages": [
#                 {"role": "system",
#                  "content": "You are a helpful assistant. Answer the question based on the provided context."},
#                 {"role": "user", "content": f"Context: {best_chunk} Question: {user_input}"}
#             ],
#             "temperature": 0.7,
#             "max_tokens": -1,
#             "stream": False
#         }
#
#         headers = {"Content-Type": "application/json"}
#         response = requests.post(COMPLETIONS_API_URL, json=payload, headers=headers)
#         if response.status_code == 200:
#             data = response.json()
#             answer = data['choices'][0]['message']['content']
#         else:
#             answer = "Error: Could not get a response from LM Studio."
#     except requests.exceptions.RequestException as e:
#         return render_template('index.html', error=f"Error processing the request: {e}")
#
#     return render_template('index.html', question=user_input, answer=answer)
#
#
# if __name__ == "__main__":
#     app.run(debug=True)


# from flask import Flask, request, render_template, session
# import requests
# from bs4 import BeautifulSoup, Comment
# from selenium import webdriver
# from webdriver_manager.chrome import ChromeDriverManager
# from selenium.webdriver.chrome.service import Service
# from datasets import load_dataset, Dataset, DatasetDict, load_from_disk
# import time
# import os
# from urllib.parse import urljoin
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np
#
# app = Flask(__name__)
# app.secret_key = 'supersecretkey'
#
# # Initialize variables
# EMBEDDINGS_API_URL = "http://localhost:1234/v1/embeddings"
# MODEL_IDENTIFIER = "model-identifier"  # Replace with your model identifier
# OPENAI_API_KEY = 'lm-studio'
# dataset_path = 'embedded_testdata_nomic_embed_text_v1.5'
#
# # Load existing dataset or create a new one
# try:
#     dataset = load_dataset('lemon-mint/embedded_testdata_nomic_embed_text_v1.5')
# except FileNotFoundError:
#     dataset = DatasetDict({
#         'train': Dataset.from_dict({
#             'user_id': [],
#             'url': [],
#             'datetime': [],
#             'text': [],
#             'embedding': []
#         })
#     })
#
#
# def setup_driver():
#     service = Service(ChromeDriverManager().install())
#     options = webdriver.ChromeOptions()
#     options.add_argument('--headless')
#     options.add_argument('--disable-gpu')
#     options.add_argument('--no-sandbox')
#     driver = webdriver.Chrome(service=service, options=options)
#     return driver
#
#
# def crawl_website(url, max_retries=3):
#     pages = set()
#     to_visit = [url]
#     visited = set()
#     while to_visit:
#         current_url = to_visit.pop()
#         if current_url in visited:
#             continue
#
#         retries = 0
#         while retries < max_retries:
#             try:
#                 headers = {
#                     'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'
#                 }
#                 response = requests.get(current_url, headers=headers)
#                 response.raise_for_status()
#                 break
#             except requests.exceptions.RequestException:
#                 retries += 1
#                 time.sleep(3)
#                 if retries == max_retries:
#                     continue
#
#         visited.add(current_url)
#         pages.add(current_url)
#         soup = BeautifulSoup(response.text, 'html.parser')
#         for a_tag in soup.find_all('a', href=True):
#             link = a_tag['href']
#             full_link = urljoin(url, link)
#             if url in full_link and full_link not in visited and full_link not in to_visit:
#                 to_visit.append(full_link)
#     return pages
#
#
# def extract_text_from_pages(pages):
#     driver = setup_driver()
#     content = ""
#     for page in pages:
#         retries = 0
#         success = False
#         while retries < 3:
#             try:
#                 driver.get(page)
#                 time.sleep(3)
#                 success = True
#                 break
#             except Exception:
#                 retries += 1
#                 time.sleep(3)
#
#         if not success:
#             continue
#
#         soup = BeautifulSoup(driver.page_source, 'html.parser')
#         texts = soup.find_all(string=True)
#         visible_texts = filter(tag_visible, texts)
#         visible_content = " ".join(t.strip() for t in visible_texts)
#         if visible_content:
#             content += visible_content + " "
#
#     driver.quit()
#     return content.strip()
#
#
# def tag_visible(element):
#     if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
#         return False
#     if isinstance(element, Comment):
#         return False
#     return True
#
#
# def get_embedding(text):
#     payload = {
#         "input": [text],
#         "model": MODEL_IDENTIFIER
#     }
#     headers = {
#         "Content-Type": "application/json",
#         "Authorization": f"Bearer {OPENAI_API_KEY}"
#     }
#     response = requests.post(EMBEDDINGS_API_URL, headers=headers, json=payload)
#     if response.status_code == 200:
#         return response.json()['data'][0]['embedding']
#     else:
#         raise Exception(f"Error: {response.status_code} - {response.text}")
#
#
# @app.route('/')
# def home():
#     return render_template('index.html')
#
#
# @app.route('/prepare', methods=['POST'])
# def prepare():
#     url = request.form.get('url')
#     user_id = request.form.get('user_id')
#     if not url or not user_id:
#         return render_template('index.html', error="Please provide both a URL and User ID.")
#
#     try:
#         pages = crawl_website(url)
#         print("pages", pages)
#         content = extract_text_from_pages(pages)
#         print("content", content)
#         if not content:
#             return render_template('index.html', error="Could not extract content from the provided URL.")
#
#         embedding = get_embedding(content)
#         print("embedding", embedding)
#         datetime_now = time.strftime("%Y-%m-%d %H:%M:%S")
#
#         new_data = {
#             'user_id': [user_id],
#             'url': url,
#             'datetime': datetime_now,
#             'text': content,
#             'embedding': embedding
#         }
#
#         # Update the dataset with new data
#         global dataset
#         print("dataset>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>", dataset)
#         dataset['train'] = dataset['train'].add_item(new_data)
#         print("dataset", dataset)
#         # Save updated dataset locally
#         dataset.save_to_disk(dataset_path)
#         print("datasetSAVEEEEEEEEEEEEEEEEEEE", dataset)
#         return render_template('index.html', message="Content prepared and saved successfully.", url=url)
#     except Exception as e:
#         return render_template('index.html', error=f"Error processing the request: {e}")
#
#
# @app.route('/view_dataset')
# def view_dataset():
#     try:
#         # Load the dataset from the local directory
#         dataset = load_from_disk(dataset_path)
#
#         # Convert the dataset to HTML for viewing
#         dataset_html = dataset['train'].to_pandas().to_html()
#
#         return render_template('view_dataset.html', dataset_html=dataset_html)
#     except Exception as e:
#         return f"Error loading dataset: {e}"
#
#
# @app.route('/ask', methods=['POST'])
# def ask():
#     question = request.form.get('text')
#     if not question:
#         return render_template('index.html', error="Please enter a question.")
#
#     try:
#         # Get embedding for the question
#         question_embedding = get_embedding(question)
#
#         # Load the dataset from the local directory
#         dataset = load_from_disk(dataset_path)
#
#         # Get all embeddings and texts from the dataset
#         embeddings = np.array(dataset['train']['embedding'])
#         texts = dataset['train']['text']
#
#         # Compute cosine similarities
#         similarities = cosine_similarity([question_embedding], embeddings).flatten()
#
#         # Get the index of the most similar embedding
#         most_similar_index = np.argmax(similarities)
#
#         # Get the corresponding text
#         answer = texts[most_similar_index]
#
#         return render_template('index.html', question=question, answer=answer)
#     except Exception as e:
#         return render_template('index.html', error=f"Error processing the question: {e}")
#
#
# if __name__ == "__main__":
#     app.run(debug=True)

from flask import Flask, request, render_template, session
import requests
from bs4 import BeautifulSoup, Comment
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
# from datasets import load_dataset, Dataset, DatasetDict, load_from_disk

from datasets import Dataset, DatasetDict, load_from_disk, concatenate_datasets, load_dataset
import time
import os
from urllib.parse import urljoin
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
ITEMS_PER_PAGE = 10
import pandas as pd

app = Flask(__name__)
app.secret_key = 'supersecretkey'

# Initialize variables
EMBEDDINGS_API_URL = "http://localhost:1234/v1/embeddings"
MODEL_IDENTIFIER = "model-identifier"  # Replace with your model identifier
OPENAI_API_KEY = 'lm-studio'
dataset_path = 'embedded_testdata_nomic_embed_text_v1.5'
COMPLETIONS_API_URL = "http://localhost:1234/v1/chat/completions"

# try:
#     dataset = load_dataset('lemon-mint/embedded_testdata_nomic_embed_text_v1.5')
# except Exception as e:
    # Load existing dataset or create a new one
# try:
if os.path.exists(dataset_path):
    dataset = load_from_disk(dataset_path)
    print(">>>>>>>>>>", dataset)
else:
    dataset = DatasetDict({
        'train': Dataset.from_dict({
            'user_id': [],
            'url': [],
            'datetime': [],
            'text': [],
            'embedding': []
        })
    })


# try:
#     dataset = load_dataset('lemon-mint/embedded_testdata_nomic_embed_text_v1.5')
# except FileNotFoundError:
#     dataset = DatasetDict({
#         'train': Dataset.from_dict({
#             'user_id': [],
#             'url': [],
#             'datetime': [],
#             'text': [],
#             'embedding': []
#         })
#     })

def setup_driver():
    service = Service(ChromeDriverManager().install())
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    driver = webdriver.Chrome(service=service, options=options)
    return driver


def crawl_website(url, max_retries=3):
    print("url", url)
    pages = set()
    pages_to_scrap = []
    to_visit = [url]
    visited = set()
    while to_visit:
        current_url = to_visit.pop()
        print("current_url", current_url)
        if current_url in visited:
            continue

        retries = 0
        while retries < max_retries:
            try:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'
                }
                response = requests.get(current_url, headers=headers)
                response.raise_for_status()
                break
            except requests.exceptions.RequestException:
                retries += 1
                time.sleep(3)
                if retries == max_retries:
                    continue

        visited.add(current_url)
        pages.add(current_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        print("visited", visited)
        print("pages", pages)
        for a_tag in soup.find_all('a', href=True):
            link = a_tag['href']
            print("link", link)
            full_link = urljoin(url, link)
            print("full_link", full_link)
            pages_to_scrap.append(full_link)
            if url in full_link and full_link not in visited and full_link not in to_visit:
                to_visit.append(full_link)
                pages_to_scrap.append(full_link)
                print("to_visit in if condition", to_visit)
        print("to_visit>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>", to_visit)
        print("pages_to_scrap@@@@@@@@@@@@@@@@@@@", pages_to_scrap)
    return pages_to_scrap


def extract_text_from_pages(pages):
    driver = setup_driver()
    content = ""
    print("pages>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>", pages)
    for page in pages:
        retries = 0
        success = False
        while retries < 3:
            try:
                driver.get(page)
                time.sleep(3)
                success = True
                break
            except Exception:
                retries += 1
                time.sleep(3)

        if not success:
            continue

        soup = BeautifulSoup(driver.page_source, 'html.parser')
        texts = soup.find_all(string=True)
        visible_texts = filter(tag_visible, texts)
        visible_content = " ".join(t.strip() for t in visible_texts)
        if visible_content:
            content += visible_content + " "

    driver.quit()
    print("content.strip()", content.strip())
    return content.strip()


def tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True


def get_embedding(text):
    payload = {
        "input": [text],
        "model": MODEL_IDENTIFIER
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }
    response = requests.post(EMBEDDINGS_API_URL, headers=headers, json=payload)
    print(">>>>>>>>>>>>>>>response", response)
    if response.status_code == 200:
        return response.json()['data'][0]['embedding']
    else:
        raise Exception(f"Error: {response.status_code} - {response.text}")


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/prepare', methods=['POST'])
def prepare():
    url = request.form.get('url')
    # user_id = request.form.get('user_id')
    # if not url or not user_id:
    #     return render_template('index.html', error="Please provide both a URL and User ID.")

    try:
        pages = crawl_website(url)
        content = extract_text_from_pages(pages)
        print("contentcontent", content)
        if not content:
            return render_template('index.html', error="Could not extract content from the provided URL.")

        embedding = get_embedding(content)
        print("embedding>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>", embedding)
        datetime_now = time.strftime("%Y-%m-%d %H:%M:%S")

        new_data = Dataset.from_dict({
            # 'user_id': [user_id],
            'url': [url],
            'datetime': [datetime_now],
            'text': [content],
            'embedding': [embedding]
        })
        print("new_data", new_data)
        # Update the dataset with new data
        global dataset
        dataset['train'] = concatenate_datasets([dataset['train'], new_data])
        print("dataset")
        # Save updated dataset locally
        dataset.save_to_disk(dataset_path)
        print("dataset1234")
        return render_template('index.html', message="Content prepared and saved successfully.", url=url)
    except Exception as e:
        return render_template('index.html', error=f"Error processing the request: {e}")


# @app.route('/view_dataset')
# def view_dataset():
#     try:
#         # Load the dataset from the local directory
#         dataset = load_from_disk(dataset_path)
#
#         # Convert the dataset to HTML for viewing
#         dataset_html = dataset['train'].to_pandas().to_html()
#
#         return render_template('view_dataset.html', dataset_html=dataset_html)
#     except Exception as e:
#         return f"Error loading dataset: {e}"
ITEMS_PER_PAGE = 10
import pandas as pd
# @app.route('/view_dataset')
# def view_dataset():
#     page = request.args.get('page', 1, type=int)
#     start = (page - 1) * ITEMS_PER_PAGE
#     end = start + ITEMS_PER_PAGE
#     dataset_slice = dataset[start:end]
#
#     total_pages = (len(dataset) + ITEMS_PER_PAGE - 1) // ITEMS_PER_PAGE
#
#     return render_template(
#         'view_dataset.html',
#         dataset_html=dataset_slice.to_pandas().to_html(),
#         page=page,
#         total_pages=total_pages
#     )
@app.route('/view_dataset', methods=['GET'])
def view_dataset():
    page = request.args.get('page', 1, type=int)
    start = (page - 1) * ITEMS_PER_PAGE
    end = start + ITEMS_PER_PAGE

    # Access the train split and slice the dataset
    dataset_slice = dataset['train'][start:end]
    # dataset_slice = dataset['train'].select(range(start, end))

    # Convert the dataset slice to a pandas DataFrame
    df = pd.DataFrame(dataset_slice)

    # Generate HTML table from the DataFrame
    dataset_html = df.to_html(index=False)

    total_items = len(dataset['train'])
    print("total_items", total_items)
    total_pages = (total_items + ITEMS_PER_PAGE - 1) // ITEMS_PER_PAGE
    print("total_pages",total_pages)

    return render_template(
        'view_dataset.html',
        dataset_html=dataset_html,
        page=page,
        total_pages=total_pages
    )


@app.route('/')
def index():
    return render_template('index.html')


# @app.route('/ask', methods=['POST'])
# def ask():
#     question = request.form.get('text')
#     if not question:
#         return render_template('index.html', error="Please enter a question.")
#
#     try:
#         # Get embedding for the question
#         question_embedding = get_embedding(question)
#
#         # Load the dataset from the local directory
#         dataset = load_from_disk(dataset_path)
#
#         # Get all embeddings and texts from the dataset
#         embeddings = np.array(dataset['train']['embedding'])
#         texts = dataset['train']['text']
#
#         # Compute cosine similarities
#         similarities = cosine_similarity([question_embedding], embeddings).flatten()
#
#         # Get the index of the most similar embedding
#         most_similar_index = np.argmax(similarities)
#
#         # Get the corresponding text
#         answer = texts[most_similar_index]
#
#         return render_template('index.html', question=question, answer=answer)
#     except Exception as e:
#         return render_template('index.html', error=f"Error processing the question: {e}")
@app.route('/ask', methods=['POST'])
def ask():
    question = request.form.get('text')
    if not question:
        return render_template('index.html', error="Please enter a question.")

    try:
        # Get embedding for the question
        question_embedding = get_embedding(question)

        # Load the dataset from the local directory
        dataset = load_from_disk(dataset_path)

        # Get all embeddings and texts from the dataset
        embeddings = np.array(dataset['train']['embedding'])
        texts = dataset['train']['text']

        # Compute cosine similarities
        similarities = cosine_similarity([question_embedding], embeddings).flatten()

        # Get the index of the most similar embedding
        most_similar_index = np.argmax(similarities)

        # Get the corresponding text
        best_chunk = texts[most_similar_index]

        # Use the retrieved text to generate an answer
        payload = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant. Answer the question based on the provided context."},
                {"role": "user", "content": f"Context: {best_chunk} Question: {question}"}
            ],
            "temperature": 0.7,
            "max_tokens": 150,  # Adjust as necessary
            "stream": False
        }

        headers = {"Content-Type": "application/json"}
        response = requests.post(COMPLETIONS_API_URL, json=payload, headers=headers)
        if response.status_code == 200:
            data = response.json()
            answer = data['choices'][0]['message']['content']
        else:
            answer = "Error: Could not get a response from the language model."

        return render_template('index.html', question=question, answer=answer)
    except Exception as e:
        return render_template('index.html', error=f"Error processing the question: {e}")


if __name__ == "__main__":
    app.run(debug=True)

