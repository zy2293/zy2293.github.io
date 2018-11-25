
# coding: utf-8

# This notebook (and the slides from lecture 8) will help you go straight from training a model in Colab to deploying it in a webpage with TensorFlow.js - without having to leave the browser.

# Configure this notebook to work with your GitHub account by populating these fields.

# In[1]:


get_ipython().system('pip install tensorflowjs')


# In[2]:


# your github username
USER_NAME = "zy2293" 

# the email associated with your commits
# (may not matter if you leave it as this)
USER_EMAIL = "zy2293@columbia.edu" 

# the user token you've created (see the lecture 8 slides for instructions)
TOKEN = "5be437ac858dec0bf496ab7d1dfe58df0f7c8624" 

# site name
# for example, if my user_name is "foo", then this notebook will create
# a site at https://foo.github.io/hw4/
SITE_NAME = "hw4"


# Next, run this cell to configure git.

# In[3]:


get_ipython().system('git config --global user.email {USER_NAME}')
get_ipython().system('git config --global user.name  {USER_EMAIL}')


# Clone your GitHub pages repo (see the lecture 8 slides for instructions on how to create one).

# In[4]:


import os
repo_path = USER_NAME + '.github.io'
if not os.path.exists(os.path.join(os.getcwd(), repo_path)):
  get_ipython().system('git clone https://{USER_NAME}:{TOKEN}@github.com/{USER_NAME}/{USER_NAME}.github.io')


# In[5]:


os.chdir(repo_path)
get_ipython().system('git pull')


# Create a folder for your site.

# In[6]:


project_path = os.path.join(os.getcwd(), SITE_NAME)
if not os.path.exists(project_path): 
  os.mkdir(project_path)
os.chdir(project_path)


# These paths will be used by the converter script.

# In[7]:


# DO NOT MODIFY
MODEL_DIR = os.path.join(project_path, "model_js")
if not os.path.exists(MODEL_DIR):
  os.mkdir(MODEL_DIR)


# In[8]:


MODEL_DIR


# As an example, we will create and vectorize a few documents. (Check out https://www.gutenberg.org/ for a bunch of free e-books.)

# In[114]:


# read in first 1000 sentences from Alice in Wonderland
import urllib.request
import nltk

def read_books(url, starting_sent):
    response = urllib.request.urlopen(url)
    data = response.readlines()
    
    book = []
    read = False # won't start reading until meeting the first sentence in the content
    count = 0
    for line in data:
        line = line.decode('utf-8')
        if line.startswith('\r\n'):
            pass
        else:
            if line.startswith(starting_sent):
                read = True
            if read:
                count += 1
                book.append(line[:-2])
    
    book = " ".join(book)   
#    sentences = re.split(r' *[\.\?!][\'"\)\]]* *', book) # split texts into sentences
    sentences = nltk.sent_tokenize(book)
    if len(sentences) >= 1000:
        return sentences[:1000]
    else:
        return sentences


# In[117]:


# Dracula
dracula_url = "http://www.gutenberg.org/cache/epub/345/pg345.txt"
start_line = "_3 May. Bistritz._--Left Munich at 8:35 P. M., on 1st May"
dracula = read_books(dracula_url, start_line)


# In[119]:


# Tower of London
tower_url = "https://www.gutenberg.org/files/58271/58271-0.txt"
start_line = "The Tower of London is the most interesting fortress in Great"
tower_of_london = read_books(tower_url, start_line)


# In[121]:


# Blue Jacket
jacket_url = "http://www.gutenberg.org/cache/epub/58270/pg58270.txt"
start_line = "The big bell of Woolwich Dockyard had just commenced its"
blue_jacket = read_books(jacket_url, start_line)


# In[123]:


all_books = dracula
all_books.extend(tower_of_london)
all_books.extend(blue_jacket)


# In[213]:


x_train = all_books
y_train = np.zeros(len(x_train))
# 0: Dracula; 1: Tower of London; 2: Blue Jacket
y_train[1000:2000] = 1
y_train[2000:] = 2


# Tokenize the documents, create a word index (word -> number).

# In[214]:


max_len = 100
num_words = 10000
from keras.preprocessing.text import Tokenizer
# Fit the tokenizer on the training data
t = Tokenizer(num_words=num_words)
t.fit_on_texts(x_train)


# In[215]:


len(t.word_index)


# Here's how we vectorize a document.

# In[216]:


vectorized = t.texts_to_sequences(dracula)
# print(vectorized)


# Apply padding if necessary.

# In[217]:


from keras.preprocessing.sequence import pad_sequences
padded = pad_sequences(vectorized, maxlen=max_len, padding='post')


# In[218]:


print(padded)


# We will save the word index in metadata. Later, we'll use it to convert words typed in the browser to numbers for prediction.

# In[219]:


metadata = {
  'word_index': t.word_index,
  'max_len': max_len,
  'vocabulary_size': num_words,
}


# Define a model.

# In[267]:


embedding_size = 20
n_classes = 3
epochs = 15

import keras
model = keras.Sequential()
model.add(keras.layers.Embedding(num_words, embedding_size, input_shape=(max_len,)))
#model.add(keras.layers.Flatten())
model.add(keras.layers.LSTM(128, return_sequences=True))
model.add(keras.layers.LSTM(32, return_sequences=False))
#model.add(keras.layers.Dense(32, activation = 'relu'))
model.add(keras.layers.Dense(3, activation='softmax'))
model.compile('rmsprop', 'sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()


# Prepare some training data.

# In[221]:


x_train = t.texts_to_sequences(x_train)
x_train = pad_sequences(x_train, maxlen=max_len, padding='post')
print(x_train)


# In[268]:


model.fit(x_train, y_train, epochs=epochs)


# Demo using the model to make predictions.

# In[265]:


test_example = "Left Munich at 8:35 P. M., on 1st May, arriving at Vienna early next morning."
x_test = t.texts_to_sequences([test_example])
x_test = pad_sequences(x_test, maxlen=max_len, padding='post')
print(x_test)


# In[269]:


preds = model.predict(x_test)
print(preds)
import numpy as np
print(np.argmax(preds))


# Convert the model

# In[270]:


import json
import tensorflowjs as tfjs

metadata_json_path = os.path.join(MODEL_DIR, 'metadata.json')
json.dump(metadata, open(metadata_json_path, 'wt'))
tfjs.converters.save_keras_model(model, MODEL_DIR)
print('\nSaved model artifcats in directory: %s' % MODEL_DIR)


# Write an index.html and an index.js file configured to load our model.

# In[271]:


index_html = """
<!doctype html>

<body>
  <style>
    #textfield {
      font-size: 120%;
      width: 60%;
      height: 200px;
    }
  </style>
  <h1>
    Title
  </h1>
  <hr>
  <div class="create-model">
    <button id="load-model" style="display:none">Load model</button>
  </div>
  <div>
    <div>
      <span>Vocabulary size: </span>
      <span id="vocabularySize"></span>
    </div>
    <div>
      <span>Max length: </span>
      <span id="maxLen"></span>
    </div>
  </div>
  <hr>
  <div>
    <select id="example-select" class="form-control">
      <option value="example1">Dracula</option>
      <option value="example2">The Tower of London</option>
      <option value="example3">Blue Jackets</option>
    </select>
  </div>
  <div>
    <textarea id="text-entry"></textarea>
  </div>
  <hr>
  <div>
    <span id="status">Standing by.</span>
  </div>

  <script src='https://cdn.jsdelivr.net/npm/@tensorflow/tfjs/dist/tf.min.js'></script>
  <script src='index.js'></script>
</body>
"""


# In[272]:


index_js = """
const HOSTED_URLS = {
  model:
      'model_js/model.json',
  metadata:
      'model_js/metadata.json'
};

const examples = {
  'example1':
      'Buda-Pesth seems a wonderful place.',
  'example2':
      'The Port of London held a high position from the beginning of the history of Western Europe.',
  'example3':
      'Clare knew him at once, Crushe having been the second lieutenant of his last ship, and as such having twice endeavoured to get him flogged.'
};

function status(statusText) {
  console.log(statusText);
  document.getElementById('status').textContent = statusText;
}

function showMetadata(metadataJSON) {
  document.getElementById('vocabularySize').textContent =
      metadataJSON['vocabulary_size'];
  document.getElementById('maxLen').textContent =
      metadataJSON['max_len'];
}

function settextField(text, predict) {
  const textField = document.getElementById('text-entry');
  textField.value = text;
  doPredict(predict);
}

function setPredictFunction(predict) {
  const textField = document.getElementById('text-entry');
  textField.addEventListener('input', () => doPredict(predict));
}

function disableLoadModelButtons() {
  document.getElementById('load-model').style.display = 'none';
}

function doPredict(predict) {
  const textField = document.getElementById('text-entry');
  const result = predict(textField.value);
  score_string = "Class scores: ";
  for (var x in result.score) {
    score_string += x + " ->  " + result.score[x].toFixed(3) + ", "
  }
  //console.log(score_string);
  status(
      score_string + ' elapsed: ' + result.elapsed.toFixed(3) + ' ms)');
}

function prepUI(predict) {
  setPredictFunction(predict);
  const testExampleSelect = document.getElementById('example-select');
  testExampleSelect.addEventListener('change', () => {
    settextField(examples[testExampleSelect.value], predict);
  });
  settextField(examples['example1'], predict);
}

async function urlExists(url) {
  status('Testing url ' + url);
  try {
    const response = await fetch(url, {method: 'HEAD'});
    return response.ok;
  } catch (err) {
    return false;
  }
}

async function loadHostedPretrainedModel(url) {
  status('Loading pretrained model from ' + url);
  try {
    const model = await tf.loadModel(url);
    status('Done loading pretrained model.');
    disableLoadModelButtons();
    return model;
  } catch (err) {
    console.error(err);
    status('Loading pretrained model failed.');
  }
}

async function loadHostedMetadata(url) {
  status('Loading metadata from ' + url);
  try {
    const metadataJson = await fetch(url);
    const metadata = await metadataJson.json();
    status('Done loading metadata.');
    return metadata;
  } catch (err) {
    console.error(err);
    status('Loading metadata failed.');
  }
}

class Classifier {

  async init(urls) {
    this.urls = urls;
    this.model = await loadHostedPretrainedModel(urls.model);
    await this.loadMetadata();
    return this;
  }

  async loadMetadata() {
    const metadata =
        await loadHostedMetadata(this.urls.metadata);
    showMetadata(metadata);
    this.maxLen = metadata['max_len'];
    console.log('maxLen = ' + this.maxLen);
    this.wordIndex = metadata['word_index']
  }

  predict(text) {
    // Convert to lower case and remove all punctuations.
    const inputText =
        text.trim().toLowerCase().replace(/(\.|\,|\!)/g, '').split(' ');
    // Look up word indices.
    const inputBuffer = tf.buffer([1, this.maxLen], 'float32');
    for (let i = 0; i < inputText.length; ++i) {
      const word = inputText[i];
      inputBuffer.set(this.wordIndex[word], 0, i);
      //console.log(word, this.wordIndex[word], inputBuffer);
    }
    const input = inputBuffer.toTensor();
    //console.log(input);

    status('Running inference');
    const beginMs = performance.now();
    const predictOut = this.model.predict(input);
    //console.log(predictOut.dataSync());
    const score = predictOut.dataSync();//[0];
    predictOut.dispose();
    const endMs = performance.now();

    return {score: score, elapsed: (endMs - beginMs)};
  }
};

async function setup() {
  if (await urlExists(HOSTED_URLS.model)) {
    status('Model available: ' + HOSTED_URLS.model);
    const button = document.getElementById('load-model');
    button.addEventListener('click', async () => {
      const predictor = await new Classifier().init(HOSTED_URLS);
      prepUI(x => predictor.predict(x));
    });
    button.style.display = 'inline-block';
  }

  status('Standing by.');
}

setup();
"""


# In[273]:


with open('index.html','w') as f:
  f.write(index_html)
  
with open('index.js','w') as f:
  f.write(index_js)


# In[274]:


get_ipython().system('ls')


# Commit and push everything. Note: we're storing large binary files in GitHub, this isn't ideal (if you want to deploy a model down the road, better to host it in a cloud storage bucket).

# In[276]:


get_ipython().system('git add . ')
get_ipython().system('git commit -m "colab -> github"')
get_ipython().system('git push https://{USER_NAME}:{TOKEN}@github.com/{USER_NAME}/{USER_NAME}.github.io/ master')


# All done! Hopefully everything worked. You may need to wait a few moments for the changes to appear in your site. If not working, check the JavaScript console for errors (in Chrome: View -> Developer -> JavaScript Console).

# In[277]:


print("Now, visit https://%s.github.io/%s/" % (USER_NAME, SITE_NAME))


# If you are debugging and Chrome is failing to pick up your changes, though you've verified they're present in your GitHub repo, see the second answer to: https://superuser.com/questions/89809/how-to-force-refresh-without-cache-in-google-chrome
