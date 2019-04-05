
# coding: utf-8

# In[16]:


import os
import time
import numpy as np
import pandas as pd
import subprocess


# In[11]:


#no of words pair
LEN = 353


# In[2]:


with open('combined.csv') as f:
    read_data = f.read()


# In[48]:


#words: (w1, w2), score: similarity score
dic = {'words': [],
      'score': []}
for idx, line in enumerate(read_data.split('\n')):
    
    if idx<= LEN and idx != 0:
        wordls = line.split(',')
        
        dic['words'].append( (wordls[0], wordls[1]) )
        dic['score'].append(float(wordls[2]))


# In[66]:


stats_ls = []
count = 0


# In[67]:


word_ls = dic['words']


while count != LEN:
    item = word_ls[count]
    k1, k2, k3 = item[0], item[1] , item[0] + "+" + item[1]
    keyls = [k1, k2, k3]
    
    temp_ls = []
    for keyword in keyls:
        res = subprocess.check_output(["wget", "-U", "Firefox/3.0.15", "http://www.google.com/search?q=" + keyword, "-O", "file.html"])

        with open('file.html', 'r',encoding = 'latin-1') as f:
            html_data = f.read()

        number = html_data.split("<div class=\"sd\" id=\"resultStats\">About ")[1].split(' ')[0].replace(',', '')
        temp_ls.append(number)
        time.sleep(2)
        
    count = count + 1
    stats_ls.append(temp_ls)
    print("Processing Complete: word pair no: ", count)


# In[69]:


len(stats_ls)


# In[70]:


word_file = open('word-count.txt', 'w')
word_file.write("word-1 word-2 pair\n")


for l in stats_ls:
    word_file.write(l[0] + " " + l[1] + " " + l[2] + "\n")

word_file.close()

