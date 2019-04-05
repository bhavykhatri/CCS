
# coding: utf-8

# In[1]:


import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:

# First install the word2vec-api and run it. Use the instructions given in word2vec-api link.
res = subprocess.check_output(["curl", "http://127.0.0.1:5000/word2vec/similarity?w1=sex&w2=sex"])


# In[3]:


#human similarity from 
human_sim = pd.read_csv('combined.csv', header=0)


# In[4]:


w1 = human_sim['Word 1'].values
w2 = human_sim['Word 2'].values
true_sim = human_sim['Human (mean)'].values


# In[5]:


word_sim = []
for i in range(353):
    k1, k2 = w1[i], w2[i]
    res = subprocess.check_output(["curl", "http://127.0.0.1:5000/word2vec/similarity?w1=" +  k1 + "&w2=" + k2])
    word_sim.append(float(res))


# In[6]:


word_sim_arr = np.array(word_sim)


# In[7]:


print(word_sim_arr)


# In[8]:


scaled_sim = 10*(word_sim_arr - np.min(word_sim_arr) )/(np.max(word_sim_arr) - np.min(word_sim_arr))


# In[9]:


plt.rcParams['figure.figsize'] = [6,6]
plt.scatter(true_sim, scaled_sim, facecolors='none', s = 20, edgecolors='b')
plt.xlabel("Human Similarity ratings")
plt.ylabel("Scaled Word 2 vec")
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig('word2vec.jpg')

