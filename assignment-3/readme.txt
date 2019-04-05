The codes are present in the following files:


----------------------------------------------------------------------------------------
#         -      notebook-file                -           python file
----------------------------------------------------------------------------------------
Q1           1-Google-Searches.ipynb                question1_Google_Searches.py
Q2        2-Normalised-Google-Distance.ipynb     question2_Normalised_Google_Distance.py
Q3               3-Word2vec.ipynb                       question3_Word2vec.py

All codes are self explanatory and solution steps can be found in report.pdf



----------------------------------------------------------------
1. Search for keyword for example 'animal' and save the google response to file.html
  wget -U 'Firefox/3.0.15' http://www.google.com/search\?q\=animal -O file.html
2. Search for keyword "resultStats" in the downloaded document 'file.html' 
  - a general count line looks like this "<div class="sd" id="resultStats">About 3,51,000 results</div>"

---------------------------------------------
Combined.csv is the actual human similarity data from:
http://www.cs.technion.ac.il/~gabr/resources/data/wordsim353/

Word2vec API:
https://github.com/3Top/word2vec-api
