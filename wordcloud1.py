# -*- coding: utf-8 -*-
"""
Created on Wed May 12 10:35:12 2021

@author: yashr
"""

from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import pandas as pd

trainx = pd.read_csv("D:\\Sem 4\\ADM\\ADM Project\\trainx.csv")
trainy = pd.read_csv("D:\\Sem 4\\ADM\\ADM Project\\trainy.csv")

train = pd.DataFrame()
train_gen = pd.DataFrame()

train['Text'] = trainx.Text
train['Category'] = trainy.Category

traingrp = train.groupby('Category')
tech = traingrp.get_group('tech')
ent =traingrp.get_group('entertainment')
bus = traingrp.get_group('business')
pol = traingrp.get_group('politics')
spo = traingrp.get_group('sport')





trainx_gen = pd.read_csv("D:\\Sem 4\\ADM\\ADM Project\\trainx_gen.csv")
trainy_gen = pd.read_csv("D:\\Sem 4\\ADM\\ADM Project\\trainy_gen.csv")


train_gen['Text'] = trainx_gen['0']
train_gen['Category'] = trainy.Category

traingengrp = train_gen.groupby('Category')
techgen = traingengrp.get_group('tech')
entgen = traingengrp.get_group('entertainment')
busgen = traingengrp.get_group('business')
polgen =  traingengrp.get_group('politics')
spogen =  traingengrp.get_group('sport')


comment_words = ''
stopwords = set(STOPWORDS)
stopwords.update(['s', 'u', 'said', 'say', 'will', 'one', 'made', 'told', 'put',
                  'us', 'mr','new', 'come', 'still', 'now', 'two', 'year', 'many', 'first', 'way',
                  'well', 'want', 'see', 'may', 'need', 'm', 'o', 'make', 'says', 'time',
                  'government','take', 'people'])
  

stopwordsent = stopwords
stopwordsent.update(['government','game', 'firm'])


stopwordsspo = stopwords
stopwordsspo.update(['labour', 'film', 'company'])

def wc(df):
    comment_words = ''
    for val in trainx.Text:
      
    # typecaste each val to string
        val = str(val)
  
    # split the value
        tokens = val.split()
      
    # Converts each token into lowercase
        for i in range(len(tokens)):
            tokens[i] = tokens[i].lower()
      
        comment_words += " ".join(tokens)+" "
    return comment_words
# iterate through the csv file
##############################################

techplt = wc(tech)
wordcloud = WordCloud(width = 800, height = 800,
                background_color ='white',
                stopwords = stopwords,
                min_font_size = 10).generate(techplt)

plt.figure(figsize = (8, 8), facecolor = None)
plt.title("tech original cloud")
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)
  
plt.show()

############################################
entplt = wc(ent)
wordcloud1= WordCloud(width = 800, height = 800,
                background_color ='white',
                stopwords = stopwordsent,
                min_font_size = 10).generate(entplt)

plt.figure(figsize = (8, 8), facecolor = None)
plt.title("Entertainment original cloud")
plt.imshow(wordcloud1)
plt.axis("off")
plt.tight_layout(pad = 0)
  
plt.show()
#####################################
busplt = wc(bus)
wordcloud2 = WordCloud(width = 800, height = 800,
                background_color ='white',
                stopwords = stopwords,
                min_font_size = 10).generate(busplt)

plt.figure(figsize = (8, 8), facecolor = None)
plt.title("Business original cloud")
plt.imshow(wordcloud2)
plt.axis("off")
plt.tight_layout(pad = 0)
  
plt.show()
###################################################
polplt = wc(pol)
wordcloud3 = WordCloud(width = 800, height = 800,
                background_color ='white',
                stopwords = stopwords,
                min_font_size = 10).generate(polplt)

plt.figure(figsize = (8, 8), facecolor = None)
plt.title("Politics original cloud")
plt.imshow(wordcloud3)
plt.axis("off")
plt.tight_layout(pad = 0)
  
plt.show()
####################################################
spoplt = wc(spo)
wordcloud4 = WordCloud(width = 800, height = 800,
                background_color ='white',
                stopwords = stopwordsspo,
                min_font_size = 10).generate(spoplt)

plt.figure(figsize = (8, 8), facecolor = None)
plt.title("sports original cloud")
plt.imshow(wordcloud4)
plt.axis("off")
plt.tight_layout(pad = 0)
  
plt.show()
###############################################
###############################################


techplt = wc(techgen)
wordcloud5 = WordCloud(width = 800, height = 800,
                background_color ='black',
                stopwords = stopwords,
                min_font_size = 10).generate(techplt)

plt.figure(figsize = (8, 8), facecolor = None)
plt.title("Tech generated cloud")
plt.imshow(wordcloud5)
plt.axis("off")
plt.tight_layout(pad = 0)
  
plt.show()

############################################
entplt = wc(entgen)
wordcloud6 = WordCloud(width = 800, height = 800,
                background_color ='black',
                stopwords = stopwordsent,
                min_font_size = 10).generate(entplt)

plt.figure(figsize = (8, 8), facecolor = None)
plt.title("Entertainment generated cloud")
plt.imshow(wordcloud6)
plt.axis("off")
plt.tight_layout(pad = 0)
  
plt.show()
#####################################
busplt = wc(bus)
wordcloud7 = WordCloud(width = 800, height = 800,
                background_color ='black',
                stopwords = stopwords,
                min_font_size = 10).generate(busplt)

plt.figure(figsize = (8, 8), facecolor = None)
plt.title("Business generated cloud")
plt.imshow(wordcloud7)
plt.axis("off")
plt.tight_layout(pad = 0)
  
plt.show()
###################################################
polplt = wc(polgen)
wordcloud8 = WordCloud(width = 800, height = 800,
                background_color ='black',
                stopwords = stopwords,
                min_font_size = 10).generate(polplt)

plt.figure(figsize = (8, 8), facecolor = None)
plt.title("Politics generated cloud")
plt.imshow(wordcloud8)
plt.axis("off")
plt.tight_layout(pad = 0)
  
plt.show()
####################################################
spoplt = wc(spogen)
wordcloud9 = WordCloud(width = 800, height = 800,
                background_color ='black',
                stopwords = stopwordsspo,
                min_font_size = 10).generate(spoplt)

plt.figure(figsize = (8, 8), facecolor = None)
plt.title("Sports  Generated cloud")
plt.imshow(wordcloud9)
plt.axis("off")
plt.tight_layout(pad = 0)
  
plt.show()
###############################################
###############################################