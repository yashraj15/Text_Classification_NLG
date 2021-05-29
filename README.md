## Description

This research project aims to provide a comparative analysis on text classification based on three sets of
experiments i.e., using the base data that contains the original training dataset, increasing the size of
training set by generating new articles through natural language generation and lastly, using smaller
training sets ranging from 2-10% of the original training size. I will be using BERT and SVM models for
text classification. The data will be generated using Open AI’s gpt2(Generative Pretrained Transformer 2).
The data used will comprise of long text inputs like articles, literature, reviews, etc. 

## Example Word Clouds of Base training data vs generated Training Data
Business articles original vs generated

<p float="left">
  <img src="/wcplotnew/ent_original.png" width="300" /> 
  <span>      </span>
  <img src="/wcplotnew/business_gen.png" width="300" />
</p>

Political articles original vs generated
<p float="left">
  <img src="/wcplotnew/pol_original.png" width="300" /> 
  <span>      </span>
  <img src="/wcplotnew/pol_gen.png" width="300" />
</p>


## Results

|     Baseline Results                                         |                     |                           |                       |                                |                      |                |
|--------------------------------------------------------------|---------------------|---------------------------|-----------------------|--------------------------------|----------------------|----------------|
|     Training Data                                            |     SVM Accuracy    |     SVM F1     (Micro)    |     SVM F1 (Macro)    |     SVM F1      ( weighted)    |     BERT Accuracy    |     BERT F1    |
|     200 Articles                                             |     80.76           |     80.76                 |     79.48             |     81.022                     |     82.481           |     84.38      |
|     400 Articles                                             |     85.234          |     85.234                |     84.215            |     85.336                     |     86.324           |     87.667     |
|     600 Articles                                             |     87.472          |     87.472                |     86.633            |     87.457                     |     89.556           |     90.428     |
|     800 Articles                                             |     91.275          |     91.275                |     90.802            |     91.263                     |     92.325           |     94.428     |
|     1000 Articles                                            |     94.183          |     94.1834               |     93.822            |     94.172                     |     96.308           |     98.768     |
|     Results – Generated Text – Inequal Distribution          |                     |                           |                       |                                |                      |                |
|     Training Data                                            |     SVM Accuracy    |     SVM F1     (Micro)    |     SVM F1 (Macro)    |     SVM F1      (weighted)     |     BERT Accuracy    |     BERT F1    |
|      (200 articles + 200)                                    |     81.2            |     81.2                  |     80.8              |     81.04                      |     83.8             |     85.1       |
|      (200 articles + 400)                                    |     81.5            |     81.5                  |     81.1              |     81                         |     84.1             |     87.8       |
|      (200 articles + 600)                                    |     83.3            |     83.3                  |     82.9              |     83.1                       |     85.3             |     90.4       |
|     (200 articles + 800)                                     |     84.1            |     84.2                  |     84                |     84.2                       |     91.9             |     93         |
|      (200 articles + 1000)                                   |     87.8            |     87.7                  |     87.76             |     87.8                       |     93.6             |     96.6       |
|     Results – Generated Text – Equal Distribution            |                     |                           |                       |                                |                      |                |
|     Training Data                                            |     SVM Accuracy    |     SVM F1     (Micro)    |     SVM F1 (Macro)    |     SVM F1      ( weighted)    |     BERT Accuracy    |     BERT F1    |
|      (200 articles + 200)                                    |     81.2            |     81.2                  |     80.8              |     81.04                      |     83.8             |     85.1       |
|      (400 articles + 400)                                    |     85.2            |     85.23                 |     85.06             |     84.92                      |     84.5             |     89.6       |
|      (800 articles + 800)                                    |     88.8            |     88.8                  |     88.28             |     88.85                      |     91.6             |     93.5       |
|     1000 Generated                                           |     79.6            |     79.6                  |     78.6              |     79.79                      |     93.07            |     98.04      |
|      (1000 articles + 1000)                                  |     88.36           |     88.36                 |     87.76             |     88.4                       |     95.3             |     98.6       |
