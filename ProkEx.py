
# coding: utf-8

# # Prokex MOL

#%% In[1]:
import os
os.chdir('C:\\Users\\matte\\OneDrive\\Thesis Matteo\\prokexpy')
cwd = os.getcwd()
#print (cwd)
from prokex import ontology
from prokex import bpm
from prokex import getnames
from prokex import preprocessing
from prokex import testresults
from prokex import getnames
import pandas as pd
import nltk
import numpy as np
import string 
import operator
import re
import csv
import codecs
#import pprint as pp


#cwd = os.getcwd()
#print (cwd)


# # Read Ontology from file
ontology = ontology.readontologyfromxlsx(cwd + "\\Contents.xlsx")


# # Load Tasks & Task Descriptions
#%% In[2]: Stack a table based on given lists
def stackTableonLists(Xlist: list, Ylist:list, unstackedTable: pd)-> pd:
    y=[]
    for x in unstackedTable[unstackedTable.columns[0]]:
        y.append(Xlist.index(x))
    z=[]
    for x in unstackedTable[unstackedTable.columns[1]]:
        z.append(Ylist.index(x))
    ST=np.zeros((len(Xlist),len(Ylist)),dtype=bool)
    ST[tuple(zip(*np.column_stack((y,z))))] = True
    return ST;   


#%% In[3]: Download Activities from BPM
#from lxml import objectify
#from pandas import DataFrame
df = bpm.readbpmfromxlml (cwd + "\\Model.xml")
## Define Activities
activities = sorted(df['Task'].unique().tolist())

with open(cwd + "\\activities.csv", 'w') as myfile:
    wr = csv.writer(myfile, lineterminator='\n')
    for val in activities:
        wr.writerow([val]) 
  
#%% In[4]: Download Roles from BPM
## Define Roles
roles = sorted(df['Role'].unique().tolist())
with open(cwd + "\\roles.csv", 'w') as myfile:
    wr = csv.writer(myfile, lineterminator='\n')
    for val in roles:
        wr.writerow([val]) 
        
#%% In[5]: Importing from BPM AR Activity Roles matrix
AR=stackTableonLists(activities, roles, df[['Task',  'Role']])
np.savetxt(cwd + "\\AR.csv", AR, delimiter=",")

#%% In[6]: Importing from ConceptGroup the list of Knowledge Nodes and merge with the final list coming from the Studio testing
conceptsxls = pd.read_excel(cwd + "\\ConceptGroups.xlsx",  sheet_name='CG', header=0) 
concepts = conceptsxls['Concept ID'].unique().tolist()
#Get Test Results from Studio
testtable = testresults.loadTestJson()
concepts = concepts+testtable['nodeId'].unique().tolist()
used = set()
concepts = sorted([x for x in concepts if x not in used and (used.add(x) or True)])
with codecs.open(cwd + "\\concepts.csv", 'w', "utf-8") as myfile:
    wr = csv.writer(myfile, lineterminator='\n')
    for val in concepts:
        wr.writerow([val]) 
#%% In[7]: Define Individuals from Studio Test.
individuals = sorted(testtable['userId'].unique().tolist())
with codecs.open(cwd + "\\individuals.csv", 'w', "utf-8") as myfile:
    wr = csv.writer(myfile, lineterminator='\n')
    for val in individuals:
        wr.writerow([val]) 

#%% In[8]: Calculating IT Individual Test matrix
IT=stackTableonLists(individuals, concepts, testtable[['userId',  'nodeId']])
np.savetxt(cwd + "\\IT.csv", IT, delimiter=",")

#%% In[11]: Load Positions
jobDescr = pd.read_excel(cwd + "\\JobDescriptions.xlsx",  sheet_name='positions', header=0) 
positions = sorted(jobDescr['Position'].unique().tolist())
with codecs.open(cwd + "\\positions.csv", 'w', "utf-8") as myfile:
    wr = csv.writer(myfile, lineterminator='\n')
    for val in positions:
        wr.writerow([val]) 

#%% In[12]: RP: Role Positions
RP=stackTableonLists(roles, positions, jobDescr[['Role',  'Position']])
np.savetxt(cwd + "\\RP.csv", RP, delimiter=",")

#%% In[12]: IP: Individuals Positions
HRRecords = pd.read_excel(cwd + "\\HRRecords.xlsx",  sheet_name='individuals', header=0) 
IP=stackTableonLists(individuals, positions, HRRecords[['Individual',  'Position']])
np.savetxt(cwd + "\\IP.csv", IP, delimiter=",")

#%% In[10]:Calculating RI Roles associated with Individuals
RI=np.dot(RP,IP.transpose())
np.savetxt(cwd + "\\RI.csv", RI, delimiter=",")
#%% In[10]:Calculating AP Activities associated with Positions
AP=np.dot(AR,RP)
np.savetxt(cwd + "\\AP.csv", AP, delimiter=",")
#%% In[10]:Calculating AI Activities associated with Individuals
AI=np.dot(AP,IP.transpose())
np.savetxt(cwd + "\\AI.csv", AI, delimiter=",")
#%% In[9]: Calculating AK Activity Knowledge matrix
AK=stackTableonLists(activities, concepts, conceptsxls[['Task',  'Concept ID']])
np.savetxt(cwd + "\\AK.csv", AK, delimiter=",")

#%% In[10]:Calculating RK Knowledge required by Role
RK=np.dot(AR.transpose(),AK)
RK=np.logical_and(RK, np.ones(RK.shape))
np.savetxt(cwd + "\\RK.csv", RK, delimiter=",")

#%% In[10]:Calculating PK Knowledge required in a Position
PK=np.dot(AP.transpose(),AK)
PK=np.logical_and(PK, np.ones(PK.shape))
np.savetxt(cwd + "\\PK.csv", PK, delimiter=",")
#%% In[10]:Calculating IK Knowledge required for an Individual
IK=np.dot(AI.transpose(),AK)
IK=np.logical_and(IK, np.ones(IK.shape))
np.savetxt(cwd + "\\IK.csv", IK, delimiter=",")

#%% In[17]: Knowledge Fits


#%% In[17]: Fit Function
def Fit(knowledge: np.matrix, test: np.matrix)-> np.matrix:
    result=np.empty(knowledge.shape)
    if knowledge.shape==test.shape:
        result=test
        for x in range (test.shape[0]):
            for y in range (test.shape[1]):              
                if knowledge.item((x, y))== 0:
                    #print ("x=" + str(x)+" y=" + str(y)+ " Val="+str(knowledge.item((x, y))))
                    result[x, y]=1
    else:
        raise ValueError('Fit function Error: Knowledge and Tested matrices should have the same size')
    return result;   
#%% In[17]: Spare Function

def Spare(knowledge: np.matrix, test: np.matrix)-> np.matrix:
    result=np.empty(knowledge.shape)
    if knowledge.shape==test.shape:
        result=test
        for x in range (test.shape[0]):
            for y in range (test.shape[1]):              
                if knowledge.item((x, y))== 1:
                    #print ("x=" + str(x)+" y=" + str(y)+ " Val="+str(knowledge.item((x, y))))
                    result[x, y]=0
    else:
        raise ValueError('Spare function Error: Knowledge and Tested matrices should have the same size')
    return result;   
#%% In[17]: ElementsByColumn
def ElementsByColumn(mt: np.matrix)-> np.matrix:
    return mt.sum(axis=0);   

#%% In[17]: DividePerSize
def DividePerSize(mt: np.matrix, v: np.matrix)-> np.matrix:
    result=np.empty(mt.shape)
    #print("MT: "+ str(mt.shape)+" v: "+ str(v.shape))
    if mt.shape[0]==mt.shape[0]:
        result=np.nan_to_num((mt.T / v).T)
    else:
        raise ValueError('DividePerSize Error: The number of rows of the matrix shall be equal to the columns of the vetor!')
    return result;   

#%% In[10]: Calculating IFit: Individual Knowledge Fit
IFit=np.logical_or(IT,np.logical_not(IK))
np.savetxt(cwd + "\\IFit.csv", IFit, delimiter=",")
IFit.shape


#%% In[10]: Calculating PFit:  Knowledge Fit at the  position
PFit=Fit(PK,DividePerSize(np.dot(IP.T,IT),ElementsByColumn(IP)))
np.savetxt(cwd + "\\PFit.csv", PFit, delimiter=",")
PFit.shape

#%% In[10]: Calculating RFit:  Knowledge Fit at the  role
RFit=Fit(RK,DividePerSize(np.dot(RI,IT),ElementsByColumn(RI.T)))
np.savetxt(cwd + "\\RFit.csv", RFit, delimiter=",")
RFit.shape
#%% In[17]: Score Function

def Score(knowledge: np.matrix, fit: np.matrix):
    
    if knowledge.shape==fit.shape:
        result=np.nan_to_num(np.divide(np.sum(np.multiply(knowledge, fit),axis=1),np.sum(knowledge,axis=1)))
    else:
        raise ValueError('Score function Error: Knowledge and Fit matrices should have the same size')
    return result;   
Score(IK,IFit)
#%% In[17]: Score Function

def GScore(knowledge: np.matrix, fit: np.matrix):
    np.seterr(divide='ignore', invalid='ignore')
    if knowledge.shape==fit.shape:
        result=np.nan_to_num(np.mean(np.square(Score(knowledge, fit)),axis=0))
    else:
        raise ValueError('Score function Error: Knowledge and Fit matrices should have the same size')
    return result;   
GScore(IK,IFit)

#%% In[18]:
abbreviations = preprocessing.readabbreviationsfromcsv (cwd + "\\abbreviations.csv")
stopWords = preprocessing.readstopwordsfromcsv (cwd + "\\stoplist.csv")
stopWords_2 = set(nltk.corpus.stopwords.words('English'))

#print(stopwords_2)
top_fraction = 1 # consider top third candidate keywords by score

# Create a list to store the data
ScoredKeyphrases = []
i = 0

for Description in df['Description']:
    text = Description
   
    
    #These two lines of code is to flat text including bullets
    fileContent = re.sub("  +", ".", text)
    fileContentAscii = ''.join(filter(lambda x:x in string.printable,fileContent))
    #Replace two full stops with one full stop
    fileContentAscii = re.sub(r'\.+', ".", fileContentAscii)
    
    
    #Instead of  sentences = nltk.sent_tokenize(fileContentAscii) add these lines of code,so that not spleat lines including any abbreviation 
    from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
    punkt_param = PunktParameters()
    #abbreviation = ['Mr' , 'etc', 'fig','e.g']
    abbreviation = abbreviations
    punkt_param.abbrev_types = set(abbreviation)
    tokenizer = PunktSentenceTokenizer(punkt_param)
    sentences = tokenizer.tokenize(fileContentAscii)
#    phrase_list = preprocessing.generate_candidate_keywords(sentences,stopWords)
    phrase_list = getnames.generate_candidate_keywords_v3(sentences, stopWords)
    word_scores = preprocessing.calculate_word_scores(phrase_list)
    phrase_scores = preprocessing.calculate_phrase_scores(phrase_list, word_scores)
    sorted_phrase_scores = sorted(phrase_scores.items(),key=operator.itemgetter(1), reverse=True)
    n_phrases = len(sorted_phrase_scores)
    ScoredKeyphrases.append(sorted_phrase_scores[0:int(n_phrases/top_fraction)])



df['ScoredKeyphrases'] = ScoredKeyphrases  
df.head()
df.to_csv(cwd+'\\keywords.csv', index=False, encoding='utf-8')


# # Explode Scored Keyphrases

#%% In[20]:


initialList = pd.DataFrame(columns=('Task', 'Phrase', 'Score' ,  'Description'))
i = 0

for idx in df.index:
    for item in df.at[idx,'ScoredKeyphrases']:
        Phrase, Score = item
        initialList.loc[i] = [df.at[idx,'Task'],Phrase, Score,  df.at[idx,'Description']]
        i += 1
#initialList.head()


# # Export Initial List to CSV File

#%% In[21]:


filename = cwd + '\\ScoredKeywordsV2.csv'
initialList.to_csv(filename, index=False, encoding='utf-8')


# # TfidfVectorizer

#%% In[22]:


from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(initialList['Phrase'])
#shape of tf-idf matrix
print (tfidf_matrix.shape)


#%% In[23]:


print (tfidf_matrix)


#%% In[24]:


# vocabulary list
tfidf_vectorizer.get_feature_names()


# # NearestNeighbors

#%% In[25]:


from sklearn.neighbors import NearestNeighbors
model_tf_idf = NearestNeighbors(algorithm='brute', metric='cosine')
model_tf_idf.fit(tfidf_matrix)

print (model_tf_idf)
#%% In[26]:



#!pip install html2text
#!pip install urllib2
#import urllib2
#import html2text
#ontology = pd.read_csv (cwd + "\\lmlist.csv",  encoding='utf-8', sep=';', header=0, index_col=False)
#ontology.columns = ['id','curriculum','nameEn','hasLearningMaterialHU','hasLearningMaterialEN', 'nulla']
#ontology['Concept'] = ontology['nameEn'].replace('_',' ').str.lower()
#ontology['Concept'] = ontology['Concept'].str.lower()
#ontology = ontology[pd.notnull(ontology['Concept'])]


#print (ontology[['nameEn', 'Concept']].head)


# # Load Ontology

#%% In[27]:


#ontology = pd.read_excel(cwd + "\\Contents.xlsx",  sheet_name='Data', header=0)
#ontology.columns = ['ConceptID', 'Concept']
#ontology['Concept'] = ontology['nameEn'].str.replace('_',' ').str.lower()
#print (ontology['nameEn'])
#ontology.head()
#ontology = ontology[pd.notnull(ontology['Concept'])]
# Remove bad Concepts
#those with false description
#print ('Out of ' , len(ontology), ' concepts there are ', len(ontology[ontology['Concept']=="false"]), 'ones with false Description.\n')
#ontology = ontology[ontology['Concept']!="false"]
# now those starting with question

#print ('Out of ' , len(ontology), ' concepts there are ', len(ontology[ontology['Concept'].str[:8]=="question"]), 'that are Questions.\n')

#ontology = ontology[ontology['Concept'].str[:8]=="question"]
#%% In[28]:


new_term_freq_matrix = tfidf_vectorizer.transform(ontology['Concept'].apply(str))
print(new_term_freq_matrix)
print(model_tf_idf)
print(ontology[2394:])
print(initialList[76:77])
#%% In[29]:


df_matching = pd.DataFrame(columns=('Task' , 'Phrase', 'Concept' , 'Concept ID' , 'Distance', 'Word Distance'))

import difflib
for index, row in ontology.iterrows():
    strText =  row['Concept']
     #Run the following cell to obtain the index for the first article:
    searched_idx = ontology[ontology['Concept'] == strText].index[0]
    distances, indices = model_tf_idf.kneighbors(new_term_freq_matrix[searched_idx], n_neighbors=1) #returns the first 3 neighbors
    nearest_indices = indices.flatten()
    nearest_index = nearest_indices[0]
    nearest_distances = distances.flatten()
    nearest_distance = nearest_distances[0]
    df_matching.loc[index] = [initialList.loc[nearest_index,'Task'], initialList.loc[nearest_index,'Phrase'],ontology.loc[searched_idx,'Concept'],ontology.loc[searched_idx,'id'], nearest_distance, difflib.SequenceMatcher(None," ".join(initialList.loc[nearest_index,'Phrase'])," ".join(ontology.loc[searched_idx,'Concept'])).ratio()*100]


#%% In[ ]:

df_matching=df_matching.sort_values(by=['Task' , 'Phrase', 'Concept', 'Distance', 'Word Distance'],ascending=[1, 1, 1, 1, 0]).reset_index(drop=True)

#%% In[ ]:


filename = cwd + '\\MatchingLNN.csv'
df_matching.to_csv(filename, index=False, encoding='utf-8')

