# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 15:33:13 2018

@author: matte
"""
from prokex import getnames
from prokex import preprocessing
from nltk.corpus import stopwords
stopwords = stopwords.words('english')
killwords =['notification', 'line', 'havaria', 'minimal number']
text1 = "Notification of the work package shall be issued prior to execution for each maintenance work item (excluding havaria) generated in the Computerized Maintenance Management System, hereinafter: CMMS."
text2 = "On MOL Group level preferable CMMS is standard SAP solution. Equipment database in CMMS has to be fulfilled and updated by dedicated responsible person. Processes in CMMS have to be simplified enough, but still in line with MOL Group LDA, with minimal number of paperwork."
  
sentences=[]
sentences.append(text1)
sentences.append(text2)


#phrase_list = getnames.generate_candidate_keywords_v3(sentences, stopwords)
phrase_list = preprocessing.generate_candidate_keywords(sentences,stopwords)
phrase_list2 = getnames.generate_from_text(text1, stopwords)
phrase_list3 = getnames.generate_candidate_keywords_v3(sentences,killwords)
