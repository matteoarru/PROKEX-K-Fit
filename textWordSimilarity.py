# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 09:50:42 2018

@author: matte
"""

text1=["poor", "management"]
text2=["good", "management"]
text3=["people", "management"]
text4=["management"]
import difflib
seq = difflib.SequenceMatcher(None," ".join(text1)," ".join(text2)).ratio()*100
print (seq)
seq = difflib.SequenceMatcher(None," ".join(text1)," ".join(text3)).ratio()*100
print (seq)
seq = difflib.SequenceMatcher(None," ".join(text1)," ".join(text4)).ratio()*100
print (seq)
seq = difflib.SequenceMatcher(None," ".join(text1)," ".join(text1)).ratio()*100
print (seq)