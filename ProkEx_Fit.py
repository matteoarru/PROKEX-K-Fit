
# coding: utf-8

# # Prokex MOL

#%% In[1]:
import os
os.chdir('C:\\Users\\matte\\OneDrive\\Thesis Matteo\\prokexpy')
cwd = os.getcwd()
#print (cwd)
from prokex import ontology
from prokex import bpm
from prokex import testresults
import pandas as pd
import numpy as np
import csv
import codecs
#import pprint as pp


#cwd = os.getcwd()
#print (cwd)


# # Read Ontology from file
ontology = ontology.readontologyfromxlsx(cwd + "\\Contents.xlsx")

#%% In[2]: Remove Version from a Concept
def removeVersionConcept(conceptNameList):
    z=[]
    for x in conceptNameList: 
        z.append(x[:x.rfind("-")])
        
    return z

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

#%% In[2]: Stack a table based on given lists
def SaveStackTable(Xlist: list, Ylist:list, stackedTable, fileName: str):
    stackedTable=pd.DataFrame(data=stackedTable,columns=Ylist, index=Xlist)
    stackedTable.to_csv(fileName)
    return;


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
SaveStackTable(activities, roles, AR, cwd + "\\AR.csv")
#%% In[6]: Importing from ConceptGroup the list of Knowledge Nodes and merge with the final list coming from the Studio testing
conceptsxls = pd.read_excel(cwd + "\\ConceptGroups.xlsx",  sheet_name='CG', header=0) 
conceptsxls['Concept ID']=removeVersionConcept(conceptsxls['Concept ID'])
concepts = conceptsxls['Concept ID'].unique().tolist()
#Get Test Results from Studio
testtable = testresults.loadTestJson()
testtable['nodeId']=removeVersionConcept(testtable['nodeId'])
concepts =concepts+testtable['nodeId'].unique().tolist()
used = set()
concepts = sorted([x for x in concepts if x not in used and (used.add(x) or True)])
with codecs.open(cwd + "\\concepts.csv", 'w', "utf-8") as myfile:
    wr = csv.writer(myfile, lineterminator='\n')
    for val in concepts:
        wr.writerow([val]) 
#%% In[7]: Define Individuals from Studio Test.
individuals = sorted(testtable['userId'].unique().tolist())
RemoveIndividuals = pd.read_excel(cwd + "\\RemoveIndividuals.xlsx",  sheet_name='blacklisted', header=0)['Individual'].unique().tolist()
individuals = [x for x in individuals if x not in RemoveIndividuals]
with codecs.open(cwd + "\\individuals.csv", 'w', "utf-8") as myfile:
    wr = csv.writer(myfile, lineterminator='\n')
    for val in individuals:
        wr.writerow([val]) 

#%% In[8]: Calculating IT Individual Test matrix
testtable = testtable[~testtable['userId'].isin(RemoveIndividuals)]
#testtable['nodeId']=removeVersionConcept(testtable['nodeId'])
IT=stackTableonLists(individuals, concepts, testtable[['userId',  'nodeId']])
SaveStackTable(individuals, concepts, IT, cwd + "\\IT.csv")
#%% In[11]: Load Positions
jobDescr = pd.read_excel(cwd + "\\JobDescriptions.xlsx",  sheet_name='positions', header=0) 
positions = sorted(jobDescr['Position'].unique().tolist())
with codecs.open(cwd + "\\positions.csv", 'w', "utf-8") as myfile:
    wr = csv.writer(myfile, lineterminator='\n')
    for val in positions:
        wr.writerow([val]) 

#%% In[12]: RP: Role Positions
RP=stackTableonLists(roles, positions, jobDescr[['Role',  'Position']])
SaveStackTable(roles, positions, RP, cwd + "\\RP.csv")

#%% In[12]: IP: Individuals Positions
HRRecords = pd.read_excel(cwd + "\\HRRecords.xlsx",  sheet_name='individuals', header=0) 
IP=stackTableonLists(individuals, positions, HRRecords[['Individual',  'Position']])
SaveStackTable(individuals, positions, IP, cwd + "\\IP.csv")
IPT=IP.T.astype(float)
SaveStackTable(positions,individuals,  IPT, cwd + "\\IPT.csv")

#%% In[10]:Calculating RI Roles associated with Individuals
RI=np.dot(RP,IP.transpose())
SaveStackTable(roles, individuals, RI, cwd + "\\RI.csv")

#%% In[10]:Calculating AP Activities associated with Positions
AP=np.dot(AR,RP)
SaveStackTable(activities, positions, AP, cwd + "\\AP.csv")
#%% In[10]:Calculating AI Activities associated with Individuals
AI=np.dot(AP,IP.transpose())
SaveStackTable(activities, individuals, AI, cwd + "\\AI.csv")

#%% In[9]: Calculating AK Activity Knowledge matrix
AK=stackTableonLists(activities, concepts, conceptsxls[['Task',  'Concept ID']])
SaveStackTable(activities, concepts, AK, cwd + "\\AK.csv")

#%% In[10]:Calculating RK Knowledge required by Role
RK=np.dot(AR.transpose(),AK)
RK=np.logical_and(RK, np.ones(RK.shape))
SaveStackTable(roles, concepts, RK, cwd + "\\RK.csv")

#%% In[10]:Calculating PK Knowledge required in a Position
PK=np.dot(AP.transpose(),AK)
PK=np.logical_and(PK, np.ones(PK.shape))
SaveStackTable(positions, concepts, PK, cwd + "\\PK.csv")

#%% In[10]:Calculating IK Knowledge required for an Individual
IK=np.dot(AI.transpose(),AK)
IK=np.logical_and(IK, np.ones(IK.shape))
SaveStackTable(individuals, concepts, IK, cwd + "\\IK.csv")

#%% In[17]: Knowledge Fits


#%% In[17]: Fit Function
def Fit(knowledge: np.matrix, test: np.matrix)-> np.matrix:
    result=np.empty(knowledge.shape).astype(float)
    if knowledge.shape==test.shape:
        result=np.copy(test)
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
        result=np.copy(test)
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
    result=np.empty(mt.shape).astype(float)
    #print("MT: "+ str(mt.shape)+" v: "+ str(v.shape))
    if mt.shape[0]==mt.shape[0]:
        result=np.nan_to_num((mt.T.astype(float) / v.astype(float)).T)
    else:
        raise ValueError('DividePerSize Error: The number of rows of the matrix shall be equal to the columns of the vetor!')
    return result;   

#%% In[10]: Calculating IFit: Individual Knowledge Fit
IFit=np.logical_or(IT,np.logical_not(IK))
SaveStackTable(individuals, concepts, IT, cwd + "\\IT.csv")
SaveStackTable(individuals, concepts, IFit, cwd + "\\IFit.csv")
SaveStackTable(individuals, concepts, IT, cwd + "\\IT.csv")
ISpare=Spare(IK,IT)
SaveStackTable(individuals, concepts, IT, cwd + "\\IT.csv")
SaveStackTable(individuals, concepts, ISpare, cwd + "\\ISpare.csv")

#%% In[10]: Calculating PFit:  Knowledge Fit at the  position
PT=np.dot(IPT,IT)
SaveStackTable(positions, concepts, PT, cwd + "\\PT.csv")
PFit=Fit(PK,DividePerSize(np.dot(IP.T.astype(float),IT.astype(float)),ElementsByColumn(IP.astype(float))))
SaveStackTable(positions, concepts, PFit, cwd + "\\PFit.csv")
PSpare=Spare(PK,DividePerSize(np.dot(IP.T.astype(float),IT.astype(float)),ElementsByColumn(IP.astype(float))))
SaveStackTable(positions, concepts, PSpare, cwd + "\\PSpare.csv")

#%% In[10]: Calculating RFit:  Knowledge Fit at the  role
RFit=Fit(RK,DividePerSize(np.dot(RI.astype(float),IT.astype(float)),ElementsByColumn(RI.T.astype(float))))
SaveStackTable(roles, concepts, RFit, cwd + "\\RFit.csv")
RSpare=Spare(RK,DividePerSize(np.dot(RI.astype(float),IT.astype(float)),ElementsByColumn(RI.T.astype(float))))
SaveStackTable(roles, concepts, RSpare, cwd + "\\RSpare.csv")
#%% In[10]: Calculating AFit:  Knowledge Fit at the  Activity
AFit=Fit(AK,DividePerSize(np.dot(AI.astype(float),IT.astype(float)),ElementsByColumn(AI.T.astype(float))))
SaveStackTable(activities, concepts, AFit, cwd + "\\AFit.csv")
ASpare=Spare(AK,DividePerSize(np.dot(AI.astype(float),IT.astype(float)),ElementsByColumn(AI.T.astype(float))))
SaveStackTable(activities, concepts, ASpare, cwd + "\\ASpare.csv")
#%% In[17]: Score Function

def Score(knowledge: np.matrix, fit: np.matrix):
    
    if knowledge.shape==fit.shape:
        result=np.nan_to_num(np.divide(np.sum(np.multiply(knowledge, fit),axis=1),np.sum(knowledge,axis=1)))
    else:
        raise ValueError('Score function Error: Knowledge and Fit matrices should have the same size')
    return result;  
#%% In[17]: Individual Scores
IScore=Score(IK,IFit)

ISScore=Score(np.logical_not(IK),ISpare)
np.c_[IScore,ISScore]
SaveStackTable(individuals, ['#','Fit Score','Spare Score'], np.c_[np.sum(IK,axis=1),IScore, ISScore], cwd + "\\IScore.csv")
#%% In[17]: Position Scores
PScore=Score(PK,PFit)
PSScore=Score(np.logical_not(PK),PSpare)
SaveStackTable(positions, ['#','Fit Score','Spare Score'], np.c_[np.sum(PK,axis=1),PScore, PSScore], cwd + "\\PScore.csv")
#%% In[17]: Role Scores
RScore=Score(RK,RFit)
RSScore=Score(np.logical_not(RK),RSpare)
SaveStackTable(roles, ['#','Fit Score','Spare Score'], np.c_[np.sum(RK,axis=1),RScore, RSScore], cwd + "\\RScore.csv")
#%% In[17]: Activities Scores
AScore=Score(AK,AFit)
ASScore=Score(np.logical_not(AK),ASpare)
SaveStackTable(activities, ['#','Fit Score','Spare Score'], np.c_[np.sum(AK,axis=1),AScore, ASScore], cwd + "\\AScore.csv")
#%% In[17]: Score Function
 
def GScore(knowledge: np.matrix, fit: np.matrix):
    np.seterr(divide='ignore', invalid='ignore')
    if knowledge.shape==fit.shape:
        result=np.nan_to_num(np.mean(np.square(Score(knowledge, fit)),axis=0))
    else:
        raise ValueError('Score function Error: Knowledge and Fit matrices should have the same size')
    return result;   
GScore(IK,IFit)
#%% In[17]: Full Factorial
def nextMatrix(startmatrix: np.matrix):
    
    if   np.array_equal(startmatrix, startmatrix.astype(bool)):
        result =startmatrix.astype(bool)
        i=1
        for x in np.nditer(result, op_flags=['readwrite']):
            if i==1:
                if x==0:
                    x[...]=1
                    i=0
                else:
                    x[...]=0
                    i=1
    else:
        raise ValueError('nextMatrix function Error: Full factorial requires a Binary Matrix')
    return  result;   

#%% In[17]: Full Factorial
def TopolFactorial(startmatrix: np.matrix):
    
    if   np.array_equal(startmatrix, startmatrix.astype(bool)):
        result=np.zeros(startmatrix.shape).astype(bool)
        tensor=np.array([]).astype(bool)
        for i in range (0, (2**float(np.prod(startmatrix.shape))-1)):
            result=nextMatrix(result)
            np.append(tensor,result)
    else:
        raise ValueError('nextMatrix function Error: Full factorial requires a Binary Matrix')
    return  tensor;   
import time
startime=time.time()
RPT= nextMatrix(RP)
(time.time()-startime)*(2**float(np.prod(RP.shape))-1)/(1200*24*365)


#IPT=TopolFactorial(IP)
#IPT.shape

2**(float(np.prod(IP.shape)))-1


2**55



#%% In[17]: Full Factorial


def FitTab(knowledge: np.matrix, test: np.matrix)-> np.matrix:
    result=np.empty(knowledge.shape).astype(float)
    if knowledge.shape[1]==test.shape[1]:
        if np.array_equal(knowledge, knowledge.astype(bool)):
            result=DividePerSize(np.dot(test.astype(float),knowledge.T.astype(float)).T, ElementsByColumn(knowledge.T.astype(float))).T
        else:
            raise ValueError('FitTab function Error: FitTab requires a Binary Knowledge Matrix')
    else:
        raise ValueError('FitTab function Error: Knowledge and Tested matrices should have the same number of columns')
    return result;   


IPFitTab=FitTab(PK,IT)
SaveStackTable( individuals, positions, IPFitTab, cwd + "\\IPFitTab.csv")
PRFitTab=FitTab(RK,PT)
SaveStackTable( positions, roles, PRFitTab, cwd + "\\PRFitTab.csv")
#%% In[17]: Matcher - https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.optimize.linear_sum_assignment.html
# https://en.wikipedia.org/wiki/Hungarian_algorithm


#cost = np.array([[4, 1, 3], [2, 0, 5], [3, 2, 2]])
IPFitTabbar=pd.read_excel(cwd + "\\IPFitTabbar.xlsx", sheet_name="IPFitTabbar")
IPFitTab=np.array(IPFitTabbar)
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
#%% In[2]:
# Matcher - https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.optimize.linear_sum_assignment.html
# https://en.wikipedia.org/wiki/Hungarian_algorithm


#cost = np.array([[4, 1, 3], [2, 0, 5], [3, 2, 2]])
cost=np.ones(IPFitTab.shape)-IPFitTab
from scipy.optimize import linear_sum_assignment

row_ind, col_ind = linear_sum_assignment(cost)
col_ind
#%% In[2]:
IPbar=np.zeros((len(col_ind),len(col_ind)),dtype=bool)

IPbar[tuple(zip(*np.column_stack((row_ind,col_ind))))] = True
IPbar =pd.DataFrame(IPbar,index=IPFitTabbar.index.values,columns=IPFitTabbar.columns.values)
IPbar
#%% In[2]:
IPbar.to_excel(cwd + "\\IPbar.xlsx",'IPbar')