import os
import sys
import re
import matplotlib
import pandas as pd
import numpy as np
from os.path import splitext
import ipaddress as ip
import tldextract
from tqdm import tqdm
# import whois
import datetime
from urllib.parse import urlparse

df = pd.read_csv("data.csv")
#df=df.sample(frac=1)
df = df.sample(frac=1).reset_index(drop=True)[:10000]
df.head()

#2016's top most suspicious TLD and words
Suspicious_TLD=['zip','cricket','link','work','party','gq','kim','country','science','tk', 'cn', 'ru', 'fr', 'de', 'com.br', 'pt', 'gr', 'ro', 'com.ar', 'co.id', 'in', 'cz', 'it', 'pl', 'com.mx', 'mx', 'sch.id', 'by', 'cl', 'hu', 'vn', 'ir', 'mu', 'ly']
Suspicious_Domain=['luckytime.co.kr','mattfoll.eu.interia.pl','trafficholder.com','dl.baixaki.com.br','bembed.redtube.comr','tags.expo9.exponential.com','deepspacer.com','funad.co.kr','trafficconverter.biz']
#trend micro's top malicious domains 

# Method to count number of dots
def countdots(url):  
    return url.count('.')

# Method to count number of delimeters
def countdelim(url):
    count = 0
    delim=[';','_','?','=','&']
    for each in url:
        if each in delim:
            count = count + 1
    
    return count

# Is IP addr present as th hostname, let's validate
import ipaddress as ip #works only in python 3

def isip(uri):
    try:
        if ip.ip_address(uri):
            return 1
    except:
        return 0

#method to check the presence of hyphens
def isPresentHyphen(url):
    return url.count('-')

#method to check the presence of @
def isPresentAt(url):
    return url.count('@')

def countSubDomain(subdomain):
    if not subdomain:
        return 0
    else:
        return len(subdomain.split('.'))

def countQueries(query):
    if not query:
        return 0
    else:
        return len(query.split('&'))

def countSubDir(url):
    return url.count('/')

featureSet = pd.DataFrame(columns=('url','len of url',\
'no of subdir','no of subdomain','len of domain','no of queries',\
'presence of Suspicious_TLD', 'presence of at','label'))

from urllib.parse import urlparse
import tldextract
tlds = {}
def getFeatures(url, label): 
    result = []
    url = "http://" + url
    url = str(url)
    
    #add the url to feature set
    result.append(url)
    
    #parse the URL and extract the domain information
    path = urlparse(url)
    ext = tldextract.extract(url)
    
    #counting number of dots in subdomain    
#     result.append(countdots(ext.subdomain))
    
    #checking hyphen in domain   
#     result.append(isPresentHyphen(path.netloc))
    
    #length of URL    
    result.append(len(url))
    
    #checking presence of double slash    
#     result.append(isPresentDSlash(path.path))
    
    #Count number of subdir    
    result.append(countSubDir(path.path))
    
    #number of sub domain    
    result.append(countSubDomain(ext.subdomain))
    
    #length of domain name    
    result.append(len(path.netloc))
    
    #count number of queries    
    result.append(len(path.query))
    
    #Adding domain information
    
    #if IP address is being used as a URL     
#     result.append(isip(ext.domain))
    
    #presence of Suspicious_TLD
    result.append(1 if ext.suffix in Suspicious_TLD else 0)
    if ext.suffix not in tlds:
        tlds[ext.suffix] = 0
    tlds[ext.suffix] += label - 0.5
    
    #presence of suspicious domain
#     result.append(1 if '.'.join(ext[1:]) in Suspicious_Domain else 0 )
    
    #checking @ in the url    
    result.append(isPresentAt(path.netloc))
    
     
    '''
      
    #Get domain information by asking whois
    domain = '.'.join(ext[1:])
    w = whois.whois(domain)
    
    avg_month_time=365.2425/12.0
    
                  
    #calculate creation age in months
                  
    if w.creation_date == None or type(w.creation_date) is str :
        result.append(-1)
        #training_df['create_age(months)'] = -1
    else:
        if(type(w.creation_date) is list): 
            create_date=w.creation_date[-1]
        else:
            create_date=w.creation_date

        if(type(create_date) is datetime.datetime):
            today_date=datetime.datetime.now()
            create_age_in_mon=((today_date - create_date).days)/avg_month_time
            create_age_in_mon=round(create_age_in_mon)
            result.append(create_age_in_mon)
            #training_df['create_age(months)'] = create_age_in_mon
            
        else:
            result.append(-1)
            #training_df['create_age(months)'] = -1
    
    #calculate expiry age in months
                  
    if(w.expiration_date==None or type(w.expiration_date) is str):
        #training_df['expiry_age(months)'] = -1
        result.append(-1)
    else:
        if(type(w.expiration_date) is list):
            expiry_date=w.expiration_date[-1]
        else:
            expiry_date=w.expiration_date
        if(type(expiry_date) is datetime.datetime):
            today_date=datetime.datetime.now()
            expiry_age_in_mon=((expiry_date - today_date).days)/avg_month_time
            expiry_age_in_mon=round(expiry_age_in_mon)
            #training_df['expiry_age(months)'] = expiry_age_in_mon
            #### appending  in months Appended to the Vector
            result.append(expiry_age_in_mon)
        else:
            #training_df['expiry_age(months)'] = -1
            result.append(-1)#### expiry date error so append -1

    #find the age of last update
                  
    if(w.updated_date==None or type(w.updated_date) is str):
        #training_df['update_age(days)'] = -1
        result.append(-1)
    else:
        if(type(w.updated_date) is list):
            update_date=w.updated_date[-1]
        else:
            update_date=w.updated_date
        if(type(update_date) is datetime.datetime):
            today_date=datetime.datetime.now()
            update_age_in_days=((today_date - update_date).days)
            result.append(update_age_in_days)
            #training_df['update_age(days)'] = update_age_in_days #### appending updated age in days Appended to the Vector
        else:
            result.append(-1)
            #training_df['update_age(days)'] = -1
    
    #find the country who is hosting this domain
    if(w.country == None):
        #training_df['country'] = "None"
        result.append("None")
    else:
        #training_df['country'] = w.country
        result.append(w.country)
     ''' 
    
    #result.append(get_ext(path.path))
    result.append(str(label))
    return result
                  
    #Yay! finally done!  


pos_featureSet = pd.DataFrame(columns=('url','len of url',\
'no of subdir','no of subdomain','len of domain','no of queries',\
'presence of Suspicious_TLD', 'presence of at','label'))
j = 0
for i in tqdm(range(len(df))):
    features = getFeatures(df["url"].loc[i], df["label"].loc[i])    
    featureSet.loc[i] = features
    if df["label"].loc[i] == 1:
        pos_featureSet.loc[j] = features
        j += 1


featureSet.to_csv('url.csv')
pos_featureSet.to_csv('positive_url.csv')