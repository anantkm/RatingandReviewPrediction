#Assignment_3_COMP_9321
#Author: Anant Krishna Mahale
#zID: 5277610

import ast
import json
import pandas as pd
import sys
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import datetime as dt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from scipy.stats.stats import pearsonr
import time
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

#ref: this function was submitted as part of Assignement 1.
def repairMe (x, what_to_filter):
    charList = []    
    finalString = ''
    tempList = ast.literal_eval(x)
    for dictionary in tempList:
        if (len(dictionary[what_to_filter])>1):
            s = dictionary[what_to_filter]
            s = s.replace(',', '')
            charList.append(s)
    finalString = ', '.join(str(x) for x in(charList))
    return (finalString[:500])

#This function is used to encode the Date columns as described in the report. 
def filterMonth(x):
    imp_month=[5,6,7,11,12]
    if x in imp_month:
        return 1
    return 0

#function to predict the Revnue.
def predictRevenue ( df_raw_train, df_raw_test):

    reqColumns = ['movie_id','cast','crew','budget','genres','keywords','production_companies','production_countries','release_date','runtime','spoken_languages','revenue']
    columnList = list(df_raw_train.columns.values)
    for name in columnList:
        if name not in reqColumns:
            df_raw_train.drop(name, axis=1, inplace=True)
            df_raw_test.drop(name, axis=1, inplace=True)

    df_raw_train = df_raw_train.set_index('movie_id');
    df_raw_test = df_raw_test.set_index('movie_id');

    df_raw_train['release_date']=pd.to_datetime(df_raw_train['release_date'])
    df_raw_train['release_date']= pd.DatetimeIndex(df_raw_train['release_date']).month
    df_raw_train['release_date']=df_raw_train['release_date'].astype(str)
    df_raw_train['release_date']=df_raw_train['release_date'].astype(int)


    df_raw_test['release_date']=pd.to_datetime(df_raw_test['release_date'])
    df_raw_test['release_date']= pd.DatetimeIndex(df_raw_test['release_date']).month
    df_raw_test['release_date']=df_raw_test['release_date'].astype(str)
    df_raw_test['release_date']=df_raw_test['release_date'].astype(int)
            

    df_raw_train[df_raw_train['budget'].apply(lambda x: str(x).isdigit())]
    df_raw_train[df_raw_train['runtime'].apply(lambda x: str(x).isdigit())]
    df_raw_train[df_raw_train['revenue'].apply(lambda x: str(x).isdigit())]


    df_raw_test[df_raw_test['budget'].apply(lambda x: str(x).isdigit())]
    df_raw_test[df_raw_test['runtime'].apply(lambda x: str(x).isdigit())]
    df_raw_test[df_raw_test['revenue'].apply(lambda x: str(x).isdigit())];


    df_raw_train = df_raw_train[df_raw_train['revenue']>100000]
    df_raw_train = df_raw_train[df_raw_train['budget']>100000]

    to_repair = ['cast','crew','genres','keywords','production_companies','production_countries','spoken_languages']
    for item in to_repair:
        df_raw_train[item] = df_raw_train[item].apply(repairMe, what_to_filter ='name')
        df_raw_test[item] = df_raw_test[item].apply(repairMe, what_to_filter ='name') 

    df_raw_train['release_date'] = df_raw_train['release_date'].apply(filterMonth)
    df_raw_test['release_date'] = df_raw_test['release_date'].apply(filterMonth)

    cast_vect = CountVectorizer()
    crew_vect = CountVectorizer()
    genres_vect = CountVectorizer()
    keywords_vect = CountVectorizer()
    production_companies_vect = CountVectorizer()
    production_countries_vect = CountVectorizer()
    spoken_languages_vect = CountVectorizer()

    df_train_filtered = df_raw_train[['release_date','runtime','budget']].copy()
    df_test_filtered = df_raw_test[['release_date','runtime','budget']].copy()

    cast_vect_train = cast_vect.fit_transform(df_raw_train['cast'])
    crew_vect_train = crew_vect.fit_transform(df_raw_train['crew'])
    genres_vect_train = genres_vect.fit_transform(df_raw_train['genres'])
    keywords_vect_train = keywords_vect.fit_transform(df_raw_train['keywords'])
    production_companies_vect_train = production_companies_vect.fit_transform(df_raw_train['production_companies'])
    production_countries_vect_train = production_countries_vect.fit_transform(df_raw_train['production_countries'])
    spoken_languages_vect_train = spoken_languages_vect.fit_transform(df_raw_train['spoken_languages'])

    cast_vect_test = cast_vect.transform(df_raw_test['cast'])
    crew_vect_test = crew_vect.transform(df_raw_test['crew'])
    genres_vect_test = genres_vect.transform(df_raw_test['genres'])
    keywords_vect_test = keywords_vect.transform(df_raw_test['keywords'])
    production_companies_vect_test = production_companies_vect.transform(df_raw_test['production_companies'])
    production_countries_vect_test = production_countries_vect.transform(df_raw_test['production_countries'])
    spoken_languages_vect_test = spoken_languages_vect.transform(df_raw_test['spoken_languages'])

    cast_col_train = pd.DataFrame(cast_vect_train.toarray(), columns=cast_vect.get_feature_names(), index= df_raw_train.index)
    crew_col_train  = pd.DataFrame(crew_vect_train.toarray(), columns=crew_vect.get_feature_names(), index= df_raw_train.index)
    genres_col_train  = pd.DataFrame(genres_vect_train.toarray(), columns=genres_vect.get_feature_names(), index= df_raw_train.index)
    keywords_col_train  = pd.DataFrame(keywords_vect_train.toarray(), columns=keywords_vect.get_feature_names(), index= df_raw_train.index)
    production_companies_col_train  = pd.DataFrame(production_companies_vect_train.toarray(), columns=production_companies_vect.get_feature_names(), index= df_raw_train.index)
    production_countries_col_train  = pd.DataFrame(production_countries_vect_train.toarray(), columns=production_countries_vect.get_feature_names(), index= df_raw_train.index)
    spoken_languages_col_train  = pd.DataFrame(spoken_languages_vect_train.toarray(), columns=spoken_languages_vect.get_feature_names(), index= df_raw_train.index)

    cast_col_test = pd.DataFrame(cast_vect_test.toarray(), columns=cast_vect.get_feature_names(), index= df_raw_test.index)
    crew_col_test  = pd.DataFrame(crew_vect_test.toarray(), columns=crew_vect.get_feature_names(), index= df_raw_test.index)
    genres_col_test  = pd.DataFrame(genres_vect_test.toarray(), columns=genres_vect.get_feature_names(), index= df_raw_test.index)
    keywords_col_test  = pd.DataFrame(keywords_vect_test.toarray(), columns=keywords_vect.get_feature_names(), index= df_raw_test.index)
    production_companies_col_test  = pd.DataFrame(production_companies_vect_test.toarray(), columns=production_companies_vect.get_feature_names(), index= df_raw_test.index)
    production_countries_col_test  = pd.DataFrame(production_countries_vect_test.toarray(), columns=production_countries_vect.get_feature_names(), index= df_raw_test.index)
    spoken_languages_col_test  = pd.DataFrame(spoken_languages_vect_test.toarray(), columns=spoken_languages_vect.get_feature_names(), index= df_raw_test.index)

    df_train_filtered = df_train_filtered.merge(cast_col_train, left_index=True, right_index=True, how='left')
    df_train_filtered = df_train_filtered.merge(crew_col_train, left_index=True, right_index=True, how='left')
    df_train_filtered = df_train_filtered.merge(genres_col_train, left_index=True, right_index=True, how='left')
    df_train_filtered = df_train_filtered.merge(keywords_col_train, left_index=True, right_index=True, how='left')
    df_train_filtered = df_train_filtered.merge(production_companies_col_train, left_index=True, right_index=True, how='left')
    df_train_filtered = df_train_filtered.merge(production_countries_col_train, left_index=True, right_index=True, how='left')
    df_train_filtered = df_train_filtered.merge(spoken_languages_col_train, left_index=True, right_index=True, how='left')

    df_test_filtered = df_test_filtered.merge(cast_col_test, left_index=True, right_index=True, how='left')
    df_test_filtered = df_test_filtered.merge(crew_col_test, left_index=True, right_index=True, how='left')
    df_test_filtered = df_test_filtered.merge(genres_col_test, left_index=True, right_index=True, how='left')
    df_test_filtered = df_test_filtered.merge(keywords_col_test, left_index=True, right_index=True, how='left')
    df_test_filtered = df_test_filtered.merge(production_companies_col_test, left_index=True, right_index=True, how='left')
    df_test_filtered = df_test_filtered.merge(production_countries_col_test, left_index=True, right_index=True, how='left')
    df_test_filtered = df_test_filtered.merge(spoken_languages_col_test, left_index=True, right_index=True, how='left')

    # X = df_train_filtered
    # y = df_raw_train[['revenue']].copy()
    # X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.33, random_state = 0)

    X_train = df_train_filtered.copy()
    y_train = df_raw_train[['revenue']].copy()


    X_test = df_test_filtered.copy()
    y_test = df_raw_test[['revenue']].copy()

    lm = linear_model.LinearRegression()
    lm.fit(X_train,y_train)
    y_pred_lm = lm.predict(X_test)

    mse = mean_squared_error(y_test, y_pred_lm, squared=False)
    correlation = pearsonr(y_test.values.flatten(), y_pred_lm.flatten())
    
    outputData_summary=[['z5277610',"{:.2f}".format(mse),"{:.2f}".format(correlation[0])]]
    outputDataFrame_summary= pd.DataFrame(outputData_summary, columns = ['zid','MSR','correlation']) 
    df_pred = pd.DataFrame(y_pred_lm, index=X_test.index, columns=['predicted_revenue'])

    print('Writing the Part 1 output to current Directory...')
    outputDataFrame_summary.to_csv(r'z5277610.PART1.summary.csv',index=False)
    df_pred.to_csv(r'z5277610.PART1.output.csv')

#function to classify the ratings. 
def classifyRating( df_raw_train, df_raw_test):
    
    reqColumns = ['movie_id','cast','crew','budget','genres','homepage','keywords','original_language','original_title','overview','production_companies','production_countries','release_date','runtime','spoken_languages','tagline','rating']
    columnList = list(df_raw_train.columns.values)
    for name in columnList:
        if name not in reqColumns:
            df_raw_train.drop(name, axis=1, inplace=True)
            df_raw_test.drop(name, axis=1, inplace=True)

    df_raw_train['release_date']=pd.to_datetime(df_raw_train['release_date'])
    df_raw_train['release_date']= pd.DatetimeIndex(df_raw_train['release_date']).month
    df_raw_train['release_date']=df_raw_train['release_date'].astype(str)
    df_raw_train['release_date']=df_raw_train['release_date'].astype(int)

    df_raw_test['release_date']=pd.to_datetime(df_raw_test['release_date'])
    df_raw_test['release_date']= pd.DatetimeIndex(df_raw_test['release_date']).month
    df_raw_test['release_date']=df_raw_test['release_date'].astype(str)
    df_raw_test['release_date']=df_raw_test['release_date'].astype(int)

    df_raw_train = df_raw_train.set_index('movie_id')
    df_raw_test = df_raw_test.set_index('movie_id')

    df_raw_train[df_raw_train['budget'].apply(lambda x: str(x).isdigit())]
    df_raw_train[df_raw_train['runtime'].apply(lambda x: str(x).isdigit())]
    df_raw_train[df_raw_train['rating'].apply(lambda x: str(x).isdigit())]
    df_raw_train = df_raw_train[df_raw_train.rating != 1]
    df_raw_train = df_raw_train[df_raw_train['budget']>100000]


    df_raw_test[df_raw_test['budget'].apply(lambda x: str(x).isdigit())]
    df_raw_test[df_raw_test['runtime'].apply(lambda x: str(x).isdigit())]
    df_raw_test[df_raw_test['rating'].apply(lambda x: str(x).isdigit())]

    to_repair = ['cast','crew','genres','keywords','production_companies','production_countries','spoken_languages']
    for item in to_repair:
        df_raw_train[item] = df_raw_train[item].apply(repairMe, what_to_filter ='name')
        df_raw_test[item] = df_raw_test[item].apply(repairMe, what_to_filter ='name') 


    df_raw_train['release_date'] = df_raw_train['release_date'].apply(filterMonth)
    df_raw_test['release_date'] = df_raw_test['release_date'].apply(filterMonth)


    #df_raw_train = shuffle(df_raw_train)
    df_raw_train = df_raw_train.sort_values(by=['rating'], ascending=True)
    df_raw_train = df_raw_train[:1470]

    
    cast_vect = CountVectorizer()
    crew_vect = CountVectorizer()
    genres_vect = CountVectorizer()
    keywords_vect = CountVectorizer()
    production_companies_vect = CountVectorizer()
    production_countries_vect = CountVectorizer()
    spoken_languages_vect = CountVectorizer()

    df_train_filtered = df_raw_train[['release_date','runtime','budget']].copy()
    df_test_filtered = df_raw_test[['release_date','runtime','budget']].copy()

    cast_vect_train = cast_vect.fit_transform(df_raw_train['cast'])
    crew_vect_train = crew_vect.fit_transform(df_raw_train['crew'])
    genres_vect_train = genres_vect.fit_transform(df_raw_train['genres'])
    keywords_vect_train = keywords_vect.fit_transform(df_raw_train['keywords'])
    production_companies_vect_train = production_companies_vect.fit_transform(df_raw_train['production_companies'])
    production_countries_vect_train = production_countries_vect.fit_transform(df_raw_train['production_countries'])
    spoken_languages_vect_train = spoken_languages_vect.fit_transform(df_raw_train['spoken_languages'])

    cast_vect_test = cast_vect.transform(df_raw_test['cast'])
    crew_vect_test = crew_vect.transform(df_raw_test['crew'])
    genres_vect_test = genres_vect.transform(df_raw_test['genres'])
    keywords_vect_test = keywords_vect.transform(df_raw_test['keywords'])
    production_companies_vect_test = production_companies_vect.transform(df_raw_test['production_companies'])
    production_countries_vect_test = production_countries_vect.transform(df_raw_test['production_countries'])
    spoken_languages_vect_test = spoken_languages_vect.transform(df_raw_test['spoken_languages'])

    cast_col_train = pd.DataFrame(cast_vect_train.toarray(), columns=cast_vect.get_feature_names(), index= df_raw_train.index)
    crew_col_train  = pd.DataFrame(crew_vect_train.toarray(), columns=crew_vect.get_feature_names(), index= df_raw_train.index)
    genres_col_train  = pd.DataFrame(genres_vect_train.toarray(), columns=genres_vect.get_feature_names(), index= df_raw_train.index)
    keywords_col_train  = pd.DataFrame(keywords_vect_train.toarray(), columns=keywords_vect.get_feature_names(), index= df_raw_train.index)
    production_companies_col_train  = pd.DataFrame(production_companies_vect_train.toarray(), columns=production_companies_vect.get_feature_names(), index= df_raw_train.index)
    production_countries_col_train  = pd.DataFrame(production_countries_vect_train.toarray(), columns=production_countries_vect.get_feature_names(), index= df_raw_train.index)
    spoken_languages_col_train  = pd.DataFrame(spoken_languages_vect_train.toarray(), columns=spoken_languages_vect.get_feature_names(), index= df_raw_train.index)

    cast_col_test = pd.DataFrame(cast_vect_test.toarray(), columns=cast_vect.get_feature_names(), index= df_raw_test.index)
    crew_col_test  = pd.DataFrame(crew_vect_test.toarray(), columns=crew_vect.get_feature_names(), index= df_raw_test.index)
    genres_col_test  = pd.DataFrame(genres_vect_test.toarray(), columns=genres_vect.get_feature_names(), index= df_raw_test.index)
    keywords_col_test  = pd.DataFrame(keywords_vect_test.toarray(), columns=keywords_vect.get_feature_names(), index= df_raw_test.index)
    production_companies_col_test  = pd.DataFrame(production_companies_vect_test.toarray(), columns=production_companies_vect.get_feature_names(), index= df_raw_test.index)
    production_countries_col_test  = pd.DataFrame(production_countries_vect_test.toarray(), columns=production_countries_vect.get_feature_names(), index= df_raw_test.index)
    spoken_languages_col_test  = pd.DataFrame(spoken_languages_vect_test.toarray(), columns=spoken_languages_vect.get_feature_names(), index= df_raw_test.index)

    df_train_filtered = df_train_filtered.merge(cast_col_train, left_index=True, right_index=True, how='left')
    df_train_filtered = df_train_filtered.merge(crew_col_train, left_index=True, right_index=True, how='left')
    df_train_filtered = df_train_filtered.merge(genres_col_train, left_index=True, right_index=True, how='left')
    df_train_filtered = df_train_filtered.merge(keywords_col_train, left_index=True, right_index=True, how='left')
    df_train_filtered = df_train_filtered.merge(production_companies_col_train, left_index=True, right_index=True, how='left')
    df_train_filtered = df_train_filtered.merge(production_countries_col_train, left_index=True, right_index=True, how='left')
    df_train_filtered = df_train_filtered.merge(spoken_languages_col_train, left_index=True, right_index=True, how='left')

    df_test_filtered = df_test_filtered.merge(cast_col_test, left_index=True, right_index=True, how='left')
    df_test_filtered = df_test_filtered.merge(crew_col_test, left_index=True, right_index=True, how='left')
    df_test_filtered = df_test_filtered.merge(genres_col_test, left_index=True, right_index=True, how='left')
    df_test_filtered = df_test_filtered.merge(keywords_col_test, left_index=True, right_index=True, how='left')
    df_test_filtered = df_test_filtered.merge(production_companies_col_test, left_index=True, right_index=True, how='left')
    df_test_filtered = df_test_filtered.merge(production_countries_col_test, left_index=True, right_index=True, how='left')
    df_test_filtered = df_test_filtered.merge(spoken_languages_col_test, left_index=True, right_index=True, how='left')


    X_train = df_train_filtered.copy()
    y_train = df_raw_train[['rating']].copy()
    X_test = df_test_filtered.copy()
    y_test = df_raw_test[['rating']].copy() 

    # X = df_train_filtered
    # y = df_raw_train[['rating']].copy()
    # X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.30, random_state = 0)
 

    rfc = RandomForestClassifier(random_state=42)
    rfc.fit(X_train, y_train.values.ravel())
    y_pred_rfc = rfc.predict(X_test)           
    report = classification_report(y_test, y_pred_rfc, output_dict=True )
    accuracy = report['accuracy']
    macro_precision =  report['macro avg']['precision'] 
    macro_recall = report['macro avg']['recall']  
    outputData_summary=[['z5277610',"{:.2f}".format(macro_precision),"{:.2f}".format(macro_recall),"{:.2f}".format(accuracy)]]
    outputDataFrame_summary= pd.DataFrame(outputData_summary, columns = ['zid','average_precision','average_recall','accuracy']) 
    df_pred = pd.DataFrame(y_pred_rfc, index=X_test.index, columns=['predicted_rating'])
    print('Writing the Part 2 output to current Directory...')
    outputDataFrame_summary.to_csv(r'z5277610.PART2.summary.csv',index=False)
    df_pred.to_csv(r'z5277610.PART2.output.csv')


if __name__ == '__main__':
    
    if len(sys.argv) < 3 or len(sys.argv) > 3:
        print('Enter Valid Input')
        sys.exit(0)

    training_data = sys.argv[1]  #path for traning data.
    validation_data = sys.argv[2]    #path for testing data. 
    try:
        #reading the movie file
        df_raw_train = pd.read_csv(training_data)
        df_raw_test = pd.read_csv(validation_data)
    except:
        print('Please input valid .csv File')
        sys.exit(0)

    #reading the movie file
    # df_raw_train = pd.read_csv('training.csv')
    # df_raw_test = pd.read_csv('validation.csv')

    #making the copy of Dataframe for Regression 
    df_raw_train_reg = df_raw_train.copy()
    df_raw_test_reg = df_raw_test.copy()

    #making the copy of Dataframe for Classificaiton. 
    df_raw_train_clf = df_raw_train.copy()
    df_raw_test_clf = df_raw_test.copy()


    print('Running Regression Algorithm...')
    start_time_reg = time.time()    
    predictRevenue(df_raw_train_reg,df_raw_test_reg)
    print("Regression Duration: {:.2f} seconds".format(time.time() - start_time_reg) )

    print('Running Classifier Algorithm...')
    start_time_clf = time.time()    
    classifyRating(df_raw_train_clf,df_raw_test_clf)
    print("Classifier Duration: {:.2f} seconds".format(time.time() - start_time_clf) )
    print("Overall duration: {:.2f} seconds".format(time.time() - start_time_reg))