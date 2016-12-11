import utils
import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn import preprocessing
from collections import defaultdict
def read_csv(filepath):
    
    '''
    Read the events.csv, mortality_events.csv and event_feature_map.csv files into events, mortality and feature_map.
    
    Return events, mortality and feature_map
    '''

    #Columns in events.csv - patient_id,event_id,event_description,timestamp,value
    events_df = pd.read_csv(filepath + 'events.csv', parse_dates=['timestamp'])
     
    #Columns in mortality_event.csv - patient_id,timestamp,label
    mortality_df = pd.read_csv(filepath + 'mortality_events.csv', parse_dates=['timestamp'])
 
    #Columns in event_feature_map.csv - idx,event_id
    feature_map_df = pd.read_csv(filepath + 'event_feature_map.csv')
 

    return events_df, mortality_df, feature_map_df


def calculate_index_date(events, mortality, deliverables_path):
    
    '''

    Refer to instructions in Q3 a

    Suggested steps:
    1. Create list of patients alive ( mortality_events.csv only contains information about patients deceased)
    2. Split events into two groups based on whether the patient is alive or deceased
    3. Calculate index date for each patient
    
    IMPORTANT:
    Save indx_date to a csv file in the deliverables folder named as etl_index_dates.csv. 
    Use the global variable deliverables_path while specifying the filepath. 
    Each row is of the form patient_id, indx_date.
    The csv file should have a header 
    For example if you are using Pandas, you could write: 
        indx_date.to_csv(deliverables_path + 'etl_index_dates.csv', columns=['patient_id', 'indx_date'], index=False)

    Return indx_date
    '''
    events['count']=1

    results1=events[["patient_id","count"]].groupby("patient_id").sum()
    total_patient=len(results1.index)
    

    results2=events[["patient_id","timestamp","count"]].groupby(["patient_id","timestamp"]).sum()

    datelist,idlist=[],[]
    for i in range(total_patient):
        id1=results1.index[i]
        idlist.append(id1)
        if id1 in mortality['patient_id'].tolist():
            mortality.index=mortality['patient_id']
            datelist.append(utils.date_offset(mortality.loc[id1]['timestamp'],-30))
        else:
            maxtime=utils.date_convert(results2.loc[id1].index[0])
            for j in range(len(results2.loc[id1].index)):
                nowtime=utils.date_convert(results2.loc[id1].index[j])
                if nowtime>maxtime:
                    maxtime=nowtime
            datelist.append(maxtime)

    
    indx_date = pd.DataFrame(np.array([idlist,datelist]).T,columns=['patient_id','indx_date'])
    indx_date.to_csv(deliverables_path + 'etl_index_dates.csv', columns=['patient_id', 'indx_date'], index=False)
   
    return indx_date


def filter_events(events, indx_date, deliverables_path):
    
    '''

    Refer to instructions in Q3 a

    Suggested steps:
    1. Join indx_date with events on patient_id
    2. Filter events occuring in the observation window(IndexDate-2000 to IndexDate)
    
    
    IMPORTANT:
    Save filtered_events to a csv file in the deliverables folder named as etl_filtered_events.csv. 
    Use the global variable deliverables_path while specifying the filepath. 
    Each row is of the form patient_id, event_id, value.
    The csv file should have a header 
    For example if you are using Pandas, you could write: 
        filtered_events.to_csv(deliverables_path + 'etl_filtered_events.csv', columns=['patient_id', 'event_id', 'value'], index=False)

    Return filtered_events
    '''
    joined=pd.merge(events,indx_date,how="left",on=['patient_id'])
    joined=joined.dropna(axis=0)
    joined['mindate']=joined['indx_date']-timedelta(days=2000)
    convTS=joined['timestamp'].apply(utils.date_convert)
    filtered_events= joined[(convTS>=joined['mindate'])&(convTS<=joined['indx_date'])]
    filtered_events.to_csv(deliverables_path + 'etl_filtered_events.csv', columns=['patient_id', 'event_id', 'value'], index=False)  
    return filtered_events


def aggregate_events(filtered_events_df, mortality_df,feature_map_df, deliverables_path):
    
    '''

    Refer to instructions in Q3 a

    Suggested steps:
    1. Replace event_id's with index available in event_feature_map.csv
    2. Remove events with n/a values
    3. Aggregate events using sum and mean to calculate feature value 
    4. Normalize the values obtained above using min-max normalization
    
    
    IMPORTANT:
    Save aggregated_events to a csv file in the deliverables folder named as etl_aggregated_events.csv. 
    Use the global variable deliverables_path while specifying the filepath. 
    Each row is of the form patient_id, event_id, value.
    The csv file should have a header .
    For example if you are using Pandas, you could write: 
        aggregated_events.to_csv(deliverables_path + 'etl_aggregated_events.csv', columns=['patient_id', 'feature_id', 'feature_value'], index=False)

    Return filtered_events
    '''
    joined=pd.merge(filtered_events_df,feature_map_df,how='left',on='event_id')
    joined=joined[joined['value'].notnull()]
    split1,split2=joined[joined['idx']<2680],joined[joined['idx']>=2680]
    aggre1=split1[['patient_id','idx','value']].groupby(['patient_id','idx']).sum()
    aggre2=split2[['patient_id','idx','value']].groupby(['patient_id','idx']).count()
    result=pd.concat([aggre1,aggre2]).sort()
    aggregated_events = result.reset_index()
    listgroup=aggregated_events.groupby('idx')
    f=open(deliverables_path + 'etl_aggregated_events.csv','wb')
    f.write('patient_id,'+'idx'+',value'+'\n')
    for name,group in listgroup:
        normalized= (group['value'].as_matrix())/(group['value'].as_matrix().max())
        for i in range(len(normalized)):
            f.write(str(int(group.iloc[i]['patient_id']))+','+str(name)+','+str(normalized[i])+'\n')
    f.close()
    aggregated_events=pd.read_csv(deliverables_path + 'etl_aggregated_events.csv')
    return aggregated_events

def create_features(events, mortality, feature_map):
    
    deliverables_path = '../deliverables/'

    #Calculate index date
    indx_date = calculate_index_date(events, mortality, deliverables_path)


    #Filter events in the observation window
    filtered_events = filter_events(events, indx_date,  deliverables_path)
    
    #Aggregate the event values for each patient 
    aggregated_events = aggregate_events(filtered_events, mortality, feature_map, deliverables_path)

    ''' - 
    1. patient_features :  Key - patient_id and value is array of tuples(feature_id, feature_value)
    2. mortality : Key - patient_id and value is mortality label
    
    '''   
    patient_features = defaultdict(list)
    deadpool=mortality['patient_id']#500 people dead
    mortality = {}
    for i in range(len(aggregated_events.index)):
        line=aggregated_events.iloc[i]
        patient_features[line['patient_id']].append((line['idx'],line['value']))
    paId=indx_date['patient_id']
    #print len(set(paId).intersection(set(deadpool))) 500
    for i in range(len(paId)):#1000 ID
        if paId[i] in deadpool.as_matrix():
            mortality[paId[i]]=1
        else:
            mortality[paId[i]]=0
    
    return patient_features, mortality


def save_svmlight(patient_features, mortality, op_file, op_deliverable):
    
    '''
    Create two files:
    1. op_file - which saves the features in svmlight format. (See instructions in Q3d for detailed explanation)
    2. op_deliverable - which saves the features in following format:
       patient_id1 label feature_id:feature_value feature_id:feature_value feature_id:feature_value ...
       patient_id2 label feature_id:feature_value feature_id:feature_value feature_id:feature_value ...  
    
    Note: Please make sure the features are ordered in ascending order, and patients are stored in ascending order as well.     
    '''
    deliverable1 = open(op_file, 'wb')
    deliverable2 = open(op_deliverable, 'wb')
    copy=patient_features.keys()[:]
    copy.sort()
    for key in copy:
        deliverable1.write(str(mortality[key])+' '+utils.bag_to_svmlight(sorted(patient_features[key]))+'\n');
        deliverable2.write(str(int(key))+' '+str(mortality[key])+' '+utils.bag_to_svmlight(sorted(patient_features[key]))+'\n');
    deliverable1.close()
    deliverable2.close()

def main():
    train_path = '../data/train/'
    events, mortality, feature_map = read_csv(train_path)
    patient_features, mortality = create_features(events, mortality, feature_map)
    print patient_features
    print mortality
    save_svmlight(patient_features, mortality, '../deliverables/features_svmlight.train', '../deliverables/features.train')

if __name__ == "__main__":
    main()




