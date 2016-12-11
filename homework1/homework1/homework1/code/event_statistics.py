import time
import pandas as pd
import numpy as np
import utils

def read_csv(filepath):
    '''
    Read the events.csv and mortality_events.csv files. Variables returned from this function are passed as input to the metric functions.
    This function needs to be completed.
    '''
    events = pd.read_csv(filepath+"events.csv")

    mortality = pd.read_csv(filepath+"mortality_events.csv",index_col='patient_id')

    return events, mortality

def event_count_metrics(events, mortality):
    '''
    Event count is defined as the number of events recorded for a given patient.
    This function needs to be completed.
    '''
    dead_count, alive_count=[],[]

    events['count']=1

    results=events[["patient_id","count"]].groupby("patient_id").sum()

    total_patient=len(results.index)
    
    for i in range(total_patient):
        id1=results.index[i]
        sum1=results.iloc[i]['count']
        if id1 in mortality.index:
            dead_count.append(sum1)
        else:
            alive_count.append(sum1)
    
    avg_dead_event_count = np.mean(dead_count)

    max_dead_event_count = max(dead_count)

    min_dead_event_count = min(dead_count)

    avg_alive_event_count = np.mean(alive_count)

    max_alive_event_count = max(alive_count)

    min_alive_event_count = min(alive_count)

    return min_dead_event_count, max_dead_event_count, avg_dead_event_count, min_alive_event_count, max_alive_event_count, avg_alive_event_count

def encounter_count_metrics(events, mortality):
    '''
    Encounter count is defined as the count of unique dates on which a given patient visited the ICU. 
    This function needs to be completed.
    '''
    dead_count, alive_count=[],[]

    events['count']=1

    results1=events[["patient_id","count"]].groupby("patient_id").sum()

    total_patient=len(results1.index)

    results2=events[["patient_id","timestamp","count"]].groupby(["patient_id","timestamp"]).sum()

    results2['value1']=1

    for i in range(total_patient):
        id1=results1.index[i]
        sum1=results2.loc[id1].sum()['value1']
        if id1 in mortality.index:
            dead_count.append(sum1)
        else:
            alive_count.append(sum1)
    
    avg_dead_encounter_count =np.mean(dead_count)

    max_dead_encounter_count =  max(dead_count)

    min_dead_encounter_count = min(dead_count)

    avg_alive_encounter_count =  np.mean(alive_count)

    max_alive_encounter_count = max(alive_count)

    min_alive_encounter_count = min(alive_count)

    return min_dead_encounter_count, max_dead_encounter_count, avg_dead_encounter_count, min_alive_encounter_count, max_alive_encounter_count, avg_alive_encounter_count

def record_length_metrics(events, mortality):
    '''
    Record length is the duration between the first event and the last event for a given patient. 
    This function needs to be completed.
    '''
    dead_count, alive_count=[],[]

    events['count']=1

    results1=events[["patient_id","count"]].groupby("patient_id").sum()

    total_patient=len(results1.index)

    results2=events[["patient_id","timestamp","count"]].groupby(["patient_id","timestamp"]).sum()

    for i in range(total_patient):
        id1=results1.index[i]
        maxtime=utils.date_convert(results2.loc[id1].index[0])
        mintime=maxtime
        for j in range(len(results2.loc[id1].index)):
            nowtime=utils.date_convert(results2.loc[id1].index[j])
            if nowtime>maxtime:
                maxtime=nowtime
            if nowtime<mintime:
                mintime=nowtime
        day=(maxtime-mintime).days
        if id1 in mortality.index:
            dead_count.append(day)
        else:
            alive_count.append(day)
            
            
    avg_dead_rec_len = np.mean(dead_count)

    max_dead_rec_len = max(dead_count)

    min_dead_rec_len = min(dead_count)

    avg_alive_rec_len = np.mean(alive_count)

    max_alive_rec_len = max(alive_count)

    min_alive_rec_len = min(alive_count)

    return min_dead_rec_len, max_dead_rec_len, avg_dead_rec_len, min_alive_rec_len, max_alive_rec_len, avg_alive_rec_len

def main():

    #Modify the filepath to point to the CSV files in train_data
    train_path = "C:/users/dawei/bigdata-bootcamp/homework1/data/train/"
    events, mortality = read_csv(train_path)
 

    #Compute the event count metrics
    start_time = time.time()
    event_count = event_count_metrics(events, mortality)
    end_time = time.time()
    print("Time to compute event count metrics: " + str(end_time - start_time) + "s")
    print event_count

 

    #Compute the encounter count metrics
    start_time = time.time()
    encounter_count = encounter_count_metrics(events, mortality)
    end_time = time.time()
    print("Time to compute encounter count metrics: " + str(end_time - start_time) + "s")
    print encounter_count


    #Compute record length metrics
    start_time = time.time()
    record_length = record_length_metrics(events, mortality)
    end_time = time.time()
    print("Time to compute record length metrics: " + str(end_time - start_time) + "s")
    print record_length
    
if __name__ == "__main__":
    main()



