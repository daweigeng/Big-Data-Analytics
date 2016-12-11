\copy (select subject_id, date_part('year',age) as year) 
		from (select age(ad.mt,patients.dob) as age,ad.subject_id)
				from patients,(select min(admittime) as mt, subject_id from
									admissions group by subject_id) ad 
		where ad.subject_id=patients.subject_id) id) 
to '\paper\code\data\age.csv' delimiter ',' csv header;

\copy (select noteevents.subject_id, string_agg(text,' ') as text from noteevents,(select noteevents.subject_id,max(chartdate-dob) as age from noteevents,patients where noteevents.subject_id=patients.subject_id group by noteevents.subject_id) id where noteevents.subject_id=id.subject_id and category!='Discharge summary' and age>'6570 days' group by noteevents.subject_id) 
to 'E:/A+GTCLASS/CSE8803/paper/code/data/note1.csv' delimiter ',' csv header;

\copy (select * as text from patients) to 'E:/A+GTCLASS/CSE8803/paper/code/data/patients_tab.csv' delimiter ',' csv header;


//feature construction:

\copy (Select subject_id, count(hadm_id) from admissions group by subject_id)
to 'E:\A+GTCLASS\CSE8803\PAPER\CODE\DATA\count_admin.csv' delimiter ',' csv header;

\copy (Select subject_id, avg(los) from icustays group by subject_id)
to 'E:\A+GTCLASS\CSE8803\PAPER\CODE\DATA\count_admin_stay.csv' delimiter ',' csv header;

\copy (Select noteevents.subject_id, text, from icustays group by subject_id)
to 'E:\A+GTCLASS\CSE8803\PAPER\CODE\DATA\count_admin_stay.csv' delimiter ',' csv header;

