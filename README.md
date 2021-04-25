This is the compiled directory of all the python programs created for a Research Project funded by the DoD (Department of Defense) at the School of Government & Public Policy, University of Arizona. 

All software was developed by Omar Raef Gebril.

This research was aimed at identifying and geolocating violence committed by Non-State Actors in the region of Afghanistan.
The 4 W's ( Who did What to Whom and Where )



Contents of This Directory (intercoder_reliability):

	pandas(folder) 					-> 	Contains output CSVs of each annotator's data
	HTMLfileExtraction(folder) 			-> 	Contains files subset and data extraction from their HTML files
	extract_AAB(folder) 				-> 	Contains coder_AAB json annotations corpus
	extract_ORG(folder) 				-> 	Contains coder_ORG json annotations corpus
	extract_AK(folder) 				-> 	Contains coder_AK json annotations corpus
	intercoderReliability_produceCSVs.py(program)	-> 	Program to output annotation CSVs
	comparison_labels.py(program) 			-> 	Program to output inter-coder-reliability results

* CSVs have already been produced and you can find them in the folder 'pandas'
* To run tests, you only need to run comparison_labels.py, UNLESS
* Unless changes were made to our annotation corpus, in which case you need to
	First run intercoderReliability_produceCSVs.py,
	Then run comparison_labels.py


Entity legend for AAB is:

e_1 = Source (entity)
e_10 = Actor_Filter (entity)
e_2 = Target (entity)
e_3 = Action (entity)
e_6 = Location_Filter (entity)
e_7 = District (entity)
e_8 = Province (entity)
e_9 = Action_Filter (entity)
r_4 = 2crkgzh6f1v(e_1|e_3) (relation)
r_5 = hqtl7ah1zml(e_3|e_2) (relation)

Entity legend for ORG:

e_1 = Target (entity)
e_10 = Actor_Filter (entity)
e_2 = Action (entity)
e_3 = District (entity)
e_4 = Location_Filter (entity)
e_5 = Province (entity)
e_6 = Source (entity)
e_9 = Action_Filter (entity)
r_7 = 4mta7zhnb0l(e_6|e_2) (relation)
r_8 = uc9tx98g9pm(e_2|e_1) (relation)

Entity legend for AK is:

e_1 = Target (entity)
e_10 = Actor_Filter (entity)
e_2 = Action (entity)
e_3 = District (entity)
e_4 = Location_Filter (entity)
e_5 = Province (entity)
e_6 = Source (entity)
e_9 = Action_Filter (entity)
r_7 = qh1p8am4pal(e_6|e_2) (relation)
r_8 = n35bvh37lm0(e_2|e_1) (relation)