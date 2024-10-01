Jupyter notebook “FullCa2Pipeline_HongliEdits”: this is the pipeline! It contains several sections:
o	1. Convert Behavioral Output File to Required Trial-by-Trial Format: this reformats behavioral data from a Matlab format to a CSV format where the rows represent individual trials
o	2. Alignment of Ca2+ Signal and Plotting: self-explanatory; requires cell identification files from Suite2P and the behavior CSV. This section is short because a lot of the magic happens in the “required_python_code” scripts that Albert wrote.
o	3. Multiple Linear Regression: self-explanatory; adapted from Hongli – largely Hongli’s code, with small adjustments by me
o	4. Examining MLR Results: my code to examine the fraction of neurons that significantly encode task-related variables
o	5. Plotting neuron activity from saved dataframes: my code to generate heatmaps and line plots showing individual neuron activity for specific stimulus types
o	6. Other plots: older code I wrote with help from a previous undergrad, Nick Kiel, to look at average neuron activity aligned to stimulus

Jupyter notebook “Suite2P_Analysis”:  contains code for preprocessing a single imaging session, sorting that session by activity percentiles, and plotting Ca2+ traces; plotting the imaging FOVs; and calculating mean integrated activity and cell count across multiple animals and sessions.

Jupyter notebook “Behavior_Analysis”: contains code for examine running velocity and licking behavior, and assessing the trial type statistics of each session (i.e., fraction of hit trials in each session, when miss trials occurred, etc.)

Code for preprocessing behavioral data files include:
- get_Headfix_GoNo_EventTimes.m
- get_session_files.m
- gonogo_extract_behavior_data.m
- run_for_sessions.m

Code for behavioral data analysis include:
- behavior_pipeline_full.m
- cumulative_cues.m
- dprime_1session.m
- get_behavior.m
- remove_disengaged_trials.m
You will also need dprime_simple.m and nansem.m, which are availabe for download from their original authors online.


Calcium imaging time series were motion corrected with Suite2P or NormCorre and manually inspected for quality. ROIs are then identified through Suite2P.	Imaging data is aligned with behavior through the “FullCa2Pipeline_HongliEdits” Jupyter notebook, sections 1 and 2. The behavior must be preprocessed first. “FullCa2Pipeline_HongliEdits” references python scripts available in the "adolescent_Ca2_preprocessing" repo, which must be stored locally to run the notebook.

