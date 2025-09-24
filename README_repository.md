# Elise-Katerina_MEP

## Project description

This projects contains scripts for image analysis of GUVs encapsulating septin and/or actin. 
It is assumed that at least vesicle detection has already been carried out in DisGUVery and the scripts are facilitate the DisGUVery output format.

**Important Note:** Hashtags in the cells of the output csv files of DisGUVery must be deleted before use in python as errors occur. Hashtags appear in the title of the first column of the detected vesicles files and intensity profiles exported data. In the latter the second empty row that starts with a # must also be removed before processing. 


## Repository contents:

**Main files:**

- **main.py:** 

The main script for the analysis of the Intensity profiles.

- **skeleton.py:** 

The file containing all necessary functions required in main.py.

- **analysis.py:** 

Script that collects all vesicles of a specific composition marked "OK" in new csv files and using those creates a boxplot figure for all compositions. 

- **README_script.md:**

The file containing the main information and instructions on the main Intensity profiles analysis script.


**Preliminary files:**

- **intensity_profiles.py:** 

Primary script for the analysis of the Intensity profiles, before organizing the code and splitting into functions. Appropriate when only septin is present. 

- **intensity_profiles_with_actin.py:** 

Primary script for the analysis of the Intensity profiles, before organizing the code and splitting into functions. Appropriate when both septin and actin are present. 


**Assistive files:**

- **intensity_linear_profiles.py:** 

Script that creates different equally-distanced linear profiles for a vesicle and draws them for visual inspection during troubleshooting.

- **processing_intensity_disguvery_data.py:** 

Script that reads the intensity profiles calculated by DisGUVery allowing for possible further processing.

- **working_with_tiff_csv.py:** 

Script that contains usefull functions for handling tiff and csv files, including image z-stacks and time evolution stacks. Meant to be used for quick reference when writing new code. 

- **playing_around.py:**

Script that contains both usefull functions for handling tiff and csv files as well as a first calculation of the intenisty radial profiles. 


# Analysis of Intensity profiles

This script calculates the radial and angular intensity profiles of vesicles detected by DisGUVery. It accounts for background noise and quantifies protein-membrane co-localization.

-----------------------------------------------------------------------------------
Input
-----------------------------------------------------------------------------------
path_to_script  :    Specify the location of the script.
		     Note: Both main.py and skeleton.py should be in the same folder. 

Septin and Actin:    Boolean parameters 
		     If the corresponding protein is present, assign True, otherwise 
		     assign False. 	       

Analysis parameters: See related comments in the main.py script.

Function booleans:   Determine desired optional output apart from the localization
		     quantification. Additional option of a manual choice of the 
		     thresholding method used through "threshold_method_manual".
		     If output is desired, assign True, otherwise assign False. 
			   
-----------------------------------------------------------------------------------
Input files : Path and file names
-----------------------------------------------------------------------------------
The data sets to be analyzed should be stored in a subfolder named "data" contained 
in the folder of the script's path (path_to_data = path_to_script + "\\data").

For every image the files necessary to run the script are:
- tiff file of the membrane channel
- tiff file(s) of the protein(s) present
- csv file with DisGUVery's detected vesicles 
  IMPORTANT NOTE: In the csv DisGUVery file the "#" should be deleted from the first
                  column's header ("# id_vesicle" should be renamed " id_vesicle").

The **csv file name** should be of the format:
"experiment_date-experiment_project_name-image_name-detected_vesicles"

example: 20221012-sept_hex_pip2_0%-Region 1-detected_vesicles

The **tiff file names** should be of the format:  
"experiment_date-experiment_project_name-image_name-'...channel...'", where 
'...channel...' should be:
- 'C1' for the membrane.
- 'C2' for the single protein present or for the septin channel if both proteins are
   present.
- 'C3' for the actin channel in case both proteins are present.

example: 20221012-sept_hex_pip2_0%-Region 1-C1
  
-----------------------------------------------------------------------------------
Output
-----------------------------------------------------------------------------------
A file containing the experiment information, the vesicle coordinates, the backgrond 
noise estimated, the calculated localization and a comment for suggested visual 
inspection is created in the path: "path_to_script + \\output" in a folder specific 
for each image analyzed. 
Additionally the intensity radial and angular profiles, the used masks and and image 
overview with annotated vesicle centres are saved as a png file, if requested by the user.

-----------------------------------------------------------------------------------
-----------------------------------------------------------------------------------


