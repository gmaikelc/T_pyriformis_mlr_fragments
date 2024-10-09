# -*- coding: utf-8 -*-
"""
Created on Wed Oct 9 17:41:37 2024

@author: Gerardo Casanola & Karel Dieguez-Santana
"""


#%% Importing libraries

from pathlib import Path
import pandas as pd
import pickle
from molvs import Standardizer
from openbabel import openbabel
from multiprocessing import freeze_support
import numpy as np
import plotly.graph_objects as go
import networkx as nx


import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches
from matplotlib.lines import Line2D

#Import Libraries
import math 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error

# packages for streamlit
import streamlit as st
from PIL import Image
import io
import base64
from rdkit.Chem import Draw, rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

from rdkit import Chem, RDConfig
from rdkit.Chem import AllChem, rdFingerprintGenerator, Descriptors, Draw
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.DataStructs import cDataStructs
from io import StringIO
from mordred import Calculator, descriptors
import seaborn as sns
import sys, os, shutil
import matplotlib.pyplot as plt
from streamlit_ketcher import st_ketcher
import time
import subprocess
import uuid
from filelock import Timeout, FileLock

from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import time



#%% PAGE CONFIG

#---------------------------------#
# Page layout
## Page expands to full width
st.set_page_config(page_title='Tetrahymena pyriformis ecotoxicity predicion based on Chemical Fragment and Explainable Machine Learning Approaches', page_icon=":computer:", layout='wide')

######
# Function to put a picture as header   
def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded

image = Image.open('cropped-header.png')
st.image(image)

#st.title(':computer: t pyriformis ecotoxicity fragments')
#with st.expander("<span style='color: blue;'>More information</span>", expanded=True):
with st.expander("More information",):
    st.write("""

    **It is a free web-application for Ecotoxicology prediction in Tetrahymena Pyriformis based on Chemical Fragments**

   The impact of the aquatic environment can be determined with the testing based on the concentration of growth inhibition (IGC50) on the ciliated protozoan
   Tetrahymena pyriformis, which are an early warning of a toxic hazard to be at the top trophic levels in aquatic ecosystems. This organism is considered adequate
   for toxicological and safety testing of chemicals.

    The ML Aquatic Ecotox Tetrahymena sp predictor is a Web App that use Machine Learning to predict the aquatic ecotoxicology risk assesment of organic compounds. 

    The tool uses the following packages [RDKIT](https://www.rdkit.org/docs/index.html), [Mordred](https://github.com/mordred-descriptor/mordred), [MOLVS](https://molvs.readthedocs.io/), [Openbabel](https://github.com/openbabel/openbabel),
    [Scikit-learn](https://scikit-learn.org/stable/)
    
    """)

with st.expander("**Workflow**"):
    image = Image.open('toc.png')
    st.image(image, caption='Fragment-based Tetrahymena Ecotoxicity workflow')


#---------------------------------#
# Sidebar - Collects user input features into dataframe
st.sidebar.header('Upload your CSV file')
st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/gmaikelc/T_pyriformis_mlr_fragments/main/virtual_smi_example.csv) 
""")

uploaded_file_1 = st.sidebar.file_uploader("Upload a CSV file with SMILES and fractions", type=["csv"])


#%% Standarization by MOLVS ####
####---------------------------------------------------------------------------####

def standardizer(df,pos):
    s = Standardizer()
    molecules = df[pos].tolist()
    standardized_molecules = []
    smi_pos=pos-2
    i = 1
    t = st.empty()
    
    

    for molecule in molecules:
        try:
            smiles = molecule.strip()
            mol = Chem.MolFromSmiles(smiles)
            standarized_mol = s.super_parent(mol) 
            standardizer_smiles = Chem.MolToSmiles(standarized_mol)
            standardized_molecules.append(standardizer_smiles)
            # st.write(f'\rProcessing molecule {i}/{len(molecules)}', end='', flush=True)
            t.markdown("Processing monomers: " + str(i) +"/" + str(len(molecules)))

            i = i + 1
        except:
            standardized_molecules.append(molecule)
    df['standarized_SMILES'] = standardized_molecules

    return df


#%% Protonation state at pH 7.4 ####
####---------------------------------------------------------------------------####

def charges_ph(molecule, ph):

    # obConversion it's neccesary for saving the objects
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("smi", "smi")
    
    # create the OBMol object and read the SMILE
    mol = openbabel.OBMol()
    obConversion.ReadString(mol, molecule)
    
    # Add H, correct pH and add H again, it's the only way it works
    #mol.AddHydrogens()
    #mol.CorrectForPH(7.4)
    #mol.AddHydrogens()
    
    # transforms the OBMOl objecto to string (SMILES)
    optimized = obConversion.WriteString(mol)
    
    return optimized

def smile_obabel_corrector(smiles_ionized):
    mol1 = Chem.MolFromSmiles(smiles_ionized, sanitize = True)

    smile_checked = Chem.MolToSmiles(mol1)
    return smile_checked


#%% formal charge calculation

def formal_charge_calculation(descriptors):
    smiles_list = descriptors["Smiles_OK"]
    charges = []
    for smiles in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smiles)
            charge = Chem.rdmolops.GetFormalCharge(mol)
            charges.append(charge)
        except:
            charges.append(None)
        
    descriptors["Formal_charge"] = charges
    return descriptors

def calc_descriptors(data, smiles_col_pos):
    
    # Create empty lists to store the results
    results3 = []
    results12 = []
    results13 = []
    results17 = []
    results18 = []
    molecule_counts1 = []
    molecule_counts2 = []
    molecule_counts4 = []
    molecule_counts5 = []
    molecule_counts6 = []
    molecule_counts7 = []
    molecule_counts8 = []
    molecule_counts9 = []
    molecule_counts10 = []
    molecule_counts11 = []
    molecule_counts14 = []
    molecule_counts15 = []
    molecule_counts16 = []
    molecule_counts19 = []
    molecule_counts20 = []
    molecule_counts21 = []
    molecule_counts22 = []
    
    # Define SMARTS patterns
    smarts_pattern1 = Chem.MolFromSmarts('[Cl]-[*]')  #OKK
    smarts_pattern2 = Chem.MolFromSmarts('[c;X3](c)(c)-[CH3]')  #OK
    smarts_pattern3 = Chem.MolFromSmarts('[O]=C-[C]-O')  #OK
    smarts_pattern4_1 = Chem.MolFromSmarts('c:c-c:c') #OK
    smarts_pattern4_2 = Chem.MolFromSmarts('c:c:c:c') #OK
    smarts_pattern5 = Chem.MolFromSmarts('[#6]-[#7]=[#8]')  #OK
    smarts_pattern6 = Chem.MolFromSmarts('[C,c][CH3]')
    smarts_pattern7 = Chem.MolFromSmarts('I-c:c:c') #OK           
    smarts_pattern8 = Chem.MolFromSmarts('[C,c]:[C,c]-[F]') #OK            
    smarts_pattern9_1 = Chem.MolFromSmarts('O-c:c-C') #OK
    smarts_pattern9_2 = Chem.MolFromSmarts('O-c:[cH]:c') #OK
    smarts_pattern10 = Chem.MolFromSmarts('[C,c][C;H2][C,c]') 
    smarts_pattern11 = Chem.MolFromSmarts('S-[C,c]') #OK
    smarts_pattern12 = Chem.MolFromSmarts('C-O-C=O') #OK
    smarts_pattern13 = Chem.MolFromSmarts('[C](=O)(O)[C,c]') #OK
    smarts_pattern14 = Chem.MolFromSmarts('[C;H1,H2](O)[C,c]') #OK
    smarts_pattern15 = Chem.MolFromSmarts('Br-c:c-O') #OK
    smarts_pattern16 = Chem.MolFromSmarts('O-[CH3]')  #OK
    smarts_pattern17 = Chem.MolFromSmarts('[C;H1,H2](=O)[C,c]') #OK
    smarts_pattern18 = Chem.MolFromSmarts('C=O')  #OK
    smarts_pattern19 = Chem.MolFromSmarts('c:c-C=O') #OK
    smarts_pattern20 = Chem.MolFromSmarts('[Br][C,c]') #OK
    smarts_pattern21 = Chem.MolFromSmarts('N')  #OK
    smarts_pattern22 = Chem.MolFromSmarts('Br-c:c-Br') #OK 
    smiles_list = []
    t = st.empty()

    # Placeholder for the spinner
    with st.spinner('CALCULATING DESCRIPTORS (STEP 1 OF 3)...'):
        time.sleep(1)  # Sleep for 5 seconds to mimic computation
        # Loop through each molecule in the dataset
  
    
        
        for pos, row in data.iterrows():
                molecule_name = row.iloc[0]  # Assuming the first column contains the molecule names
                molecule_smiles = row.iloc[smiles_col_pos]  # Assuming the specified column contains the SMILES
    
                if pd.isna(molecule_smiles) or molecule_smiles.strip() == '':
                            continue  # Skip to the next row if SMILES is empty
    
                mol = Chem.MolFromSmiles(molecule_smiles)  # Convert SMILES to RDKit Mol object
                if mol is not None:
                    smiles_ionized =  molecule_smiles #charges_ph(molecule_smiles, 7.4)
                    smile_checked = smiles_ionized #smile_obabel_corrector(smiles_ionized)
                    #smile_checked = smiles_ionized
                    smile_final = smile_checked.rstrip()
                    smiles_list.append(smile_final)
                    # Define substructure match logic for each pattern
                    fragment3 = mol.HasSubstructMatch(smarts_pattern3)
                    fragment9_1 = mol.HasSubstructMatch(smarts_pattern9_1)
                    #Check for matches in the molecule
                    fragment9_2 = mol.GetSubstructMatches(smarts_pattern9_2)
                   
                    # Filter the matches to count only unique c-O bonds
                    filtered_matches_oc = []
                    used_oxygen_atoms = set()  # Track oxygen atoms to ensure uniqueness
                    
                    for match in fragment9_2:
                        for atom_idx in match:
                            atom = mol.GetAtomWithIdx(atom_idx)
                            if atom.GetSymbol() == 'O' and atom_idx not in used_oxygen_atoms:
                                # Add the match if it contains a new oxygen
                                filtered_matches_oc.append(match)
                                used_oxygen_atoms.add(atom_idx)  # Mark this oxygen as used
                                break  # Only count one c-O per match
        
    
                    fragment12 = mol.HasSubstructMatch(smarts_pattern12)
                    fragment13 = mol.HasSubstructMatch(smarts_pattern13)
    
    
                    fragment15 = mol.GetSubstructMatches(smarts_pattern15)
                    # Find non-overlapping fragments
                    non_overlapping_matches_br_cc_o = []
                    used_indices_br_cc_o = set()
                    
                    for match in fragment15:
                        # Check if any of the indices in the current match overlap with already used indices
                        if not any(index in used_indices_br_cc_o for index in match):
                            # If there is no overlap, add this match to the non-overlapping list
                            non_overlapping_matches_br_cc_o.append(match)
                            # Mark these indices as used
                            used_indices_br_cc_o.update(match)
    
                    fragment17 = mol.HasSubstructMatch(smarts_pattern17)
                    fragment18 = mol.HasSubstructMatch(smarts_pattern18)
    
                    # Check for matches in the molecule
                    fragment19 = mol.GetSubstructMatches(smarts_pattern19)
                    
                    # Initialize a set to track used pairs of central atoms (position 1 and 2 in the match tuple)
                    used_central_pair_ccco = set()
                    
                    # Filter the matches to avoid duplicate (second, third) atom pairs
                    filtered_matches_ccco = []
                    for match in fragment19:
                        central_pair = (match[1], match[2])  # Second and third atoms (central pair)
                        
                        # Only keep the match if the pair hasn't been used yet
                        if central_pair not in used_central_pair:
                            filtered_matches_ccco.append(match)
                            # Mark the pair as used
                            used_central_pair_ccco.add(central_pair)
    
                    # Check for matches in the molecule
                    fragment22 = mol.GetSubstructMatches(smarts_pattern22)
            
                    # Find non-overlapping fragments
                    non_overlapping_matches_br = []
                    used_indices_br = set()
                    
                    for match in fragment22:
                    # Check if any of the indices in the current match overlap with already used indices
                        if not any(index in used_indices for index in match):
                            # If there is no overlap, add this match to the non-overlapping list
                            non_overlapping_matches_br.append(match)
                            # Mark these indices as used
                            used_indices_br.update(match)
    
                    
        
                    # Count the occurrences of smarts_pattern4 in the molecule
                    count_in_molecule1 = len(mol.GetSubstructMatches(smarts_pattern1))
                    count_in_molecule2 = len(mol.GetSubstructMatches(smarts_pattern2))
            
                    count_in_molecule4_1 = len(mol.GetSubstructMatches(smarts_pattern4_1))/4
                    count_in_molecule4_2 = len(mol.GetSubstructMatches(smarts_pattern4_2))/2
            
                    # Check for rings
                    ring_info = mol.GetRingInfo()
                    rings = ring_info.AtomRings()  # List of rings
                    
                    # Check if there are two rings and if they are connected
                    connected = False
                    if len(rings) >= 2:  # Check if there are at least two rings
                        # Convert ring atom indices to sets
                        ring1_atoms = set(rings[0])  # First ring's atoms
                        for i in range(1, len(rings)):
                            ring2_atoms = set(rings[i])  # Current ring's atoms
                            # Check for intersection (shared atoms)
                            if ring1_atoms.intersection(ring2_atoms):
                                connected = True
                                break
                    
                    # Adjust count1 if two rings are connected
                    if connected:
                        count_in_molecule4_2 -= 2
                        
                    count_in_molecule4 = count_in_molecule4_1 + count_in_molecule4_2
                    count_in_molecule5 = len(mol.GetSubstructMatches(smarts_pattern5))
                    count_in_molecule6 = len(mol.GetSubstructMatches(smarts_pattern6))
                    count_in_molecule7 = len(mol.GetSubstructMatches(smarts_pattern7))/2
                    count_in_molecule8 = len(mol.GetSubstructMatches(smarts_pattern8))/2
                    count_in_molecule9_1 = len(mol.GetSubstructMatches(smarts_pattern9_1))
                    count_in_molecule9_2 = len(filtered_matches_oc)
                    count_in_molecule9 = count_in_molecule9_1 + count_in_molecule9_2
                    count_in_molecule10 = len(mol.GetSubstructMatches(smarts_pattern10))
                    count_in_molecule11 = len(mol.GetSubstructMatches(smarts_pattern11))
                    count_in_molecule14 = len(mol.GetSubstructMatches(smarts_pattern14))
                    count_in_molecule15 = len(non_overlapping_matches_br_cc_o)
                    count_in_molecule16 = len(mol.GetSubstructMatches(smarts_pattern16))
                    count_in_molecule19 = len(filtered_matches_ccco)/2
                    count_in_molecule20 = len(mol.GetSubstructMatches(smarts_pattern20))
                    count_in_molecule21 = len(mol.GetSubstructMatches(smarts_pattern21))
                    count_in_molecule22 = len(non_overlapping_matches_br)
        
                    # Append the results for each fragment pattern to respective lists
                    
                    results3.append(int(fragment3))
                    results12.append(int(fragment12))
                    results13.append(int(fragment13))
                    results17.append(int(fragment17)) 
                    results18.append(int(fragment18))
                
                    molecule_counts1.append(int(count_in_molecule1))
                    molecule_counts2.append(int(count_in_molecule2))
                    molecule_counts4.append(int(count_in_molecule4))
                    molecule_counts5.append(int(count_in_molecule5))
                    molecule_counts6.append(int(count_in_molecule6))
                    molecule_counts7.append(int(count_in_molecule7))
                    molecule_counts8.append(int(count_in_molecule8))
                    molecule_counts9.append(int(count_in_molecule9))
                    molecule_counts10.append(int(count_in_molecule10))
                    molecule_counts11.append(int(count_in_molecule11))
                    molecule_counts14.append(int(count_in_molecule14))
                    molecule_counts15.append(int(count_in_molecule15))
                    molecule_counts16.append(int(count_in_molecule16))
                    molecule_counts19.append(int(count_in_molecule19))
                    molecule_counts20.append(int(count_in_molecule20))
                    molecule_counts21.append(int(count_in_molecule21))
                    molecule_counts22.append(int(count_in_molecule22))
    
    
            # Create a DataFrame to store the results
        descriptors_total = pd.DataFrame({
            'NAME': molecule_name,
            'Cl-C': molecule_counts1, 
            '(C=C),(C-C),(C-C),xC': molecule_counts2,
            'O=C-C-O': results3,
            'C=C-C=C': molecule_counts4,
            'C-N=O': molecule_counts5,
            '(C-C),xC': molecule_counts6,
            'I-C-C=C': molecule_counts7,
            'C-C-F': molecule_counts8,
            'C-C=C-O': molecule_counts9,
            '(C-C),(C-C),xC' : molecule_counts10,
            'S-C': molecule_counts11,
            'C-O-C=O' : results12,
            '(C=O),(C-C),(C-O),xC' : results13,
            '(C-C),(C-O),xC' : molecule_counts14,
            'Br-C-C-O': molecule_counts15,
            '(C-O),xC': molecule_counts16,
            '(C=O),(C-C),xC': results17,    
            'C=O': results18,
            'C=C-C=O' : molecule_counts19,
            '(Br-C),xBr': molecule_counts20,
            'N' : molecule_counts21,
            'Br-C=C-Br': molecule_counts22,
            
        })
    
        return descriptors_total, smiles_list


def reading_reorder(data, loaded_desc):
        
    #Select the specified columns from the DataFrame
    df_selected = data[loaded_desc]
    df_id = data.reset_index()
    df_id.rename(columns={'index': 'NAME'}, inplace=True)
    id = df_id['NAME'] 
    # Order the DataFrame by the specified list of columns
    test_data = df_selected.reindex(columns=loaded_desc)
    # Fill missing values with 0
    test_data = test_data.fillna(0)
    #descriptors_total = data[loaded_desc]


    return test_data, id






#%% normalizing data1
### ----------------------- ###

def normalize_data(train_data, test_data):
    df_train_normalized = pd.DataFrame(train_data)
    df_test_normalized = pd.DataFrame(test_data)

    return df_train_normalized, df_test_normalized


#%% Determining Applicability Domain (AD)

def applicability_domain(x_test_normalized, x_train_normalized):
    y_train=data_train['pLC50']
    X_train = x_train_normalized.values
    X_test = x_test_normalized.values
    # Calculate leverage and standard deviation for the training set
    hat_matrix_train = X_train @ np.linalg.inv(X_train.T @ X_train) @ X_train.T
    leverage_train = np.diagonal(hat_matrix_train)
    leverage_train=leverage_train.ravel()
    
    # Calculate leverage and standard deviation for the test set
    hat_matrix_test = X_test @ np.linalg.inv(X_train.T @ X_train) @ X_test.T
    leverage_test = np.diagonal(hat_matrix_test)
    leverage_test=leverage_test.ravel()


    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error

    # Train a linear regression model
    lr = LinearRegression()
    lr.fit(df_train_normalized, y_train)
    y_pred_train = lr.predict(df_train_normalized)
    
    std_dev_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    std_residual_train = (y_train - y_pred_train) / std_dev_train
    std_residual_train = std_residual_train.ravel()
    
    # threshold for the applicability domain
    
    h3 = 3*((x_train_normalized.shape[1]+1)/x_train_normalized.shape[0])  
    
    diagonal_compare = list(leverage_test)
    h_results =[]
    for valor in diagonal_compare:
        if valor < h3:
            h_results.append(True)
        else:
            h_results.append(False)         
    return h_results, leverage_train, leverage_test, std_residual_train 



 # Function to assign colors based on confidence values
def get_color(confidence):
    """
    Assigns a color based on the confidence value.

    Args:
        confidence (float): The confidence value.

    Returns:
        str: The color in hexadecimal format (e.g., '#RRGGBB').
    """
    # Define your color logic here based on confidence
    if confidence == "HIGH" or confidence == "Inside AD":
        return 'green'
    elif confidence == "MEDIUM":
        return 'yellow'
    else:
        confidence ==  "LOW"
        return 'red'


#%% Predictions        

def predictions(loaded_model, loaded_desc, df_test_normalized):
    scores = []
    h_values = []
    std_resd = []
    idx = data['ID']
    
    descriptors_model = loaded_desc
    # Placeholder for the spinner
    with st.spinner('CALCULATING PREDICTIONS (STEP 2 OF 3)...'):
        # Simulate a long-running computation
        time.sleep(1)  # Sleep for 5 seconds to mimic computation
     
        X = df_test_normalized[descriptors_model]
        predictions = loaded_model.predict(X)
        scores.append(predictions)
        
        # y_true and y_pred are the actual and predicted values, respectively
    
        # Create y_true array with all elements set to mean value and the same length as y_pred
        y_pred_test = predictions
        y_test = np.full_like(y_pred_test, mean_value)
        residuals_test = y_test -y_pred_test

        std_dev_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
        std_residual_test = (y_test - y_pred_test) / std_dev_test
        std_residual_test = std_residual_test.ravel()
          
        std_resd.append(std_residual_test)
        
        h_results, leverage_train, leverage_test, std_residual_train  = applicability_domain(df_test_normalized, df_train_normalized)
        h_values.append(h_results)
    

        dataframe_pred = pd.DataFrame(scores).T
        dataframe_pred.index = idx
        dataframe_pred.rename(columns={0: "pLC50"},inplace=True)
    
        dataframe_std = pd.DataFrame(std_resd).T
        dataframe_std.index = idx
                
        h_final = pd.DataFrame(h_values).T
        h_final.index = idx
        h_final.rename(columns={0: "Confidence"},inplace=True)

        std_ensemble = dataframe_std.iloc[:,0]
        # Create a mask using boolean indexing
        std_ad_calc = (std_ensemble >= 3) | (std_ensemble <= -3) 
        std_ad_calc = std_ad_calc.replace({True: 'Outside AD', False: 'Inside AD'})
   
    
        final_file = pd.concat([std_ad_calc,h_final,dataframe_pred], axis=1)
    
        final_file.rename(columns={0: "Std_residual"},inplace=True)
    
        h3 = 3*((df_train_normalized.shape[1]+1)/df_train_normalized.shape[0])  ##  Mas flexible

        final_file.loc[(final_file["Confidence"] == True) & ((final_file["Std_residual"] == 'Inside AD' )), 'Confidence'] = 'HIGH'
        final_file.loc[(final_file["Confidence"] == True) & ((final_file["Std_residual"] == 'Outside AD')), 'Confidence'] = 'LOW'
        final_file.loc[(final_file["Confidence"] == False) & ((final_file["Std_residual"] == 'Outside AD')), 'Confidence'] = 'LOW'
        final_file.loc[(final_file["Confidence"] == False) & ((final_file["Std_residual"] == 'Inside AD')), 'Confidence'] = 'MEDIUM'

        df_no_duplicates = final_file[~final_file.index.duplicated(keep='first')]
        styled_df = df_no_duplicates.style.apply(lambda row: [f"background-color: {get_color(row['Confidence'])}" for _ in row],subset=["Confidence"], axis=1)
    
        return final_file, styled_df,leverage_train,std_residual_train, leverage_test, std_residual_test


#Calculating the William's plot limits
def calculate_wp_plot_limits(leverage_train,std_residual_train, x_std_max=4, x_std_min=-4):
    
    with st.spinner('CALCULATING APPLICABILITY DOMAIN (STEP 3 OF 3)...'):
        # Simulate a long-running computation
        time.sleep(1)  # Sleep for 5 seconds to mimic computation
        # Getting maximum std value
        if std_residual_train.max() < 4:
            x_lim_max_std = x_std_max
        elif std_residual_train.max() > 4:
            x_lim_max_std = round(std_residual_train.max()) + 1

        # Getting minimum std value
        if std_residual_train.min() > -4:
            x_lim_min_std = x_std_min
        elif std_residual_train.min() < 4:
            x_lim_min_std = round(std_residual_train.min()) - 1

    
        #st.write('x_lim_max_std:', x_lim_max_std)
        #st.write('x_lim_min_std:', x_lim_min_std)

        # Calculation H critical
        n = len(leverage_train)
        p = df_train_normalized.shape[1]
        h_value = 3 * (p + 1) / n
        h_critical = round(h_value, 4)
        #st.write('Number of cases training:', n)
        #st.write('Number of variables:', p)
        #st.write('h_critical:', h_critical)

        # Getting maximum leverage value
        if leverage_train.max() < h_critical:
            x_lim_max_lev = h_critical + h_critical * 0.5
        elif leverage_train.max() > h_critical:
            x_lim_max_lev = leverage_train.max() + (leverage_train.max()) * 0.1

        # Getting minimum leverage value
        if leverage_train.min() < 0:
            x_lim_min_lev = x_lev_min - x_lev_min * 0.05
        elif leverage_train.min() > 0:
            x_lim_min_lev = 0

        #st.write('x_lim_max_lev:', x_lim_max_lev)

        return x_lim_max_std, x_lim_min_std, h_critical, x_lim_max_lev, x_lim_min_lev

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def williams_plot(leverage_train, leverage_test, std_residual_train, std_residual_test,id_list_1,
                  plot_color='cornflowerblue', show_plot=True, save_plot=False, filename=None, add_title=False, title=None):
    fig = go.Figure()

    # Add training data points
    fig.add_trace(go.Scatter(
        x=leverage_train,
        y=std_residual_train,
        mode='markers',
        marker=dict(color='cornflowerblue', size=10, line=dict(width=1, color='black')),
        name='Training'
    ))

    # Add test data points
    fig.add_trace(go.Scatter(
        x=leverage_test,
        y=std_residual_test,
        mode='markers',
        marker=dict(color='orange', size=10, line=dict(width=1, color='black')),
        name='Prediction',
        text = id_list_1, # Add compounds IDs for hover
        hoverinfo = 'text' #Show only the text when hovering
    ))

    # Add horizontal and vertical dashed lines
    fig.add_shape(type='line', x0=h_critical, y0=x_lim_min_std, x1=h_critical, y1=x_lim_max_std,
                  line=dict(color='black', dash='dash'))
    fig.add_shape(type='line', x0=x_lim_min_lev, y0=3, x1=x_lim_max_lev, y1=3,
                  line=dict(color='black', dash='dash'))
    fig.add_shape(type='line', x0=x_lim_min_lev, y0=-3, x1=x_lim_max_lev, y1=-3,
                  line=dict(color='black', dash='dash'))

    # Add rectangles for outlier zones
    fig.add_shape(type='rect', x0=x_lim_min_lev, y0=x_lim_min_std, x1=h_critical, y1=-3,
                  fillcolor='lightgray', opacity=0.4, line_width=0)
    fig.add_shape(type='rect', x0=x_lim_min_lev, y0=3, x1=h_critical, y1=x_lim_max_std,
                  fillcolor='lightgray', opacity=0.4, line_width=0)
                      
    fig.add_shape(type='rect', x0=h_critical, y0=x_lim_min_std, x1=x_lim_max_lev, y1=-3,
                  fillcolor='lightgray', opacity=0.4, line_width=0)
    fig.add_shape(type='rect', x0=h_critical, y0=3, x1=x_lim_max_lev, y1=x_lim_max_std,
                  fillcolor='lightgray', opacity=0.4, line_width=0)

    # Add annotations for outlier zones
    fig.add_annotation(x=(h_critical + x_lim_min_lev) / 2, y=-3.5, text='Outlier zone', showarrow=False,
                       font=dict(size=15))
    fig.add_annotation(x=(h_critical + x_lim_min_lev) / 2, y=3.5, text='Outlier zone', showarrow=False,
                       font=dict(size=15))
    fig.add_annotation(x=(h_critical + x_lim_max_lev) / 2, y=-3.5, text='Outlier zone', showarrow=False,
                       font=dict(size=15))
    fig.add_annotation(x=(h_critical + x_lim_max_lev) / 2, y=3.5, text='Outlier zone', showarrow=False,
                       font=dict(size=15))

    # Update layout
    fig.update_layout(
        width=600,
        height=600,
        xaxis=dict(title='Leverage', range=[x_lim_min_lev, x_lim_max_lev], tickfont=dict(size=15)),
        yaxis=dict(title='Std Residuals', range=[x_lim_min_std, x_lim_max_std], tickfont=dict(size=15)),
        legend=dict(x=0.99, y=0.825, xanchor='right', yanchor='top', font=dict(size=20)),
        showlegend=True
    )

    if add_title and title:
        fig.update_layout(title=dict(text=title, font=dict(size=20)))

    if save_plot and filename:
        fig.write_image(filename)

    if show_plot:
        fig.show()

    return fig


#%%
def filedownload1(df):
    csv = df.to_csv(index=True,header=True)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="ml_toxicity_t_pyriformis_pLC50_results.csv">Download CSV File with results</a>'
    return href


# Create a function to generate an image with a specific SMARTS pattern highlighted
def generate_molecule_image(smiles, highlight_mol, color=(1, 0, 0), mol_size=(300, 300)):
    # Function to generate a molecule image with highlighted substructure
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        rdDepictor.Compute2DCoords(mol)
        drawer = rdMolDraw2D.MolDraw2DCairo(mol_size[0], mol_size[1])
        
        match = mol.GetSubstructMatch(highlight_mol)
        highlight_atoms = list(match) if match else []
        highlight_colors = {atom_idx: color for atom_idx in highlight_atoms}
        
        drawer.DrawMolecule(mol, highlightAtoms=highlight_atoms, highlightAtomColors=highlight_colors)
        drawer.FinishDrawing()
        
        # Convert the drawing to PNG format and load it into an Image object
        png_data = drawer.GetDrawingText()
        img = Image.open(io.BytesIO(png_data))
        return img

#%% RUN

data_train = pd.read_csv("data/" + "data_Tpyriformis_22var_original_training.csv")
mean_value = data_train['pLC50'].mean()
loaded_model = pickle.load(open("models/" + "ml_model_tetrahymena_pyriformis_structural.pickle", 'rb'))
loaded_desc = pickle.load(open("models/" + "ml_descriptor_tetrahymena_pyriformis_structural.pickle", 'rb'))

# Define the SMARTS patterns you want to highlight
smarts_patterns = {
    'O=C-C-O': '[O]=C-[C]-O',
    'C-C-F': '[C,c]:[C,c]-[F]',
    '(C=O,(C-C),xC': '[C;H1,H2](=O)[C,c]'
}

# Compile SMARTS patterns into RDKit Mol objects
highlight_mols = {name: Chem.MolFromSmarts(pattern) for name, pattern in smarts_patterns.items()}



#Uploaded file calculation ####
if uploaded_file_1 is not None:
    run = st.button("RUN the Model")
    if run == True:
        data = pd.read_csv(uploaded_file_1,) 
        
        train_data = data_train[loaded_desc]
        
        # Calculate descriptors and SMILES for the first data
        descriptors_total_1, smiles_list_1 = calc_descriptors(data, 1)
            
        #df_fragments         
        #Selecting the descriptors based on model for salt water component
        test_data1, id_list_1 =  reading_reorder(descriptors_total_1,loaded_desc)
        
        X_final2= test_data1
        
        df_train_normalized, df_test_normalized = normalize_data(train_data, X_final2)
        
        final_file, styled_df,leverage_train,std_residual_train, leverage_test, std_residual_test= predictions(loaded_model, loaded_desc, df_test_normalized)
        
        x_lim_max_std, x_lim_min_std, h_critical, x_lim_max_lev, x_lim_min_lev = calculate_wp_plot_limits(leverage_train,std_residual_train, x_std_max=4, x_std_min=-4)
       
        figure  = williams_plot(leverage_train, leverage_test, std_residual_train, std_residual_test,id_list_1)
         
        col1, col2 = st.columns(2)

        with col1:
            st.header("Tetrahymena pyriformis",divider='blue')
            st.subheader(r'Predictions')
            st.write(styled_df)
        with col2:
            st.markdown("<h2 style='text-align: center; font-size: 30px;'>William's Plot (Applicability Domain)</h2>", unsafe_allow_html=True)
            st.plotly_chart(figure,use_container_width=True)
        st.markdown(":point_down: **Here you can download the results for T. pyriformis MLR model**", unsafe_allow_html=True,)
        st.markdown(filedownload1(final_file), unsafe_allow_html=True)

        # Display the top molecule with each SMARTS highlighted in separate images
        st.title("Molecule with Highlighted Substructures")
       

# Example file
else:
    st.info('👈🏼👈🏼👈🏼   Awaiting for CSV file to be uploaded.')
    if st.button('Press to use Example CSV Dataset with smiles'):
        data = pd.read_csv("virtual_smi_example.csv")
        
        train_data = data_train[loaded_desc]
        
        
        # Calculate descriptors and SMILES for the first data
        descriptors_total_1, smiles_list_1 = calc_descriptors(data, 1)
             
        #Selecting the descriptors based on model for salt water component
        test_data1, id_list_1 =  reading_reorder(descriptors_total_1,loaded_desc)

        X_final2= test_data1
        
        df_train_normalized, df_test_normalized = normalize_data(train_data, X_final2)
        
        final_file, styled_df,leverage_train,std_residual_train, leverage_test, std_residual_test= predictions(loaded_model, loaded_desc, df_test_normalized)
        
        x_lim_max_std, x_lim_min_std, h_critical, x_lim_max_lev, x_lim_min_lev = calculate_wp_plot_limits(leverage_train,std_residual_train, x_std_max=4, x_std_min=-4)
        
        figure  = williams_plot(leverage_train, leverage_test, std_residual_train, std_residual_test,id_list_1)
    
        col1, col2 = st.columns(2)

        with col1:
            st.header("Tetrahymena pyriformis",divider='blue')
            st.subheader(r'Predictions')
            st.write(styled_df)
        with col2:
            st.markdown("<h2 style='text-align: center; font-size: 25px;'>William's Plot (Applicability Domain)</h2>", unsafe_allow_html=True)
            st.plotly_chart(figure,use_container_width=True)
        st.markdown(":point_down: **Here you can download the results for T. pyriformis MLR model**", unsafe_allow_html=True,)
        st.markdown(filedownload1(final_file), unsafe_allow_html=True)

        # Display the top molecule with each SMARTS highlighted in separate images
        #st.markdown("<h2 style='text-align: center; font-size: 20px;'>Molecule with Highlighted Substructures)</h2>", unsafe_allow_html=True)
        #im = Draw.MolToImage(Chem.MolFromSmiles('CCCC(C)CC1=CC=C(CCC)C(CCC)=C1'),fitImage=True)
        #st.image(im)
        
        # Iterate over the first 5 rows of the dataframe
        for index, row in data.head(5).iterrows():
            molecule_id = row.iloc[0]  # Assuming molecule ID is in the first column
            smiles = row.iloc[1]  # Assuming SMILES string is in the second column
            st.subheader(f"Molecule ID: {molecule_id}")
            
            # Iterate over SMARTS patterns and generate separate images
            for i, (smarts_name, highlight_mol) in enumerate(highlight_mols.items()):
                # Choose different color for each SMARTS pattern
                colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]  # Red, Green, Blue for different SMARTS
                img = generate_molecule_image(smiles, highlight_mol, color=colors[i])
                
                if img:
                  # Directly display the RDKit-generated image
                    st.image(img, caption=f'Molecule ID: {molecule_id} - Highlight: {smarts_name}', use_column_width=True)
                    #display(img)
                    print(f'Molecule ID: {molecule_id} - Highlight: {smarts_name}')
                else:
                    print(f"Could not generate image for Molecule ID: {molecule_id} - SMARTS: {smarts_name}")
                
        
       
## From Drawn Structure ##########################
on2 = st.toggle('Use drawn structure',key="13")
with st.expander("SMILES editor"):
    drawer = st_ketcher(key="12")
    st.caption("Click on Apply to save the drawn structure as input.")
     
if on2:
    smiles_list=drawer
        
    run = st.button("Click to make prediction for the drawn structure")
    if run == True:  
        
        ID='1'
        data = pd.DataFrame({'ID': [ID], 'Smiles_1': [smiles_list]})

        train_data = data_train[loaded_desc]
        
        # Calculate descriptors and SMILES for the first data
        descriptors_total_1, smiles_list_1 = calc_descriptors(data, 1)
                 
        #Selecting the descriptors based on model for salt water component
        test_data1, id_list_1 =  reading_reorder(descriptors_total_1,loaded_desc)
                  
        X_final2= test_data1
 
        df_train_normalized, df_test_normalized = normalize_data(train_data, X_final2)
        #st.markdown(filedownload5(df_test_normalized), unsafe_allow_html=True)
        
        final_file, styled_df,leverage_train,std_residual_train, leverage_test, std_residual_test= predictions(loaded_model, loaded_desc, df_test_normalized)
        
        x_lim_max_std, x_lim_min_std, h_critical, x_lim_max_lev, x_lim_min_lev = calculate_wp_plot_limits(leverage_train,std_residual_train, x_std_max=4, x_std_min=-4)
        
        figure  = williams_plot(leverage_train, leverage_test, std_residual_train, std_residual_test,id_list_1)
        
        col1,col2 = st.columns(2)

        with col1:
            st.header("T. pyriformis",divider='blue')
            st.subheader(r'Predictions')
            st.write(styled_df)
        with col2:
            st.markdown("<h2 style='text-align: center; font-size: 30px;'>William's Plot (Applicability Domain)</h2>", unsafe_allow_html=True)
            st.plotly_chart(figure,use_container_width=True)
        st.markdown(":point_down: **Here you can download the results for T. pyriformis model**", unsafe_allow_html=True,)
        st.markdown(filedownload1(final_file), unsafe_allow_html=True)

#Footer edit

footer="""<style>
a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: underline;
}
a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}
.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: white;
color: black;
text-align: center;
}
</style>
<div class="footer">
<p>Made in  🐍 and <img style='display: ; 
' href="https://streamlit.io" src="https://i.imgur.com/iIOA6kU.png" target="_blank"></img> Developed by <a style='display: ;
 text-align: center' href="https://www.linkedin.com/in/gerardo-m-casanola-martin-27238553/" target="_blank">Gerardo M. Casanola</a> and <a style='display: ; 
 text-align: center' href="https://www.linkedin.com/in/karel-dieguez-santana-b41020122/" target="_blank">Karel Dieguez-Santana</a>  

</div>
"""
st.markdown(footer,unsafe_allow_html=True)
