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

from typing import Dict, Tuple, List

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


# ============== Sidebar (file upload) ==================
st.sidebar.header('Upload your CSV file')
st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/gmaikelc/T_pyriformis_mlr_fragments/main/virtual_smi_example.csv)
""")
uploaded_file_1 = st.sidebar.file_uploader(
    "Upload a CSV file with SMILES and fractions", type=["csv"]
)

# ============== Small utilities (FIX helpers) ==========
def ensure_unique_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure column names are unique to avoid narwhals DuplicateError."""
    if not df.columns.is_unique:
        df = df.loc[:, ~df.columns.duplicated()].copy()
    return df

def _as_1d_text(x):
    """Coerce DataFrame/Series/list/ndarray to list[str] 1-D for Plotly text=."""
    if isinstance(x, pd.DataFrame):
        if x.shape[1] == 0:
            return []
        x = x.iloc[:, 0]
    if isinstance(x, pd.Series):
        arr = x.values
    else:
        arr = np.asarray(x)
    arr = np.ravel(arr)
    return [str(v) for v in arr]

def _as_1d_array(x):
    return np.ravel(np.asarray(x))

# ============== Standardization ========================
def standardizer(df, pos: int):
    """Standardize SMILES using MOLVS (if available)."""
    if Standardizer is None:
        st.warning("molvs not installed; skipping standardization.")
        df['standarized_SMILES'] = df.iloc[:, pos].astype(str)
        return df

    s = Standardizer()
    molecules = df.iloc[:, pos].astype(str).tolist()
    standardized_molecules = []
    t = st.empty()

    for i, smiles in enumerate(molecules, start=1):
        try:
            sm = smiles.strip()
            mol = Chem.MolFromSmiles(sm)
            std_mol = s.super_parent(mol)
            std_smiles = Chem.MolToSmiles(std_mol)
            standardized_molecules.append(std_smiles)
        except Exception:
            standardized_molecules.append(smiles)
        t.markdown(f"Processing monomers: {i}/{len(molecules)}")

    df['standarized_SMILES'] = standardized_molecules
    return df

# ============== Optional OpenBabel protonation =========
def charges_ph(molecule: str, ph: float) -> str:
    if openbabel is None:
        return molecule
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("smi", "smi")
    mol = openbabel.OBMol()
    obConversion.ReadString(mol, molecule)
    # You can enable:
    # mol.AddHydrogens(); mol.CorrectForPH(ph); mol.AddHydrogens()
    return obConversion.WriteString(mol)

def smile_obabel_corrector(smiles_ionized: str) -> str:
    mol1 = Chem.MolFromSmiles(smiles_ionized, sanitize=True)
    return Chem.MolToSmiles(mol1) if mol1 else smiles_ionized

# ============== Formal charge ==========================
def formal_charge_calculation(descriptors: pd.DataFrame) -> pd.DataFrame:
    smiles_list = descriptors["Smiles_OK"]
    charges = []
    for smiles in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smiles)
            charges.append(Chem.rdmolops.GetFormalCharge(mol))
        except Exception:
            charges.append(None)
    descriptors["Formal_charge"] = charges
    return descriptors

# ============== SMARTS-based fragment counts ===========
def calc_descriptors(data: pd.DataFrame, smiles_col_pos: int):
    """
    Build fragment counts & presence flags.
    FIXES:
      - Proper 'NAME' list collected in loop
      - Typos corrected for sets: used_central_pair_ccco, used_indices_br
    """
    # SMARTS
    smarts_pattern1  = Chem.MolFromSmarts('[Cl]-[*]')
    smarts_pattern2  = Chem.MolFromSmarts('[c;X3](c)(c)-[CH3]')
    smarts_pattern3  = Chem.MolFromSmarts('[O]=C-[C]-O')
    smarts_pattern4_1= Chem.MolFromSmarts('c:c-c:c')
    smarts_pattern4_2= Chem.MolFromSmarts('c:c:c:c')
    smarts_pattern5  = Chem.MolFromSmarts('[#6]-[#7]=[#8]')
    smarts_pattern6  = Chem.MolFromSmarts('[C,c][CH3]')
    smarts_pattern7  = Chem.MolFromSmarts('I-c:c:c')
    smarts_pattern8  = Chem.MolFromSmarts('[C,c]:[C,c]-[F]')
    smarts_pattern9_1= Chem.MolFromSmarts('O-c:c-C')
    smarts_pattern9_2= Chem.MolFromSmarts('O-c:[cH]:c')
    smarts_pattern10 = Chem.MolFromSmarts('[C,c][C;H2][C,c]')
    smarts_pattern11 = Chem.MolFromSmarts('S-[C,c]')
    smarts_pattern12 = Chem.MolFromSmarts('C-O-C=O')
    smarts_pattern13 = Chem.MolFromSmarts('[C](=O)(O)[C,c]')
    smarts_pattern14 = Chem.MolFromSmarts('[C;H1,H2](O)[C,c]')
    smarts_pattern15 = Chem.MolFromSmarts('Br-c:c-O')
    smarts_pattern16 = Chem.MolFromSmarts('O-[CH3]')
    smarts_pattern17 = Chem.MolFromSmarts('[C;H1,H2](=O)[C,c]')
    smarts_pattern18 = Chem.MolFromSmarts('C=O')
    smarts_pattern19 = Chem.MolFromSmarts('c:c-C=O')
    smarts_pattern20 = Chem.MolFromSmarts('[Br][C,c]')
    smarts_pattern21 = Chem.MolFromSmarts('N')
    smarts_pattern22 = Chem.MolFromSmarts('Br-c:c-Br')

    names = []  # FIX collect names
    # presence/flags
    results3  = []  # O=C-C-O (presence)
    results12 = []  # C-O-C=O (presence)
    results13 = []  # (C=O),(C-C),(C-O),xC (presence)
    results17 = []  # (C=O),(C-C),xC (presence)
    results18 = []  # C=O (presence)
    # counts
    molecule_counts1  = []
    molecule_counts2  = []
    molecule_counts4  = []
    molecule_counts5  = []
    molecule_counts6  = []
    molecule_counts7  = []
    molecule_counts8  = []
    molecule_counts9  = []
    molecule_counts10 = []
    molecule_counts11 = []
    molecule_counts14 = []
    molecule_counts15 = []
    molecule_counts16 = []
    molecule_counts19 = []
    molecule_counts20 = []
    molecule_counts21 = []
    molecule_counts22 = []

    with st.spinner('CALCULATING DESCRIPTORS (STEP 1 OF 3)...'):
        time.sleep(1)

        for _, row in data.iterrows():
            molecule_name = row.iloc[0]
            smiles = str(row.iloc[smiles_col_pos]).strip()
            if not smiles or pd.isna(smiles):
                continue
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue

            names.append(molecule_name)

            # presence
            fragment3  = mol.HasSubstructMatch(smarts_pattern3)
            fragment12 = mol.HasSubstructMatch(smarts_pattern12)
            fragment13 = mol.HasSubstructMatch(smarts_pattern13)
            fragment17 = mol.HasSubstructMatch(smarts_pattern17)
            fragment18 = mol.HasSubstructMatch(smarts_pattern18)

            # 9_2 filtered: unique oxygen atoms
            matches_9_2 = mol.GetSubstructMatches(smarts_pattern9_2)
            filtered_matches_oc = []
            used_oxygen_atoms = set()
            for match in matches_9_2:
                for atom_idx in match:
                    atom = mol.GetAtomWithIdx(atom_idx)
                    if atom.GetSymbol() == 'O' and atom_idx not in used_oxygen_atoms:
                        filtered_matches_oc.append(match)
                        used_oxygen_atoms.add(atom_idx)
                        break

            # 15 filtered: non-overlapping indices
            matches_15 = mol.GetSubstructMatches(smarts_pattern15)
            non_overlapping_matches_br_cc_o = []
            used_indices_br_cc_o = set()
            for match in matches_15:
                if not any(idx in used_indices_br_cc_o for idx in match):
                    non_overlapping_matches_br_cc_o.append(match)
                    used_indices_br_cc_o.update(match)

            # 19 filtered: unique central pair (1,2)
            matches_19 = mol.GetSubstructMatches(smarts_pattern19)
            used_central_pair_ccco = set()
            filtered_matches_ccco = []
            for match in matches_19:
                central_pair = (match[1], match[2])
                if central_pair not in used_central_pair_ccco:
                    filtered_matches_ccco.append(match)
                    used_central_pair_ccco.add(central_pair)

            # 22 filtered: non-overlapping indices (FIX: used_indices_br)
            matches_22 = mol.GetSubstructMatches(smarts_pattern22)
            non_overlapping_matches_br = []
            used_indices_br = set()
            for match in matches_22:
                if not any(idx in used_indices_br for idx in match):
                    non_overlapping_matches_br.append(match)
                    used_indices_br.update(match)

            # counts
            c1  = len(mol.GetSubstructMatches(smarts_pattern1))
            c2  = len(mol.GetSubstructMatches(smarts_pattern2))
            c4_1= len(mol.GetSubstructMatches(smarts_pattern4_1))/4
            c4_2= len(mol.GetSubstructMatches(smarts_pattern4_2))/2

            # ring adjust
            ring_info = mol.GetRingInfo()
            rings = ring_info.AtomRings()
            connected = False
            if len(rings) >= 2:
                ring1_atoms = set(rings[0])
                for i in range(1, len(rings)):
                    ring2_atoms = set(rings[i])
                    if ring1_atoms.intersection(ring2_atoms):
                        connected = True
                        break
            if connected:
                c4_2 -= 2
            c4 = c4_1 + c4_2

            c5  = len(mol.GetSubstructMatches(smarts_pattern5))
            c6  = len(mol.GetSubstructMatches(smarts_pattern6))
            c7  = len(mol.GetSubstructMatches(smarts_pattern7))/2
            c8  = len(mol.GetSubstructMatches(smarts_pattern8))/2
            c9_1= len(mol.GetSubstructMatches(smarts_pattern9_1))
            c9_2= len(filtered_matches_oc)
            c9  = c9_1 + c9_2
            c10 = len(mol.GetSubstructMatches(smarts_pattern10))
            c11 = len(mol.GetSubstructMatches(smarts_pattern11))
            c14 = len(mol.GetSubstructMatches(smarts_pattern14))
            c15 = len(non_overlapping_matches_br_cc_o)
            c16 = len(mol.GetSubstructMatches(smarts_pattern16))
            c19 = len(filtered_matches_ccco)/2
            c20 = len(mol.GetSubstructMatches(smarts_pattern20))
            c21 = len(mol.GetSubstructMatches(smarts_pattern21))
            c22 = len(non_overlapping_matches_br)

            # append
            results3.append(int(fragment3))
            results12.append(int(fragment12))
            results13.append(int(fragment13))
            results17.append(int(fragment17))
            results18.append(int(fragment18))

            molecule_counts1.append(int(c1))
            molecule_counts2.append(int(c2))
            molecule_counts4.append(int(c4))
            molecule_counts5.append(int(c5))
            molecule_counts6.append(int(c6))
            molecule_counts7.append(int(c7))
            molecule_counts8.append(int(c8))
            molecule_counts9.append(int(c9))
            molecule_counts10.append(int(c10))
            molecule_counts11.append(int(c11))
            molecule_counts14.append(int(c14))
            molecule_counts15.append(int(c15))
            molecule_counts16.append(int(c16))
            molecule_counts19.append(int(c19))
            molecule_counts20.append(int(c20))
            molecule_counts21.append(int(c21))
            molecule_counts22.append(int(c22))

    descriptors_total = pd.DataFrame({
        'NAME': names,
        'Cl-C': molecule_counts1,
        '(C=C),(C-C),(C-C),xC': molecule_counts2,
        'O=C-C-O': results3,
        'C=C-C=C': molecule_counts4,
        'C-N=O': molecule_counts5,
        '(C-C),xC': molecule_counts6,
        'I-C-C=C': molecule_counts7,
        'C-C-F': molecule_counts8,
        'C-C=C-O': molecule_counts9,
        '(C-C),(C-C),xC': molecule_counts10,
        'S-C': molecule_counts11,
        'C-O-C=O': results12,
        '(C=O),(C-C),(C-O),xC': results13,
        '(C-C),(C-O),xC': molecule_counts14,
        'Br-C-C-O': molecule_counts15,
        '(C-O),xC': molecule_counts16,
        '(C=O),(C-C),xC': results17,
        'C=O': results18,
        'C=C-C=O': molecule_counts19,
        '(Br-C),xBr': molecule_counts20,
        'N': molecule_counts21,
        'Br-C=C-Br': molecule_counts22,
    })

    # Return ordered SMILES list if you need it
    return descriptors_total, None

# ============== Reorder to model descriptor order ======
def reading_reorder(data: pd.DataFrame, loaded_desc):
    df_selected = data[loaded_desc]
    df_selected = df_selected.reindex(columns=loaded_desc)
    df_selected = df_selected.fillna(0)
    df_selected = ensure_unique_cols(df_selected)

    # Build an ID series (first column of original data) to use as labels
    df_id = data.reset_index(drop=True)
    id_series = df_id.iloc[:, 0].astype(str)
    return df_selected, id_series

# ============== Normalization placeholders ============
def normalize_data(train_data, test_data):
    # Plug your real scaler here if needed. Kept identity to match your original.
    df_train_normalized = pd.DataFrame(train_data).copy()
    df_test_normalized  = pd.DataFrame(test_data).copy()
    return df_train_normalized, df_test_normalized

# ============== Applicability Domain ==================
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def applicability_domain(x_test_normalized, x_train_normalized, y_train):
    X_train = x_train_normalized.values
    X_test  = x_test_normalized.values

    # Leverage
    hat_train = X_train @ np.linalg.inv(X_train.T @ X_train) @ X_train.T
    leverage_train = np.diagonal(hat_train).ravel()

    hat_test = X_test @ np.linalg.inv(X_train.T @ X_train) @ X_test.T
    leverage_test = np.diagonal(hat_test).ravel()

    # Std residuals for training (using linear regression fitted on train)
    lr = LinearRegression()
    lr.fit(x_train_normalized, y_train)
    y_pred_train = lr.predict(x_train_normalized)
    std_dev_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    std_residual_train = (y_train - y_pred_train) / (std_dev_train if std_dev_train != 0 else 1.0)
    std_residual_train = np.ravel(std_residual_train)

    # h_critical
    h3 = 3 * ((x_train_normalized.shape[1] + 1) / x_train_normalized.shape[0])
    h_results = [val < h3 for val in leverage_test]

    return h_results, leverage_train, leverage_test, std_residual_train

# ============== Confidence color ======================
def get_color(confidence):
    if confidence in ("HIGH", "Inside AD"):
        return 'green'
    elif confidence == "MEDIUM":
        return 'yellow'
    else:
        return 'red'

# ============== Predictions wrapper ===================
def predictions(loaded_model, loaded_desc, df_test_normalized, df_train_normalized, data, mean_value):
    scores = []
    h_values = []
    std_resd = []

    # Robust ID index (FIX)
    idx = data['ID'] if 'ID' in data.columns else data.iloc[:, 0].astype(str)

    with st.spinner('CALCULATING PREDICTIONS (STEP 2 OF 3)...'):
        time.sleep(1)
        X = df_test_normalized[loaded_desc]
        preds = loaded_model.predict(X)
        scores.append(preds)

        # synthetic residuals vs mean (as in your original)
        y_pred_test = preds
        y_test = np.full_like(y_pred_test, mean_value, dtype=float)
        std_dev_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
        std_residual_test = (y_test - y_pred_test) / (std_dev_test if std_dev_test != 0 else 1.0)
        std_residual_test = std_residual_test.ravel()
        std_resd.append(std_residual_test)

        # AD based on training
        y_train = data_train['pLC50']
        h_results, leverage_train, leverage_test, std_residual_train = applicability_domain(
            df_test_normalized, df_train_normalized, y_train
        )
        h_values.append(h_results)

        # Final tables
        dataframe_pred = pd.DataFrame(scores).T
        dataframe_pred.index = idx
        dataframe_pred.rename(columns={0: "pLC50"}, inplace=True)

        dataframe_std = pd.DataFrame(std_resd).T
        dataframe_std.index = idx

        h_final = pd.DataFrame(h_values).T
        h_final.index = idx
        h_final.rename(columns={0: "Confidence"}, inplace=True)

        std_ensemble = dataframe_std.iloc[:, 0]
        std_ad_calc = (std_ensemble >= 3) | (std_ensemble <= -3)
        std_ad_calc = std_ad_calc.replace({True: 'Outside AD', False: 'Inside AD'})

        final_file = pd.concat([std_ad_calc, h_final, dataframe_pred], axis=1)
        final_file.rename(columns={0: "Std_residual"}, inplace=True)

        # Assign textual confidence
        final_file.loc[(final_file["Confidence"] == True) & (final_file["Std_residual"] == 'Inside AD'),  'Confidence'] = 'HIGH'
        final_file.loc[(final_file["Confidence"] == True) & (final_file["Std_residual"] == 'Outside AD'), 'Confidence'] = 'LOW'
        final_file.loc[(final_file["Confidence"] == False) & (final_file["Std_residual"] == 'Outside AD'),'Confidence'] = 'LOW'
        final_file.loc[(final_file["Confidence"] == False) & (final_file["Std_residual"] == 'Inside AD'), 'Confidence'] = 'MEDIUM'

        # Style
        df_no_duplicates = final_file[~final_file.index.duplicated(keep='first')]
        styled_df = df_no_duplicates.style.apply(
            lambda row: [f"background-color: {get_color(row['Confidence'])}" for _ in row],
            subset=["Confidence"], axis=1
        )

        return final_file, styled_df, leverage_train, std_residual_train, leverage_test, std_residual_test

# ============== Williams plot limits ==================
def calculate_wp_plot_limits(leverage_train, std_residual_train, x_std_max=4, x_std_min=-4):
    with st.spinner('CALCULATING APPLICABILITY DOMAIN (STEP 3 OF 3)...'):
        time.sleep(1)

        # y limits
        x_lim_max_std = x_std_max if std_residual_train.max() < 4 else round(float(std_residual_train.max())) + 1
        x_lim_min_std = x_std_min if std_residual_train.min() > -4 else round(float(std_residual_train.min())) - 1

        # h critical
        n = len(leverage_train)
        p = df_train_normalized.shape[1]
        h_value = 3 * (p + 1) / n
        h_critical = round(h_value, 4)

        # x limits leverage
        x_lim_max_lev = h_critical + h_critical*0.5 if leverage_train.max() < h_critical else float(leverage_train.max())*1.1
        x_lim_min_lev = 0.0  # FIX: clamp at 0 (leverage is >= 0)

        return x_lim_max_std, x_lim_min_std, h_critical, x_lim_max_lev, x_lim_min_lev

# ============== Williams plot (FIX: 1-D coercion) =====
def williams_plot(leverage_train, leverage_test, std_residual_train, std_residual_test, id_list_1,
                  x_lim_max_std, x_lim_min_std, h_critical, x_lim_max_lev, x_lim_min_lev,
                  show_plot=True, save_plot=False, filename=None, add_title=False, title=None):

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=_as_1d_array(leverage_train),
        y=_as_1d_array(std_residual_train),
        mode='markers',
        marker=dict(color='cornflowerblue', size=10, line=dict(width=1, color='black')),
        name='Training'
    ))

    fig.add_trace(go.Scatter(
        x=_as_1d_array(leverage_test),
        y=_as_1d_array(std_residual_test),
        mode='markers',
        marker=dict(color='orange', size=10, line=dict(width=1, color='black')),
        name='Prediction',
        text=_as_1d_text(id_list_1),          # FIX: 1-D list[str]
        hoverinfo='text'
    ))

    # Lines
    fig.add_shape(type='line', x0=h_critical, y0=x_lim_min_std, x1=h_critical, y1=x_lim_max_std,
                  line=dict(color='black', dash='dash'))
    fig.add_shape(type='line', x0=x_lim_min_lev, y0=3, x1=x_lim_max_lev, y1=3,
                  line=dict(color='black', dash='dash'))
    fig.add_shape(type='line', x0=x_lim_min_lev, y0=-3, x1=x_lim_max_lev, y1=-3,
                  line=dict(color='black', dash='dash'))

    # Zones
    fig.add_shape(type='rect', x0=x_lim_min_lev, y0=x_lim_min_std, x1=h_critical, y1=-3,
                  fillcolor='lightgray', opacity=0.4, line_width=0)
    fig.add_shape(type='rect', x0=x_lim_min_lev, y0=3, x1=h_critical, y1=x_lim_max_std,
                  fillcolor='lightgray', opacity=0.4, line_width=0)
    fig.add_shape(type='rect', x0=h_critical, y0=x_lim_min_std, x1=x_lim_max_lev, y1=-3,
                  fillcolor='lightgray', opacity=0.4, line_width=0)
    fig.add_shape(type='rect', x0=h_critical, y0=3, x1=x_lim_max_lev, y1=x_lim_max_std,
                  fillcolor='lightgray', opacity=0.4, line_width=0)

    # Labels
    fig.add_annotation(x=(h_critical + x_lim_min_lev)/2, y=-3.5, text='Outlier zone', showarrow=False, font=dict(size=15))
    fig.add_annotation(x=(h_critical + x_lim_min_lev)/2, y=3.5,  text='Outlier zone', showarrow=False, font=dict(size=15))
    fig.add_annotation(x=(h_critical + x_lim_max_lev)/2, y=-3.5, text='Outlier zone', showarrow=False, font=dict(size=15))
    fig.add_annotation(x=(h_critical + x_lim_max_lev)/2, y=3.5,  text='Outlier zone', showarrow=False, font=dict(size=15))

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
    #if show_plot:
        #st.plotly_chart(fig, use_container_width=True)
    return fig

# ============== File download helper ==================
def filedownload1(df: pd.DataFrame):
    csv = df.to_csv(index=True, header=True)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="ml_toxicity_t_pyriformis_pLC50_results.csv">Download CSV File with results</a>'
    return href

# ============== Highlight substructure images =========
FRAGMENT_EFFECTS = {
  'Cl-C': ('[Cl]-[*]', 1, 0.461),
 '(C=C),(C-C),(C-C),xC': ('[c;X3](c)(c)-[CH3]', -1, 0.24),
 'O=C-C-O': ('[O]=C-[C]-O', 1, 1.1),
 'C=C-C=C': ('c:c-c:c', 1, 0.408),
 'C-N=O': ('[#6]-[#7]=[#8]', 1, 0.328),
 '(C-C),xC': ('[C,c][CH3]', 1, 0.385),
 'I-C-C=C': ('I-c:c:c', 1, 0.847),
 'C-C-F': ('[C,c]:[C,c]-[F]', 1, 0.324),
 'C-C=C-O': ('O-c:c-C', 1, 0.156),
 '(C-C),(C-C),xC': ('[C,c][C;H2][C,c]', 1, 0.316),
 'S-C': ('S-[C,c]', -1, 1.684),
 'C-O-C=O': ('C-O-C=O', 1, 1.562),
 '(C=O),(C-C),(C-O),xC': ('[C](=O)(O)[C,c]', -1, 0.645),
 '(C-C),(C-O),xC': ('[C;H1,H2](O)[C,c]', -1, 0.324),
 'Br-C-C-O': ('Br-c:c-O', 1, 0.293),
 '(C-O),xC': ('O-[CH3]', 1, 0.141),
 '(C=O),(C-C),xC': ('[C;H1,H2](=O)[C,c]', 1, 0.652),
 'C=O': ('C=O', -1, 0.916),
 'C=C-C=O': ('c:c-C=O', 1, 0.603),
 '(Br-C),xBr': ('[Br][C,c]', 1, 0.71),
 'N': ('N', 1, 0.168),
 'Br-C=C-Br': ('Br-c:c-Br', -1, 0.516)
}


RED   = (1.00, 0.00, 0.00)
GREEN = (0.00, 0.60, 0.00)
def _tox_phrase(sign: int) -> str:
    return "tox increase" if sign >= 0 else "tox decrease"
    
def _find_fragment_matches(mol: Chem.Mol, smarts: str) -> List[Tuple[int, ...]]:
    patt = Chem.MolFromSmarts(smarts)
    if not patt:
        return []
    return list(mol.GetSubstructMatches(patt))

def _atoms_bonds_from_match(mol: Chem.Mol, match: Tuple[int, ...]):
    aset = set(match)
    bonds = []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i in aset and j in aset:
            bonds.append(b.GetIdx())
    return list(aset), bonds

def make_fragment_grid_for_molecule(
    smiles: str,
    fragment_effects: Dict[str, Tuple[str, int, float]],  # label -> (SMARTS, sign, |coef|)
    per_row: int = 3,
    tile_size=(260, 260),
    highlight_all_occurrences: bool = True,
    sort_by_magnitude: bool = True,
    show_only_present: bool = True,   # ← NEW
):
    """Repeat the SAME molecule; one tile per fragment in FRAGMENT_EFFECTS.
       If show_only_present=True, skip fragments with no matches in the molecule."""
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None
    Chem.rdDepictor.Compute2DCoords(mol)

    items = list(fragment_effects.items())
    if sort_by_magnitude:
        items.sort(key=lambda kv: kv[1][2], reverse=True)  # sort by |coef| desc

    mols, legends = [], []
    highlightAtomLists, highlightBondLists = [], []
    highlightAtomColors, highlightBondColors = [], []

    for label, (smarts, sign, mag) in items:
        matches = _find_fragment_matches(mol, smarts)
        if not matches and show_only_present:
            continue  # skip “no hit” tiles entirely

        color = RED if sign >= 0 else GREEN

        if not matches:  # show “no hit” tile only if show_only_present=False
            mols.append(mol)
            legends.append(f"{label} (no hit, |β|={mag:.3g}, {_tox_phrase(sign)})")
            highlightAtomLists.append([]); highlightBondLists.append([])
            highlightAtomColors.append({}); highlightBondColors.append({})
            continue

        # highlight all occurrences of this fragment
        atoms_all, bonds_all = set(), set()
        for m in matches:
            a_idxs, b_idxs = _atoms_bonds_from_match(mol, m)
            atoms_all.update(a_idxs); bonds_all.update(b_idxs)
        a_list = list(atoms_all); b_list = list(bonds_all)
        a_colors = {i: color for i in a_list}
        b_colors = {i: color for i in b_list}

        mols.append(mol)
        legends.append(f"{label} (|β|={mag:.3g}, {_tox_phrase(sign)})")
        highlightAtomLists.append(a_list); highlightBondLists.append(b_list)
        highlightAtomColors.append(a_colors); highlightBondColors.append(b_colors)

    if not mols:
        return None

    img = Draw.MolsToGridImage(
        mols,
        molsPerRow=per_row,
        subImgSize=tile_size,
        legends=legends,
        useSVG=False,
        highlightAtomLists=highlightAtomLists,
        highlightBondLists=highlightBondLists,
        highlightAtomColors=highlightAtomColors,
        highlightBondColors=highlightBondColors,
    )
    return img


# ============== Load model & descriptors ==============
data_train = pd.read_csv("data/" + "data_Tpyriformis_22var_original_training.csv")
data_train = ensure_unique_cols(data_train)
mean_value = data_train['pLC50'].mean()

loaded_model = pickle.load(open("models/" + "ml_model_tetrahymena_pyriformis_structural.pickle", 'rb'))
loaded_desc  = pickle.load(open("models/" + "ml_descriptor_tetrahymena_pyriformis_structural.pickle", 'rb'))

# example SMARTS to highlight
highlight_mols = {
    'O=C-C-O': Chem.MolFromSmarts('[O]=C-[C]-O'),
    'C-C-F': Chem.MolFromSmarts('[C,c]:[C,c]-[F]'),
    '(C=O),(C-C),xC': Chem.MolFromSmarts('[C;H1,H2](=O)[C,c]')
}

# ============== RUN paths =============================
if uploaded_file_1 is not None:
    run = st.button("RUN the Model")
    if run:
        data = pd.read_csv(uploaded_file_1)
        data = ensure_unique_cols(data)

        train_data = data_train[loaded_desc]
        descriptors_total_1, _ = calc_descriptors(data, 1)

        test_data1, id_list_1 = reading_reorder(descriptors_total_1, loaded_desc)
        X_final2 = test_data1

        df_train_normalized, df_test_normalized = normalize_data(train_data, X_final2)

        final_file, styled_df, leverage_train, std_residual_train, leverage_test, std_residual_test = predictions(
            loaded_model, loaded_desc, df_test_normalized, df_train_normalized, data, mean_value
        )

        x_lim_max_std, x_lim_min_std, h_critical, x_lim_max_lev, x_lim_min_lev = calculate_wp_plot_limits(
            leverage_train, std_residual_train, x_std_max=4, x_std_min=-4
        )

        figure = williams_plot(
            leverage_train, leverage_test, std_residual_train, std_residual_test, id_list_1,
            x_lim_max_std, x_lim_min_std, h_critical, x_lim_max_lev, x_lim_min_lev
        )

        col1, col2 = st.columns(2)
        with col1:
            st.header("T. pyriformis")
            st.markdown("<hr style='border: 1px solid blue;'>", unsafe_allow_html=True)
            st.subheader('Predictions')
            st.write(styled_df)
        with col2:
            st.markdown("<h2 style='text-align: center; font-size: 30px;'>William's Plot (Applicability Domain)</h2>", unsafe_allow_html=True)
            st.plotly_chart(figure, use_container_width=True, key= "williams_plot_main")

        st.markdown(":point_down: **Here you can download the results for T. pyriformis MLR model**", unsafe_allow_html=True)
        st.markdown(filedownload1(final_file), unsafe_allow_html=True)

else:
    st.info('👈🏼👈🏼👈🏼   Awaiting for CSV file to be uploaded.')
    if st.button('Press to use Example CSV Dataset with smiles'):
        data = pd.read_csv("virtual_smi_example.csv")
        data = ensure_unique_cols(data)

        train_data = data_train[loaded_desc]
        descriptors_total_1, _ = calc_descriptors(data, 1)
        test_data1, id_list_1 = reading_reorder(descriptors_total_1, loaded_desc)
        X_final2 = test_data1

        df_train_normalized, df_test_normalized = normalize_data(train_data, X_final2)

        final_file, styled_df, leverage_train, std_residual_train, leverage_test, std_residual_test = predictions(
            loaded_model, loaded_desc, df_test_normalized, df_train_normalized, data, mean_value
        )

        x_lim_max_std, x_lim_min_std, h_critical, x_lim_max_lev, x_lim_min_lev = calculate_wp_plot_limits(
            leverage_train, std_residual_train, x_std_max=4, x_std_min=-4
        )

        figure = williams_plot(
            leverage_train, leverage_test, std_residual_train, std_residual_test, id_list_1,
            x_lim_max_std, x_lim_min_std, h_critical, x_lim_max_lev, x_lim_min_lev
        )

        col1, col2 = st.columns(2)
        with col1:
            st.header("T. pyriformis")
            st.markdown("<hr style='border: 1px solid blue;'>", unsafe_allow_html=True)
            st.subheader('Predictions')
            st.write(styled_df)
        with col2:
            st.markdown("<h2 style='text-align: center; font-size: 25px;'>William's Plot (Applicability Domain)</h2>", unsafe_allow_html=True)
            st.plotly_chart(figure, use_container_width=True)

        st.markdown(":point_down: **Here you can download the results for T. pyriformis MLR model**", unsafe_allow_html=True)
        st.markdown(filedownload1(final_file), unsafe_allow_html=True)

# ===== Optional: Drawn structure block (left as in your app) =====
# You can re-add your st_ketcher section here as needed, using the same fixes above.


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
on2 = st.checkbox('Use drawn structure', key="13")

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
        
        #final_file, styled_df,leverage_train,std_residual_train, leverage_test, std_residual_test= predictions(loaded_model, loaded_desc, df_test_normalized)

        final_file, styled_df, leverage_train, std_residual_train, leverage_test, std_residual_test = \
            predictions(
                loaded_model,
                loaded_desc,
                df_test_normalized,
               df_train_normalized,   # <— add
               data,                  # <— add (the input DF you’re predicting for)
               mean_value             # <— add (from training set)
          )

        
        x_lim_max_std, x_lim_min_std, h_critical, x_lim_max_lev, x_lim_min_lev = calculate_wp_plot_limits(leverage_train,std_residual_train, x_std_max=4, x_std_min=-4)
        
        #figure  = williams_plot(leverage_train, leverage_test, std_residual_train, std_residual_test,id_list_1)

        figure = williams_plot(
             leverage_train,
             leverage_test,
             std_residual_train,
             std_residual_test,
             id_list_1,
             x_lim_max_std,
             x_lim_min_std,
             h_critical,
             x_lim_max_lev,
             x_lim_min_lev
        )

        
        col1,col2 = st.columns(2)

        with col1:
            st.header("T. pyriformis")
            st.markdown("<hr style='border: 1px solid blue;'>", unsafe_allow_html=True)
            #st.header("T. pyriformis",divider='blue')
            st.subheader(r'Predictions')
            st.write(styled_df)
        with col2:
            st.markdown("<h2 style='text-align: center; font-size: 30px;'>William's Plot (Applicability Domain)</h2>", unsafe_allow_html=True)
            st.plotly_chart(figure,use_container_width=True)
            # === Fragment Grid for the DRAWN molecule ===
        st.markdown("### Fragment grid (same molecule per tile; red ↑tox, green ↓tox)")
        grid_img = make_fragment_grid_for_molecule(
            smiles_list,            # the drawn SMILES string
            FRAGMENT_EFFECTS,       # built from your coefficients
            per_row=3,
            tile_size=(260, 260),
            highlight_all_occurrences=True,
            sort_by_magnitude=True
        )
        if grid_img is not None:
            st.image(grid_img, caption="Each tile highlights a single model fragment")
        else:
            st.info("Could not parse the drawn SMILES.")

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




