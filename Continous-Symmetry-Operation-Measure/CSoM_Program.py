# =============================================================================
# Here the functions and symmetry operations are read:
# =============================================================================
import os
import numpy as np
dirname = os.path.dirname(__file__)
os.chdir(dirname)
Path_To_Symmetry_Operations = os.path.join(dirname, 'Operations')
import CSoM_functions as csom

# =============================================================================
# Inputs:
# =============================================================================

"""
Load molecular structure:
    There are two ways to load structure files. Either all files in a folder can be loaded or files can be individually selected.
    To load an entire folder: 
        Insert the path of the folder in the keyword 'mypath'. Then all files in the folder is collected into savepaths.
        All files in the folder must be .xyz files.
    To manually load individual files: 
        Write the path to a file manually into the 'savepaths' list. 
        The list can contain as many files as needed. 
        All paths must lead to an .xyz file.
"""

#Select folder where the structures are to be analysed:
mypath = r'C:\Users\Villads\Documents\Project_SymmetryDeviation\ExampleUsage_on_LnAquaIon'

all_structure = True #Default is False.
structures_selected = [
             'NdODA3.xyz',
             ]

"""
Select pointgroups for analysis. 
All point groups must be written into the 'Point_group_names' list. They ar'e then read from the 'operations folder' found in the same folder as this script. 
If all pioint groups are to be used, this can also be done. Beware, this can be time consuming.
"""
#D4d and subgroups
point_group_names =[
    'pointgroup_C2',
    'pointgroup_C3',
    'pointgroup_C4',
    'pointgroup_C5',
    'pointgroup_C6',
    
                    ]
# if all pointgroups are to be used: Remove the outcommented line here:
#point_group_names = [os.path.basename(os.path.join(Path_To_Symmetry_Operations,f)).split('.')[0] for f in os.listdir(Path_To_Symmetry_Operations) if os.path.isfile(os.path.join(Path_To_Symmetry_Operations, f))]


"""
'N_input_zaxis' is the amount of initial principle axis used to optimize on the structure. 
The more that are used the more certain the program is to finde the global minimum.
"""
N_input_zaxis = 10 #Default is 10
N_input_angles = 2 #Default is 2
"""
'Manual_Zaxis':
    If False the principle axis will be optimized. This is default. 
    If True, no optimization will occur and the .xyz file will be analysed symmetry analyzed with the orientation give below.
    give manual coordinates for Zaxis and a manual input for the rotation angle of the x,y plane. 
"""
Manual_Zaxis = False  #Default is False.
Zaxis = np.array([0,0,1]) 
Manual_Angle = False  #Default is False.
angle = 0

"""
'ManualCentering':
    if False, the first atom of .xyz file will be used as origo.
    if True, The coordinates below, x0, y0, and z0 will be used as origo.
"""
ManualCentering = False #Default is False
x0 = 0
y0 = 0
z0 = 0

"""
'ManualCentering':
    if False, the first atom of .xyz file will be used as origo.
    if True, The coordinates below, x0, y0, and z0 will be used as origo.
"""
cut_off_distance = True #Default is False.
cutoff = 300
"""
'SaveStructure' can be set to True if the generated structures from the analysis is to be saved. 
"""
SaveStructure = True #Default is True.
# =============================================================================
# Script runs:
# =============================================================================
csom.Run_CSoM(mypath,all_structure,structures_selected,point_group_names,N_input_zaxis,N_input_angles,Manual_Zaxis,Zaxis,Manual_Angle,angle,ManualCentering,x0,y0,z0,cut_off_distance,cutoff,SaveStructure)


