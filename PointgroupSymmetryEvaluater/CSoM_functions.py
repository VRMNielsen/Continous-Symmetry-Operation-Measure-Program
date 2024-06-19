import numpy as np
import math
import re
import pandas as pd
from scipy.optimize import minimize
import os
import csv
from scipy.optimize import linear_sum_assignment
from datetime import datetime
#%%
"""
Define here the path of the SymmetryOpertationLibrary
"""
dirname = os.path.dirname(__file__)
Path_To_Symmetry_Operations = os.path.join(dirname, 'Operations')
#%%
def load_xyz(structure_path):
    """
    Load two XYZ files and center them around origo, scale them and output a np.array of the scaled coordinates.
    three center types: 
        0: None
        1: Center by average positions
        2: Center specific atom
    """
    
    structure = pd.read_csv(structure_path, skiprows=2, delim_whitespace=True, names=["atom", "x", "y", "z"])
    pattern = r'[0-9]'
    atoms = (np.expand_dims((structure.atom.values), axis=1)).T[0]
    atoms = np.array([re.sub(pattern, '', atom) for atom in atoms ])
    #print(atoms)
    structure = np.concatenate((np.expand_dims((structure.x.values), axis=1), np.expand_dims((structure.y.values), axis=1), np.expand_dims((structure.z.values), axis=1)), axis=1)

    #Center with First atom in center
    structure[:,0] = structure[:,0] - structure[0][0]
    structure[:,1] = structure[:,1] - structure[0][1]
    structure[:,2] = structure[:,2] - structure[0][2]
   # print ('Structure after centering:', structure)
    return atoms, structure


def load_xyz_cutoff(structure_path,cutoff):
    """
    Load two XYZ files and center them around origo, scale them and output a np.array of the scaled coordinates.
    three center types: 
        0: None
        1: Center by average positions
        2: Center specific atom
    """
    
    structure = pd.read_csv(structure_path, skiprows=2, delim_whitespace=True, names=["atom", "x", "y", "z"])
    pattern = r'[0-9]'
    atoms = (np.expand_dims((structure.atom.values), axis=1)).T[0]
    atoms = np.array([re.sub(pattern, '', atom) for atom in atoms ])
   # print(atoms)
    structure = np.concatenate((np.expand_dims((structure.x.values), axis=1), np.expand_dims((structure.y.values), axis=1), np.expand_dims((structure.z.values), axis=1)), axis=1)

    #Center with First atom in center
    structure[:,0] = structure[:,0] - structure[0][0]
    structure[:,1] = structure[:,1] - structure[0][1]
    structure[:,2] = structure[:,2] - structure[0][2]
   # print ('Structure after centering:', structure)
    selected_atoms = [np.linalg.norm(x) < cutoff for x in structure]
    
    return atoms[selected_atoms], structure[selected_atoms]



def load_xyz_manual_center(structure_path,x0,y0,z0,cutoff):
    """
    Load two XYZ files and center them around origo, scale them and output a np.array of the scaled coordinates.
    three center types: 
        0: None
        1: Center by average positions
        2: Center specific atom
    """
    structure = pd.read_csv(structure_path, skiprows=2, delim_whitespace=True, names=["atom", "x", "y", "z"])
    atoms = (np.expand_dims((structure.atom.values), axis=1))
    structure = np.concatenate((np.expand_dims((structure.x.values), axis=1), np.expand_dims((structure.y.values), axis=1), np.expand_dims((structure.z.values), axis=1)), axis=1)
  #  print (atoms)
    #Center with First atom in center
    structure[:,0] = structure[:,0] - x0
    structure[:,1] = structure[:,1] - y0
    structure[:,2] = structure[:,2] - z0
   # print ('Structure after centering:', structure)
    selected_atoms = [np.linalg.norm(x) < cutoff for x in structure]
    return atoms.T[0][selected_atoms], structure[selected_atoms]

def ReadSymmetryFile(PointGroup,Path_to_folder):
    # Reading the lists from the CSV file
    read_name_list = []
    read_matrix_list = []
    
    with open(os.path.join(Path_to_folder,PointGroup + '.csv'), 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            name = row[0]
            matrix_str = [row[1],row[2],row[3],row[4],row[5],row[6],row[7],row[8],row[9]]
            matrix_str = np.array(matrix_str, dtype=float).reshape(3,3)
    
            read_name_list.append(name)
            read_matrix_list.append(matrix_str)
    
    print('Read data from CSV:', PointGroup)
    return read_name_list,read_matrix_list

def rate_values(original_list):
    sorted_list = sorted(original_list)
    ratings = {}
    for i, value in enumerate(sorted_list):
        ratings[value] = i
    
    ordered_list = [ratings[value] for value in original_list]
    return ordered_list


def rotation_matrix(axis, angle):
    # Normalize the axis to ensure it's a unit vector
    axis = axis / np.linalg.norm(axis)

    # Convert the angle from degrees to radians
    angle_rad = np.deg2rad(angle)

    # Compute the cross product matrix of the rotation axis
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])

    # Compute the rotation matrix using Rodrigues' rotation formula
    R = np.eye(3) + np.sin(angle_rad) * K + (1 - np.cos(angle_rad)) * np.dot(K, K)
    return R.T

def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix.T

def symmetry_deviation_calculator(ideel_structure, disordered_structure):
    """
    Calculate symmetry deviation (𝜎) for two structures with equal amount of atoms.
    The 𝜎 is a number from 0 to 100, where 0 is completelly symmetric.
    This formula has been used from the papers:
    - Continuous symmetry maps and shape classification. The case of six-coordinated metal compounds
    - Shape maps and polyhedral interconversion paths in transition metal chemistry
    - Stereochemistry of Compounds with Coordination Number Ten
    The function needs two structure which is already aligned e.g. atom1 has to be the first atom in the list for both, atom2 has to be number 2 etc.
    The alignment can be done in mercury with the use of the overlay function.
    """

    mean_ideel = np.mean(ideel_structure, axis=0)
    diff = ideel_structure - disordered_structure
    sq_distances = np.linalg.norm(diff, axis=1)**2
    denominator = np.linalg.norm(ideel_structure - mean_ideel, axis=1)**2
    
    # Avoid division by zero
    non_zero_denominator = np.where(denominator != 0, denominator, 1)
    # Calculate sigma using vectorized operations
    sigma = np.sum(sq_distances / non_zero_denominator)


    return 100 * sigma / len(ideel_structure)

def find_vector_between_points(point1, point2):
    # Convert points to numpy arrays
    point1 = np.array(point1)
    point2 = np.array(point2)
    # Compute the vector between the points
    vector = point2 - point1
    return vector

def fibonacci_sphere(samples=5):
    print('Generating {} input vectors that are equally spaced on the unit sphere'.format(samples))

    points = np.empty((samples, 3))

    phi = math.pi * (math.sqrt(5.) - 1.)  # golden angle in radians
    indices = np.arange(samples, dtype=float)
    
    y = 1 - (indices / float(samples - 1))
    radius = np.sqrt(1 - y**2)
    theta = phi * indices  # golden angle increment
    points[:, 0] = np.cos(theta) * radius
    points[:, 1] = y
    points[:, 2] = np.sin(theta) * radius
    return points

def find_zaxis(structure,inp_atoms,SymmetryOperations, N_input_zaxis = 10 ,N_input_angles = 2):
    def point_group_deviation_sphere(structure,SymmetryOperations):
        structure = np.array(structure)
        SymmetryOperations = np.array(SymmetryOperations)
        
        deviations = []
        for index,operation in enumerate(SymmetryOperations):
        #    operation = sym[0]
            operated_structure =  np.dot(structure, operation)
            operated_structure_permuted,operated_atoms_permuted,permutation = best_permutation_multiple_atoms(structure,inp_atoms,operated_structure)
            deviations.append(symmetry_deviation_calculator(operated_structure_permuted, structure))
        return deviations
    
    points= fibonacci_sphere(samples=N_input_zaxis)
    # Generate angles
    angles_of_rotation = np.linspace(0, 180, N_input_angles)
    new_array = np.repeat(points, len(angles_of_rotation), axis=0)
    points = np.hstack((new_array, np.tile(angles_of_rotation, len(points))[:, np.newaxis]))
    
    results = []
    results_fun = []
    results_zaxises = []
    for index,optimization in enumerate(points):
        initial_params = list(optimization)
        def vector_opt_func(params):
            a, b, c, angle = params
            
            vector = find_vector_between_points([0,0,0],[a,b,c])
            RM = rotation_matrix_from_vectors([0, 0, 1], vector)
            
            new_vector = np.dot(vector, RM.T)
            
            rot_new_vector = rotation_matrix(new_vector, angle)
            
            structure_new_z_axis = np.dot(structure,np.matmul(RM.T,rot_new_vector))
            return np.mean(point_group_deviation_sphere(structure_new_z_axis, SymmetryOperations))
        
        result = minimize(vector_opt_func, initial_params, method='Powell')
        results.append(result)
        results_zaxises.append(result.x[0:3]/np.linalg.norm(result.x[0:4]))
        print('sigma deviation of {} optimization'.format(index),result.fun)
        results_fun.append(result.fun)

    optimal_zaxis = results[results_fun.index(min(results_fun))].x[0:3]
    optimal_zaxis = optimal_zaxis / np.linalg.norm(optimal_zaxis)
    optimal_zrot_angle = results[results_fun.index(min(results_fun))].x[-1]
    
    vector = find_vector_between_points([0,0,0],optimal_zaxis)
    RM = rotation_matrix_from_vectors([0, 0, 1], vector)
    new_vector = np.dot(vector, RM.T)
    rot_new_vector = rotation_matrix(new_vector, optimal_zrot_angle)
    optimal_rotation_matrix = np.matmul(RM.T,rot_new_vector)
    structure_new_z_axis = np.dot(structure,optimal_rotation_matrix)
    return structure_new_z_axis,optimal_rotation_matrix,optimal_zaxis, optimal_zrot_angle, results_zaxises


def find_angle(structure,inp_atoms,SymmetryOperations,Zaxis, N_input_angles = 2):

    def point_group_deviation_sphere(structure,SymmetryOperations):
        structure = np.array(structure)
        SymmetryOperations = np.array(SymmetryOperations)
        
        deviations = []
        for index,operation in enumerate(SymmetryOperations):
        #    operation = sym[0]
            operated_structure =  np.dot(structure, operation)
            operated_structure_permuted,operated_atoms_permuted,permutation = best_permutation_multiple_atoms(structure,inp_atoms,operated_structure)
            deviations.append(symmetry_deviation_calculator(operated_structure_permuted, structure))
        return deviations
    
    # Generate angles
    angles_of_rotation = np.linspace(0, 180, N_input_angles)
    results_fun = []
    for index,optimization in enumerate(angles_of_rotation):
        initial_params = [optimization]
        def angle_opt_func(params):
            angle = params

            if np.array_equal(Zaxis, np.array([0, 0, 1])):
                rot_new_vector = rotation_matrix(Zaxis, angle)
                structure_new_z_axis = np.dot(structure,rot_new_vector)
            else:
                RM = rotation_matrix_from_vectors([0, 0, 1], Zaxis)
                new_vector = np.dot(Zaxis, RM.T)
                rot_new_vector = rotation_matrix(new_vector, angle)
                structure_new_z_axis = np.dot(structure,np.matmul(RM.T,rot_new_vector))        
                
            return sum(point_group_deviation_sphere(structure_new_z_axis,SymmetryOperations)) / len(SymmetryOperations)
        
        result = minimize(angle_opt_func, initial_params, method='SLSQP')

        print('sigma deviation of {} optimization'.format(index),result.fun)
        results_fun.append(result.fun)

    optimal_zrot_angle = results_fun[results_fun.index(min(results_fun))]

    if np.array_equal(Zaxis, np.array([0, 0, 1])):
        rot_new_vector = rotation_matrix(Zaxis, optimal_zrot_angle)
        structure_new_z_axis = np.dot(structure,rot_new_vector)
    else:
        RM = rotation_matrix_from_vectors([0, 0, 1], Zaxis)
        new_vector = np.dot(Zaxis, RM.T)
        rot_new_vector = rotation_matrix(new_vector, optimal_zrot_angle)
        optimal_rotation_matrix = np.matmul(RM.T,rot_new_vector)
        structure_new_z_axis = np.dot(structure,optimal_rotation_matrix)
    return structure_new_z_axis, optimal_zrot_angle

def best_permutation(A, B):
    A = np.array(A)
    B = np.array(B)

    # Compute the pairwise distances between A and B
    distances = np.sqrt(np.sum((A[:, np.newaxis] - B) ** 2, axis=-1))

    # Solve the linear assignment problem to find the optimal permutation
    row_indices, col_indices = linear_sum_assignment(distances)

    # Permute the elements in B
    B_permuted = B[col_indices]
    
    return B_permuted, list(col_indices)

def best_permutation_multiple_atoms(structure,inp_atoms,operated_structure):

    unique_elements, unique_indices = np.unique(inp_atoms, return_index=True)
    full_index_list = list(range(0, len(inp_atoms)))
    atomsets=[]
    atom_indexes = []
    atomsets_operated = []
    
    operated_structure_permuted = []
    new_atom_indexes = []
    
    # Adjust the values to ensure no number is more than 1 higher than the others
    for element in unique_elements[rate_values(unique_indices)]:
        #collecting and filtering the sets of atoms that correspond to the elements in unique element
        filtering_for_element =  [i == element for i in inp_atoms]
        atomsets.append(list(structure[filtering_for_element]))
        atom_indexes.append( list(np.array(full_index_list)[filtering_for_element]))
        atomsets_operated.append(list(operated_structure[filtering_for_element]))
    
    for atoms,A, B in zip(atom_indexes,atomsets,atomsets_operated):
        B_permuted, perm = best_permutation(A, B)
        B = np.array(B)
        atoms = np.array(atoms)
        new_atom_indexes.append(list(atoms[perm]))
                
    for perm,new_perm in zip(atom_indexes,new_atom_indexes):
        for i,j in zip(perm,new_perm):
            full_index_list[i] = j
        
    operated_atoms_permuted = inp_atoms[full_index_list]
    operated_structure_permuted =  operated_structure[full_index_list]
    
    return np.array(operated_structure_permuted),np.array(operated_atoms_permuted), full_index_list



def point_group_deviation(structure,inp_atoms,point_group_symmetry,Point_group_name, savepath,point_group_symmetry_names, SaveStructure = True):
    
    deviations = []
    
    directory_path = os.path.join(os.path.dirname(savepath),  'SymmetryOperatedStructures')
    if SaveStructure:
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            
        #Updating directory path to the specific operation applied:
        directory_path = os.path.join(directory_path, Point_group_name + '_on_' + os.path.basename(savepath).split('.')[0])
        
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            
        new_coords = structure.astype(str)
        new_coords = np.insert(new_coords, 0, inp_atoms , axis=1)
        #Save the structure pefore it has been operated: 
        print(new_coords)
        np.savetxt(os.path.join(directory_path, os.path.basename(savepath).split('.')[0] + '_o' + '.xyz'), new_coords, delimiter='\t', fmt='%s', header=str(len(structure))+'\n', comments='')  

        #Saving all in one file
        with open(os.path.join(directory_path, os.path.basename(savepath).split('.')[0] + '_all' + '.xyz') , 'a') as file:
            # Write the new data using np.savetxt
            np.savetxt(file, new_coords, delimiter='\t', fmt='%s', header=str(len(new_coords) * (len(point_group_symmetry)+1)) + '\n', comments='')

    for index,operation in enumerate(point_group_symmetry):

        operated_structure =  np.dot(structure, operation)
        operated_structure_permuted,operated_atoms_permuted,permutation = best_permutation_multiple_atoms(structure,inp_atoms,operated_structure)
        
        deviations.append(symmetry_deviation_calculator(operated_structure_permuted, structure))
        print(symmetry_deviation_calculator(operated_structure_permuted, structure))
        
        if SaveStructure:
            new_coords = operated_structure_permuted.astype(str)
            new_coords = np.insert(new_coords, 0, operated_atoms_permuted , axis=1)
            output_file_path = os.path.join(directory_path, os.path.basename(savepath).split('.')[0] + '_SymOp{}_{}'.format(index+1,point_group_symmetry_names[index]) + '.xyz')
            np.savetxt(output_file_path, new_coords, delimiter='\t', fmt='%s', header=str(len(new_coords))+'\n', comments='')  
            #print('Operation{} has been applied. structure saved to: \n '.format(operation), output_file_path)
            
            #Saving all in one file
            with open(os.path.join(directory_path, os.path.basename(savepath).split('.')[0] + '_all' + '.xyz') , 'a') as file:
                # Write the new data using np.savetxt
                np.savetxt(file, new_coords, delimiter='\t', fmt='%s', comments='')
        
    return deviations


def Save_Results(savepath,Point_group_name,point_group_symmetry_names, deviations,
                 Manual_Zaxis, N_input_zaxis, Zaxis, angle, Rotation_Matrix, Manual_Angle, N_input_angles, ManualCentering, x0,y0,z0):
    #Create folder for all operations
    directory_path = os.path.dirname(savepath)+ '\SymetryOperationDeviations'
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    #Create specific folder for this structure
    directory_path = directory_path + '\{}'.format(os.path.basename(savepath).split('.')[0])
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        
    # Combine the data into a list of tuples
    data = list(zip(point_group_symmetry_names, deviations))
    # Save the data to a file
    with open(directory_path + '\\' + '{}_on_'.format(Point_group_name) + os.path.basename(savepath).split('.')[0] + '.csv', 'w') as file:
        writer = csv.writer(file)
        file.write("operations in {} applied to {}, ".format(Point_group_name, os.path.basename(savepath).split('.')[0] ))
        for symmetry, ttp in data:
            file.write(f"{symmetry} , ")
        file.write("Average  ")
            
        file.write("\n")  
        file.write("Symetry deviations in {} , ".format(Point_group_name))
        for symmetry, ttp in data:
            file.write(f"{round(ttp, 3)} , ")
            
        file.write(f"{round(sum(deviations)/len(deviations), 3)}")
        file.write("\n")  
        file.write("\n")  
        file.write("\n")      
        print('\n ***SUCESSFUL ANALYSIS***')
        print('Symmetry deviation results for', Point_group_name, 'on the structure of', os.path.basename(savepath) ,'has been saved to: \n', directory_path + '\\' + 'SymOp_' + os.path.basename(savepath).split('.')[0] + '.csv' )
        
    with open(directory_path + '\\' + 'results_' + os.path.basename(savepath).split('.')[0] + '.csv', 'a',newline='') as file:
        writer = csv.writer(file)
        # Write a new row with the name and value
        writer.writerow(['{}'.format(Point_group_name), f"{round(sum(deviations)/len(deviations), 3)}"])

    with open(directory_path + '\\' + 'Details_' + os.path.basename(savepath).split('.')[0] + '.csv', 'w',newline='') as file:
        writer = csv.writer(file)

        writer.writerow(['Details:'])
        writer.writerow(['Manual_Zaxis', Manual_Zaxis])
        if not Manual_Zaxis:
            writer.writerow(['N_input_zaxis', N_input_zaxis])
            writer.writerow([''])
            writer.writerow([''])
            writer.writerow(['Optimized Principle Axis:'])
            writer.writerow(['Optimized Z-axis', Zaxis])
            writer.writerow(['Optimized angle to xy-plane', angle])
            writer.writerow(['Rotation Matrix', Rotation_Matrix])
        if not Manual_Angle:
            writer.writerow(['N_input_angles', N_input_angles])
            writer.writerow(['Optimized angle to xy-plane', angle])            
        writer.writerow(['ManualCentering', ManualCentering])
        if ManualCentering:
            writer.writerow(['Manual coordinations', x0,y0,z0])
            writer.writerow(['Manual angle',angle])
            
            

            
def Run_CSoM(mypath,all_structure,structures_selected,point_group_names,N_input_zaxis,N_input_angles,Manual_Zaxis,Zaxis,Manual_Angle,angle,ManualCentering,x0,y0,z0,cut_off_distance,cutoff,SaveStructure):
    
    path_to_structures = [os.path.join(mypath,f) for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
    if not all_structure == True:
        #Use this if individual .xyz file is to be analyzed:
        path_to_structures = [x for x in path_to_structures if os.path.basename(x) in structures_selected]
    
    Rotation_Matrix =  np.array([[1, 0, 0],[0, 1, 0],[0, 0, 1]])
    start_time = datetime.now()
    for savepath in path_to_structures:
        
        all_deviations = []
        for Point_group_name in point_group_names:
            point_group_symmetry_names, point_group_symmetry = ReadSymmetryFile(Point_group_name,Path_To_Symmetry_Operations)
            #Loads and formats the structure:
            if ManualCentering:
                inp_atoms, inp_structure = load_xyz_manual_center(savepath,x0,y0,z0,cutoff)
            else:
                inp_atoms, inp_structure = load_xyz_cutoff(savepath,cutoff)

            #The structure is transformed with respect to the principle axis. If the principle axis is unknown, it is estimated.
            if not Manual_Zaxis:
                structure,Rotation_Matrix,Zaxis,angle,results_zaxises = find_zaxis(inp_structure,inp_atoms,point_group_symmetry, N_input_zaxis = N_input_zaxis, N_input_angles = N_input_angles)        
            elif not Manual_Angle and Manual_Zaxis:
                structure, angle = find_angle(inp_structure,inp_atoms,point_group_symmetry,Zaxis, N_input_angles = N_input_angles)   
            else: 
                #Rotate to defined new z-azis
                if np.array_equal(Zaxis, np.array([0, 0, 1])):
                    structure = inp_structure
                else:
                    R_to_zaxis = rotation_matrix_from_vectors([0,0,1], Zaxis)
                    structure = np.dot(inp_structure,R_to_zaxis)
                    #Rotate the x-y plane with defined angle:
                    R_alpha = rotation_matrix(Zaxis, angle)
                    structure = np.dot(structure,R_alpha)
            #Symmetry deviations are calculated: 
            deviations = point_group_deviation(structure,inp_atoms,point_group_symmetry,Point_group_name, savepath,point_group_symmetry_names,SaveStructure)
            # =============================================================================
            # Script results are saved:
            # =============================================================================
            Save_Results(savepath,Point_group_name,point_group_symmetry_names, deviations,Manual_Zaxis, N_input_zaxis, Zaxis, angle, Rotation_Matrix, Manual_Angle, N_input_angles, ManualCentering, x0,y0,z0)
            all_deviations.append(f"{round(sum(deviations)/len(deviations), 3)}")
            
        """
        For collecting all data into a single result matrix file.
        """
        csv_file_path = os.path.join(os.path.dirname(savepath)+ '\\SymetryOperationDeviations', 'all_results' + '.csv')
        file_exists = os.path.exists(csv_file_path)
        
        # Open the file with newline='' to prevent row separation
        with open(csv_file_path, 'a', newline='') as file:
            writer = csv.writer(file)
        
            if not file_exists:
                # If the file doesn't exist, write the header row
                writer.writerow(['structure'] +  [x.split('_')[1] for x in point_group_names])
        
            # Write a new row with the name and values
            writer.writerow([os.path.basename(savepath).split('.')[0]] + all_deviations)

    elapsed_time = datetime.now() - start_time
    print(f"Elapsed time: {elapsed_time.total_seconds():.4f} seconds")
