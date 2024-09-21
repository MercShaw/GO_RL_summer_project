
from ase.io import read,write
from ase import Atom,  Atoms
from ase import build
from ase import neighborlist
import torch 
import numpy as np 
import math

def select_carbon( action_index: int , # this ranges from 0, N_atm *2
                   oxidised_carbon:list, 
                   nn_list : dict, # here use the updated nn_list as the structures are updated in my code
                   N_atm: int, # total number of oxidised carbon 
                   EP = False):
    selected_carbon  = action_index % N_atm 
    #all_carbon_index =  [i for i in conifg if i.symbol == 'C']

    neighbor_available = [i for i in nn_list[selected_carbon] if i not in oxidised_carbon ]
    
    if selected_carbon in oxidised_carbon : 
        flag = False
        return None, None, oxidised_carbon, flag 
    else: 
        flag = True
        if EP and len( neighbor_available) > 0: 
            selected_neighbor =  np.random.choice( neighbor_available)
            oxidised_carbon.append(selected_carbon)
            oxidised_carbon.append(selected_neighbor)

            return selected_carbon, selected_neighbor, oxidised_carbon, flag 

        elif EP and len( neighbor_available) <= 0 : 
            print( 'this carbon unavailable for EP oxidaiton')
            return None, None, oxidised_carbon, False

        else : 
            oxidised_carbon.append(selected_carbon)
            return selected_carbon, None, oxidised_carbon, flag 
        
def add_EP( action_index: int ,
            oxidised_carbon:list, 
            nn_list : dict, # here use the updated nn_list as the structures are updated in my code
            config: Atoms, 
            N_atm: int, # total number of oxidised carbon 
            state_tensor: torch.tensor, 
            buckling_epoxy: float,
            ): 
    # def 
    selected_carbon, selected_neighbor, oxidised_carbon, flag =  select_carbon(action_index,
                                                     oxidised_carbon,
                                                     nn_list,
                                                     N_atm, 
                                                     EP = True)
    
    box_size =  config.get_cell().diagonal()
    # graphene sheet span through xz direction, means the carbon atom should have a similar y coordinate
    # define +y as  
    
    if selected_carbon is None : 
        # this is the case when an oxidised atm is selected 
        # we do not update anything
        pass
    else:
        bond_vector = config[selected_neighbor].position - config[selected_carbon].position
        bond_vector = minimum_image(bond_vector, box_size)
        midpoint = config[selected_carbon].position + bond_vector / 2

        # Add the oxygen atom (bond length of ~1.46 Angstroms, so ~1.26 Ang above the midpoint - https://arxiv.org/abs/1102.3797)
        # Calculate using basic trigonometry and we will place either above or below the midpoint

        if action_index // N_atm == 1:  
        # set the larger index indicate ones to be on the bottom plane aka down
            config[selected_carbon].position += [
                0, np.random.choice(buckling_epoxy),0
                ]
            
            config[selected_neighbor].position += [
                0, -1 * np.random.choice(buckling_epoxy),0
                ]
            config.append(Atom("O", midpoint + [0, 1.26, 0])) # O pos > C pos --> down
            state_tensor[action_index] = 1
             
            
            return config, oxidised_carbon, state_tensor ,flag


        elif action_index // N_atm == 0:
            config[selected_carbon].position += [
                0, -1 * np.random.choice(buckling_epoxy),0
                ]
            
            config[selected_neighbor].position += [
                0, np.random.choice(buckling_epoxy),0
                ]
            
            config.append(Atom("O", midpoint - [0, 1.26, 0]))
            state_tensor[action_index] = 1
            

            return config, oxidised_carbon, state_tensor , flag
    
    return config, oxidised_carbon, state_tensor, flag



def add_OH( action_index: int ,
            oxidised_carbon:list, 
            nn_list : dict, # here use the updated nn_list as the structures are updated in my code
            config: Atoms, 
            N_atm: int, # total number of oxidised carbon 
            state_tensor: torch.tensor, 
            buckling_OH: float, 

            ): 

    selected_carbon, selected_neighbor, oxidised_carbon, flag =  select_carbon(action_index,
                                                     oxidised_carbon,
                                                     nn_list,
                                                     N_atm 
                                                     )
    
    if selected_carbon is None : 
        pass
    
    else:

        # Hydroxyl groups added as first O and then H attached to it.
        # In both cases, it is critical to make sure that added atom is not in an unreasonably
        # close contact with other atoms. Below, check O atoms are positioned at least 1.85 Ang away
        # from other added O atoms. If not, delete appended O and search for another position.

        # Add the oxygen atom (bond lentgh of 1.49 Angstroms) either above or below the carbon atom

        #onehot_encoded_state[action_index][1] = 1 
        # update the one hot encode state of the OH array whatev the H state, as in the current architect no re-addition of the O atm is implied
        if action_index // N_atm == 0 : # down OH 
            state_tensor[action_index] = 1 
            config[selected_carbon].position += [0, np.random.choice(buckling_OH), 0]
            oxygen_pos = config[selected_carbon].position + [0, 1.49, 0]
            H_pos =  oxygen_pos + [ 0, 0.98, 0]
            # get the collision zone of the added oxygen by looking at which oxygens are close to the oxidised C
        elif action_index // N_atm == 1:
            state_tensor[action_index] = 1  # up OH
            config[selected_carbon].position += [0, -1*np.random.choice(buckling_OH), 0]
            oxygen_pos = config[selected_carbon].position - [0, 1.49, 0]
            H_pos =  oxygen_pos - [ 0, 0.98, 0]
            # get the collision zone of the added oxygen by looking at which oxygens are close to the oxidised C


        config.append(Atom("O", oxygen_pos))
        config.append(Atom("H", H_pos))

    return config, oxidised_carbon, state_tensor,  flag




# check if this is needed or not
def minimum_image(vector, box_size):
    # Apply the periodic boundary conditions to the vector
    for i in range(3):
        if abs(vector[i]) > box_size[i] / 2:
            vector[i] -= box_size[i] * round(vector[i] / box_size[i])
    return vector


def is_structure_valid(
    graphene, group, collision_zone, r_c=1.85 / 2, r_o=1.52 / 2, r_h=1.2 / 2
):
    """
    Summary
    ----------
    Check if functional group can be added to graphene sheet without causing close contacts

    Parameters
    ----------
    graphene : ase.Atoms
        Graphene structure
    group: ase.Atoms
        Functional group added to nanoriibon; Atom attached to ribbon must be placed at (0,0,0)
    collision_zone : list
        Indices of atoms that can collide with functional group
    r_c:
        Hard sphere radius for carbon (Default value is VdW radius)
    r_o:
        Hard sphere radius for oxygen (Default value is VdW radius)
    r_h:
        Hard sphere radius for hydrogen (Default value is VdW radius)
    """
    # Cut offs for checking for collisions
    threshholds = {
        "CC": r_c * 2,
        "OO": r_o * 2,
        "HH": r_h * 2,
        "HC": r_c + r_h,
        "CH": r_c + r_h,
        "OH": r_o + r_h,
        "HO": r_o + r_h,
        "CO": r_c + r_o,
        "OC": r_c + r_o,
    }
    # If collision zone is empty then structure is valid
    if len(collision_zone) == 0:
        return True

    # Loop over pairs to check for collisions
    for atom in group:
        for i in collision_zone:
            pair = graphene[i].symbol + atom.symbol
            atom_1 = np.array(graphene[i].position)
            atom_2 = np.array(atom.position)
            # Apply minimum image convention
            dx = abs(atom_1[0] - atom_2[0])
            dx = min(dx, graphene.get_cell()[0, 0] - dx)
            dy = abs(atom_1[1] - atom_2[1])
            dy = min(dy, graphene.get_cell()[1, 1] - dy)
            dz = abs(atom_1[2] - atom_2[2])
            dz = min(dz, graphene.get_cell()[2, 2] - dz)

            # Calculate distance between atoms
            min_dist = math.sqrt(dx**2 + dy**2 + dz**2)

            # Check min distance
            if min_dist <= threshholds[pair]:
                return False
    return True


