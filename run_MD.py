import ase
from ase.io import read, write
import pandas as pd
import numpy as np
import os 
import matplotlib.pyplot as plt
from mace.calculators import MACECalculator

from ase import units
from ase.md.npt import NPT
from ase.md import MDLogger
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from ase.neighborlist import NeighborList, natural_cutoffs
import networkx as nx

from mace.calculators import MACECalculator

calculator = MACECalculator(
    model_paths="./MACE_model_swa_2.model",
    device="cuda",
    default_dtype="float32")

def check_or_create_dir(path: str): 
    if not os.path.isdir(path): 
        os.makedirs(path, exist_ok=True)
    return

def set_up_md ( go : ase.Atoms, iter: int, param : dict ) : 
    '''
    start in parent dir : ./GO_RL

    take the end configuration and other necessary info to set up a MACE-MD 
    go: final GO config
    iter: number of iteration, help trakcing 
    parma: C:H:O ratio, help tracking 
   {
        'n_C': N_atom,
        'n_O': n_O,
        'n_H': n_H, 
        'T': T
    }
    '''
    data_folder = './data_energy'

    '''i decide to collect the experiments as: 
    
    /data (data_folder)
        /C_a_O_b_H_c_tK (simulation_name)               : a:b:c is number of corresponding atoms, t is the MD temperature     
            /C_a_O_b_H_c_tk_iter_i (iteration name )    : ith iteration for this system studied (i = 0 is separate random structure)
                /C_a_O_b_H_c_tk.xyz                     : initial configuration return by Net
                /result_C_a_O_b_H_c_tk.traj             : .Traj file returned after Geo-optm ( main interest)
                /result_C_a_O_b_H_c_tk.xyz              : config trajectory during MACE-MD
                /result_C_a_O_b_H_c_tk.txt              : log file returned after MACE-MD
    
    '''
    simulaiton_name  = f"C_{param['n_C']}_O_{param['n_O']}_H_{param['n_H']}_T{param['T']}K"
    simualtion_path  =  os.path.join(data_folder, simulaiton_name)
    check_or_create_dir(simualtion_path)
    os.chdir(simualtion_path)

    iteration_name = f'./{simulaiton_name}_iter_{iter}'
    check_or_create_dir(iteration_name)
    os.chdir(iteration_name)
    write(f'{simulaiton_name}.xyz', go)  
   # stay in the iteration directory to run MD
    return simulaiton_name

def run ( config : ase.Atoms ,simulation_name : str, T: float, ): 
    config.set_calculator(calculator)
    # all parameters below are set the same with the exisiting paper
    T_init = T # K
    Eqr_arg = {
    'ttime' :10, # ps
    'timestep' :0.5 * units.fs , # fs time step
    }
    #init_conf.set_cell(np.dot(init_conf.cell, strain_tensor), scale_atoms=True)

    # Set the momenta corresponding to T=300 K
    MaxwellBoltzmannDistribution(config, temperature_K=T_init)

    dyn = NPT(config, temperature_K=T_init, **Eqr_arg)
    def write_frame():
        dyn.atoms.write(f"result_{simulation_name}.xyz", append=True)

    dyn.attach(write_frame, interval=100)
    dyn.attach(
    MDLogger(dyn, config, f"result_{simulation_name}.txt", stress=True, peratom=True, mode="a"), interval=50
    )
    dyn.run(8_000)

    return 

def collect_md_data(simualtion_name: str): 
    file_pth = f"result_{simualtion_name}.xyz"
    end_config = read( file_pth,'-1')
    end_config_remove_gas = remove_gas( end_config)
    end_config_remove_gas.calc =  calculator
    end_config_remove_gas.get_potential_energy()
    
    node_energy = abs( end_config_remove_gas.calc.results['node_energy'].sum())/ len(end_config) # this can penalise water formation: we want GO to be more functionalised
    
    return node_energy

def remove_gas(config: ase.Atoms ):
    cutoff = natural_cutoffs( config )
    # there are not that much difference between natrual cutoff and the actual radii in this system 
    # for gaseous species they will be off anyway 
    nn =  NeighborList( cutoff,self_interaction=False, bothways=True)
    nn.update(config)
    graph  = nx.Graph()

    for idx, atom in enumerate(config): 
        graph.add_node(idx, pos = atom.position,  ele = atom.symbol)

    for idx, atom in enumerate(config) : 
        neighbors, offsets =  nn.get_neighbors(idx)
        for neighbor in neighbors: 
            graph.add_edge( idx, neighbor)

    # construct the configuration into a graph 
    # add the edges based on atoms's natrual cutoff

    if nx.is_connected(graph): 
        pass
    else: 
        cluster_list = list(nx.connected_components(graph))
        cluster_list.sort()
        # keep the largest cluster, which will be the GO, smaller ones will be abandoned automatically 
        atm_to_keep = list(cluster_list[0])
        config = config[atm_to_keep]
    return config 

