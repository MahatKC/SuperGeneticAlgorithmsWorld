#Amanda Israel Graeff Borges, Lucas Veit de Sá e Mateus Karvat Camara
import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time
import pandas as pd
import random
import os
import sys
from pyboy import PyBoy, WindowEvent
import json
from torch.utils.tensorboard import SummaryWriter

style.use("ggplot")     #matplotlib style

population = 10
emulation_speed = 10
generations = 100
self_crossover = True
mutation_rate = 0.2
selection_percentage = 0.2  #porcentagem dos melhores membros da população que irão pra próxima geração
threshold = 100000   #o que ser??

cromossome_size = 800         #tamanho do cromossomo (numero de ações)
time_h = 0
lucro0 = 0

class environment:
    def __init__(self):
        #start the game
        filename = 'roms/Super Mario Land.gb'
        quiet = "--quiet" in sys.argv
        self.pyboy = PyBoy(filename, window_type="headless" if quiet else "SDL2", window_scale=3, debug=quiet, game_wrapper=True)
        self.pyboy.set_emulation_speed(emulation_speed)       #velocidade de emulação
        assert self.pyboy.cartridge_title() == "SUPER MARIOLAN"

        self.mario = self.pyboy.game_wrapper()
        self.mario.start_game()
        self.done= False
        self.time = 10

        assert self.mario.score == 0
        assert self.mario.lives_left == 2
        assert self.mario.time_left == 400
        assert self.mario.world == (1, 1) #stage
        assert self.mario.fitness == 0 # A built-in fitness score for AI development
        
        #set state and action size
        self.action_size = 5 #number of possible actions
        state_full = np.asarray(self.mario.game_area())
        np.append(state_full,self.mario.level_progress) #talvez usar o append corretamente
        self.state_size = state_full.size     
              
    def reset(self):
        self.mario.reset_game() #back to the last state saved
        self.done = False
        self.pyboy.tick()
        assert self.mario.lives_left == 2
        self.position = self.mario.level_progress
        state_full = np.asarray(self.mario.game_area())
        np.append(state_full,self.mario.level_progress)
        
        return state_full

    def step(self,action):
        if action == 0:
            self.pyboy.send_input(WindowEvent.PRESS_BUTTON_A)
            self.time = 5
        elif action == 1:
            self.pyboy.send_input(WindowEvent.PRESS_ARROW_RIGHT)
            self.time = 5
        elif action == 2:
            #self.pyboy.send_input(WindowEvent.PRESS_BUTTON_A)
            #time.sleep(0.1)
            #self.pyboy.send_input(WindowEvent.PRESS_ARROW_RIGHT)
            self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_A)
            self.time = 5
        elif action == 3:
            self.pyboy.send_input(WindowEvent.RELEASE_ARROW_RIGHT)
            self.time = 5
        elif action == 4:
            self.pyboy.send_input(WindowEvent.PRESS_ARROW_LEFT)
            self.time = 5
                 
        return action, self.time

class Network():
    def __init__(self):
        self.actions = []
        self.generation = 0

        # Gera cromossomo com sequência aleatória de ações
        self.action = random.randint(5, size=cromossome_size).tolist()

        self.lucro = 0

    def get_actions(self):
        return self.actions

    def set_actions(self,actions,lucro):
        self.actions = actions
        self.lucro = lucro
        return self.lucro

def init_networks(population):
    return [Network() for _ in range(population)]

def fitness(networks):
    for network in networks:
        #init env
        env = environment()
        state_size = env.state_size
        action_size = env.action_size
        fitness = env.mario.fitness
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        
        actions = network.get_actions()
        strategies = []
        lucro_tot = 0
        
        #try:
        time_h = 0
        # loop through actions
        for act in actions:
            #print("action: ", act, " lives: ",env.mario.lives_left, " pos: ", env.mario.level_progress)
            try:
                #16,17,18,19,27,26
                filteredMario = [x for x in list(state[0]) if (x>10 and x<30)]
                #print(filteredMario)
                index_mario = list(state[0]).index(filteredMario[0])
                feet_val = state[0][index_mario + 20]
                #print('POSITION: ',state[0][index_mario], state[0][index_mario + 20])
            except:
                break

            act, tempo = env.step(act)
            
            state = np.asarray(env.mario.game_area()) 
            position = env.mario.level_progress
            state = np.reshape(state, [1, state_size])
            
            i=0
            while feet_val <= 350:
                env.pyboy.tick()
                i += 1
                if i > 60:
                    break
                
            if feet_val >= 350:
                tempo = 2
                for _ in range(tempo):
                    env.pyboy.tick() # Progresses the emulator ahead by one frame.
                
            
            t = 0.0167 * tempo
            time.sleep(t)
            
            fitness = env.mario.fitness
            
            if env.mario.lives_left == 1:
                done = True
                break
            
            time_h += 1
            
        network.lucro = fitness
        
        print('Lucro Total: {}'.format(network.lucro))

        env.pyboy.stop()
    return networks

def selection(networks):
    # Ordena os membros da população com base em seu lucro
    networks = sorted(networks, key=lambda network: network.lucro, reverse=True)
    # Seleciona os selection_percentage % melhores daquela população
    networks = networks[:int(selection_percentage * population)]

    return networks

def crossover(networks):
    children = []   
    num_pares_filhos = int(population*((1-selection_percentage)/2))
    for _ in range(num_pares_filhos):
        parent1 = random.choice(networks)
        parent2 = random.choice(networks)

        if not self_crossover:
            while parent1==parent2:
                parent1 = random.choice(networks)
                parent2 = random.choice(networks)

        child1 = Network()
        child2 = Network()

        # Escolha aleatória do ponto de crossover
        n = np.random.randint(0,cromossome_size)

        # Crossover dos pais
        p1_beginning = parent1.actions[:n]
        p1_end = parent1.actions[n:]
        p2_beginning = parent2.actions[:n]
        p2_end = parent2.actions[n:]
        
        child1.actions = p1_beginning+p2_end
        child2.actions = p2_beginning+p1_end

        children.append(child1)
        children.append(child2)
        
    networks.extend(children)
    return networks


def mutate(networks):
    # Mutação
    num_old_members = int(population*selection_percentage)
    rng = default_rng()
    for network in networks[num_old_members:]:
        num_genes_mutados = int(cromossome_size*mutation_rate)
        genes_sorteados = rng.choice(cromossome_size, size=num_genes_mutados, replace=False)
        for gene in genes_sorteados:
            network.actions[gene] = np.random.randint(0,3)

    return networks

def run(run_name):
    lucro_nets = []
    best_lucro_nets = []
    best_networks = []
    lucro_nets_media = []
    
    networks = init_networks(population)
    """
    #load best from the past
    try:
        with open('genetic_best_network_mario.json') as json_file:
            data = json.load(json_file)
        lucro = networks[0].set_actions(data.get('actions'),data.get('lucro'))
    except:
        print('Error to load')
        data = None
    """
    
    writer = SummaryWriter(log_dir='runs/'+run_name)

    for gen in range(generations):
        print (f'Generation {gen+1}')

        #action
        networks = fitness(networks)
             
        network_lucro = []
        for network in networks:
            network.generation = gen
            network_lucro.append(network.lucro)
            lucro_nets.append(network.lucro)
            if network.lucro > threshold:
                print ('Threshold met')
                print (network.get_actions())
                print ('Best lucro: {}'.format(network.lucro))
                exit(0)

        #Genetic
        networks = selection(networks)
        networks = crossover(networks)
        networks = mutate(networks)
                
        max_lucro = np.max(network_lucro)
        media = np.average(network_lucro)
        writer.add_scalar('Lucro/Máximo', max_lucro, gen+1)
        writer.add_scalar('Lucro/Média', media, gen+1)
        best_lucro_nets.append(max_lucro)
        print (f"Best Fitness: {max_lucro}")
        lucro_nets_media.append(media)
        print (f"Average Fitness: {media}\n")

    """
    population
    generations
    self_crossover
    mutation_rate
    selection_percentage
    cromossome_size
    best_result
    """

    grid_search_df = pd.DataFrame({'layer_type': ["Linear"],
                               'layers': ["(10, 20), (20, 20), (10, 3)"],
                               'num_trainable_params':['630'],
                               'time': [str(round(t1-t0,3))],
                               'activation': ["TanH"],
                               'optimizer': ["Adam"],
                               'optimizer_params': ["actual weight_decay="+str(WEIGHT_DECAY)],
                               'learning_rate': ["actual learning_rate="+str(LEARNING_RATE)],
                               'loss_func': ["MSELoss"],
                               'activate_dropout': [str(ACTIVATE_DROPOUT)],
                               'dropout_rate': [str(DROPOUT_RATE)],
                               'num_epochs': [str(NUM_EPOCHS)],
                               'batch_size': [str(BATCH_SIZE)],
                               'train_accuracy': [str(round(batchNet.accuracy(train),4))],
                               'val_accuracy': [str(round(batchNet.accuracy(val),4))]
                               })

    file_df = pd.read_csv("grid_search.csv")
    file_df = file_df.append(grid_search_df)
    file_df.to_csv("grid_search.csv",index=False)
    writer.close()
    writer.flush()

    """
    #save best network
    best_net = {'actions': networks[0].get_actions(),'lucro':networks[0].lucro, 'generation':networks[0].generation}
    with open('genetic_best_network_mario.json', 'w') as json_file:
        json.dump(best_net, json_file)
    json_file.close()
    """

def main():
    run_name="run"
    run(run_name)
                
if __name__ == '__main__':
    main()