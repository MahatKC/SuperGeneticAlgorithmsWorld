#Amanda Israel Graeff Borges, Lucas Veit de Sá e Mateus Karvat Camara
import numpy as np
from numpy.random import default_rng
import time
import pandas as pd
import random
import os
import sys
from pyboy import PyBoy, WindowEvent
import json
from torch.utils.tensorboard import SummaryWriter

emulation_speed = 5

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

        if action == 0: # Pular       
            self.pyboy.send_input(WindowEvent.PRESS_BUTTON_A)
            self.time = 2

        elif action == 1: # Soltar Pular
            self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_A)
            self.time = 2

        elif action == 2: # Andar para esquerda
            self.pyboy.send_input(WindowEvent.RELEASE_ARROW_RIGHT)
            self.pyboy.send_input(WindowEvent.PRESS_ARROW_LEFT)
            self.time = 2

        elif action == 3: # Parar de andar para esquerda
            self.pyboy.send_input(WindowEvent.RELEASE_ARROW_LEFT)
            self.time = 2

        elif action == 4: # Andar para direita
            self.pyboy.send_input(WindowEvent.RELEASE_ARROW_LEFT)
            self.pyboy.send_input(WindowEvent.PRESS_ARROW_RIGHT)
            self.time = 2

        elif action == 5: # Parar de andar para direita
            self.pyboy.send_input(WindowEvent.RELEASE_ARROW_RIGHT)
            self.time = 2

        elif action == 6: # Apertar Correr / Fogo
            self.pyboy.send_input(WindowEvent.PRESS_BUTTON_B)
            self.time = 2

        elif action == 7: # Soltar Correr / Fogo
            self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_B)
            self.time = 2

        elif action == 8: # Apertar para Baixo
            self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_A)
            self.pyboy.send_input(WindowEvent.PRESS_ARROW_DOWN)
            self.time = 2

        elif action == 9: # Soltar para Baixo
            self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_A)
            self.pyboy.send_input(WindowEvent.RELEASE_ARROW_DOWN)
            self.time = 2
                 
        return action, self.time

class Network():
    def __init__(self, cromossome_size):
        # Gera cromossomo com sequência aleatória de ações
        self.actions = np.random.randint(0, 10, size=cromossome_size).tolist()
        #self.actions = [4]*100+[0]*3+[1]+[5]+[2]*15+[3]+[0]+[5]*400
        self.generation = 0

        self.lucro = 0

    def get_actions(self):
        return self.actions

    def set_actions(self,actions,lucro):
        self.actions = actions
        self.lucro = lucro
        return self.lucro

def init_networks(population, cromossome_size):
    return [Network(cromossome_size) for _ in range(population)]

def fitness(networks):
    for network in networks:
        #init env
        env = environment()
        state_size = env.state_size
        fitness = env.mario.fitness
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        
        actions = network.get_actions()
        
        # loop through actions
        vidas = env.mario.lives_left
        for act in actions:
            try:
                #16,17,18,19,27,26
                filteredMario = [x for x in list(state[0]) if (x>10 and x<30)]
                index_mario = list(state[0]).index(filteredMario[0])
                feet_val = state[0][index_mario + 20]
            except:
                pass
                #print(iterator)
                #break

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
            
            if env.mario.lives_left < vidas:
                done = True
                break
            
            vidas = env.mario.lives_left
        network.lucro = fitness
        
        #print('Lucro Total: {}'.format(network.lucro))

        env.pyboy.stop()
    return networks

def selection(networks, population, selection_percentage):
    # Ordena os membros da população com base em seu lucro
    networks = sorted(networks, key=lambda network: network.lucro, reverse=True)
    # Seleciona os selection_percentage % melhores daquela população
    networks = networks[:int(selection_percentage * population)]

    return networks

def crossover(networks, self_crossover, population, selection_percentage, cromossome_size):
    children = []   
    num_pares_filhos = int(population*((1-selection_percentage)/2))
    for _ in range(num_pares_filhos):
        parent1 = random.choice(networks)
        parent2 = random.choice(networks)

        if not self_crossover:
            while parent1==parent2:
                parent1 = random.choice(networks)
                parent2 = random.choice(networks)

        child1 = Network(cromossome_size)
        child2 = Network(cromossome_size)

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

def mutate(networks, mutation_rate, mutation_probability, population, selection_percentage, cromossome_size):
    # Mutação
    num_old_members = int(population*selection_percentage)
    rng = default_rng()
    indices_populacao = np.arange(population)[num_old_members:]
    mutated_networks = rng.choice(indices_populacao, size=int(len(indices_populacao)*mutation_probability), replace=False).tolist()
    for network_idx in mutated_networks:
        num_genes_mutados = int(cromossome_size*mutation_rate)
        genes_sorteados = rng.choice(cromossome_size, size=num_genes_mutados, replace=False)
        for gene in genes_sorteados:
            network = networks[network_idx]
            network.actions[gene] = np.random.randint(10)

    return networks

def run(run_name, population, generations, self_crossover, mutation_rate, mutation_probability, selection_percentage, cromossome_size):
    lucro_nets = []
    best_lucro_nets = []
    lucro_nets_media = []
    
    networks = init_networks(population, cromossome_size)
    
    writer = SummaryWriter(log_dir='runs/'+run_name)

    for gen in range(generations):
        t0=time.time()
        #print (f'Generation {gen+1}')

        #action
        networks = fitness(networks)
             
        network_lucro = []
        for network in networks:
            network.generation = gen
            network_lucro.append(network.lucro)
            lucro_nets.append(network.lucro)

        #Genetic
        networks = selection(networks, population, selection_percentage)
        networks = crossover(networks, self_crossover, population, selection_percentage, cromossome_size)
        networks = mutate(networks, mutation_rate, mutation_probability, population, selection_percentage, cromossome_size)
                
        max_lucro = np.max(network_lucro)
        media = np.average(network_lucro)
        writer.add_scalar('Lucro/Máximo', max_lucro, gen+1)
        writer.add_scalar('Lucro/Média', media, gen+1)
        best_lucro_nets.append(max_lucro)
        print (f"Best Fitness: {max_lucro}")
        lucro_nets_media.append(media)
        #print (f"Average Fitness: {media}\n")
        t1=time.time()
        execution_time = np.round(t1-t0, 4)

        grid_search_df = pd.DataFrame({'run_name': [run_name],
                                'population': [population],
                                'generations':[generations],
                                'self_crossover': [str(self_crossover)],
                                'mutation_rate': [mutation_rate],
                                'mutation_probability': [mutation_probability],
                                'selection_percentage': [selection_percentage],
                                'cromossome_size': [cromossome_size],
                                'best_result': [max_lucro],
                                'tempo': [execution_time]
                                })

        #file_df = pd.read_csv("grid_search.csv")
        #file_df = file_df.append(grid_search_df)
        grid_search_df.to_csv("grid_search.csv",index=False)
    writer.close()
    writer.flush()

def main():
    population_values = [10, 20] #50
    generations_values = [10, 20, 50] #100
    self_crossover_values = [True, False]
    mutation_rate_values = [0.001, 0.01, 0.1] #0.005, 0.05, 0.2
    mutation_probability_values = [1, 0.1, 0] #0.5
    selection_percentage_values = [0.2]  #porcentagem dos melhores membros da população que irão pra próxima geração
    cromossome_size_values = [2000, 5000, 10000]      #tamanho do cromossomo (numero de ações)
    run_name="run"
    i=0
    for population in population_values:
        for generations in generations_values:
            for self_crossover in self_crossover_values:
                for mutation_rate in mutation_rate_values:
                    for mutation_probability in mutation_probability_values:
                        for selection_percentage in selection_percentage_values:
                            for cromossome_size in cromossome_size_values:
                                run_name = "run"+str(i)
                                run(run_name, population, generations, self_crossover, mutation_rate, mutation_probability, selection_percentage, cromossome_size)
                                i+=1
                                if i%10==0:
                                    print(f"Iteração {i}")

if __name__ == '__main__':
    main()