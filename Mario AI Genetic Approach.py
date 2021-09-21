#Amanda Israel Graeff Borges, Lucas Veit de Sá e Mateus Karvat Camara
import numpy as np
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

#TO-DO LIST:
# Arrumar fork X
# Arrumar crossover
# Reavaliar ações 
### Checar documentação dos inputs no pyboy X
# Função de mutação
### Checar slides da Adriana e ver como mutação geralmente acontece X
### Avaliar viabilidade de implementar a mutação do paper X
# Grid search

style.use("ggplot")     #matplotlib style

population = 5
generations = 100
mutation_rate = 0.2
threshold = 100000      #o que ser??


cromossome_size = 800         #tamanho do cromossomo (numero de ações)
time_h = 0
lucro0 = 0

class environment:
    def __init__(self):
        #start the game
        filename = 'roms/Super Mario Land.gb'
        quiet = "--quiet" in sys.argv
        self.pyboy = PyBoy(filename, window_type="headless" if quiet else "SDL2", window_scale=3, debug=quiet, game_wrapper=True)
        self.pyboy.set_emulation_speed(1)       #velocidade de emulação
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
        #generate random actions
        for i in range(cromossome_size):
            #self.action = env.action_space.sample()
            self.action = random.randint(0,5)
            self.actions.append(self.action)

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
    # Seleciona os 20% melhores daquela população
    networks = networks[:int(0.2 * len(networks))]

    return networks

def crossover(networks):
    offspring = []    
    for _ in range(population//2):
        parent1 = random.choice(networks)
        parent2 = random.choice(networks)
        child1 = Network()
        child2 = Network()

        # Escolha aleatória do ponto de crossover
        n = np.random.randint(0,cromossome_size)

        # Crossover dos pais
        p1_beginning = parent1.actions[:n]
        p1_end = parent1.acions[n:]
        p2_beginning = parent2.actions[:n]
        p2_end = parent2.acions[n:]
        
        child1.actions = p1_beginning+p2_end
        child2.actions = p2_beginning+p1_end

        offspring.append(child1)
        offspring.append(child2)
        
    networks.extend(offspring)
    return networks


def mutate(networks):
    # Mutação
    for network in networks[2:]:
        for _ in range(0,int(cromossome_size/2)):
            val = np.random.uniform(0, 1)
            if val <= mutation_rate: #mutation chance
                idx = np.random.randint(0,len(network.actions))
                network.actions[idx] = np.random.randint(0,3)

    return networks


def main():
    lucro_nets = []
    best_lucro_nets = []
    best_networks = []
    lucro_nets_media = []
    
    networks = init_networks(population)
    #load best from the past
    try:
        with open('genetic_best_network_mario.json') as json_file:
            data = json.load(json_file)
        lucro = networks[0].set_actions(data.get('actions'),data.get('lucro'))
    except:
        print('Error to load')
        data = None

    for gen in range(generations):
        print ('Generation {}'.format(gen+1))

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
        #best_networks.append(networks)
        networks = crossover(networks)
        networks = mutate(networks)
                
        print ('Best Fitness: {}'.format(max(network_lucro)))
        best_lucro_nets.append(max(network_lucro))
        #print(network_lucro)
        media = sum(network_lucro)/len(network_lucro)
        print ('Average Fitness: {}'.format(media))
        lucro_nets_media.append(media)
        print("")

    #save best network
    best_net = {'actions': networks[0].get_actions(),'lucro':networks[0].lucro, 'generation':networks[0].generation}
    with open('genetic_best_network_mario.json', 'w') as json_file:
        json.dump(best_net, json_file)
    json_file.close()
                
    plt.subplot(211)
    plt.plot([i for i in range(len(lucro_nets_media))], lucro_nets_media)
    plt.plot([i for i in range(len(best_lucro_nets))], best_lucro_nets)
    plt.ylabel(f"Average Fitness by generation")
    plt.xlabel("Generation #")

    plt.subplot(212)
    plt.plot([i for i in range(len(lucro_nets))], lucro_nets)
    #plt.plot([i for i in range(len(lucro_time))], lucro_time)

    plt.show()
                    

if __name__ == '__main__':
    main()