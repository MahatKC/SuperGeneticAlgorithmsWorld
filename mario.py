#Amanda Israel Graeff Borges, Lucas Veit de Sá e Mateus Karvat Camara
import numpy as np
from numpy.random import default_rng
import time, copy, sys, random, json
import pandas as pd
from pyboy import PyBoy, WindowEvent
from torch.utils.tensorboard import SummaryWriter

emulation_speed = 5

class environment:
    def __init__(self):
        #Ínicio do jogo
        filename = 'roms/Super Mario Land.gb'
        quiet = "--quiet" in sys.argv
        self.pyboy = PyBoy(filename, window_type="headless" if quiet else "SDL2", window_scale=3, debug=quiet, game_wrapper=True)
        self.pyboy.set_emulation_speed(emulation_speed)       #velocidade de emulação
        assert self.pyboy.cartridge_title() == "SUPER MARIOLAN"

        self.mario = self.pyboy.game_wrapper()
        self.mario.start_game()
        self.done= False
        self.time = 10

        #Variáveis do jogo
        assert self.mario.score == 0 #Pontuação
        assert self.mario.lives_left == 2 #Número restante de vidas
        assert self.mario.time_left == 400 #Tempo restante
        assert self.mario.world == (1, 1) #stage
        assert self.mario.fitness == 0 # Avalização de aptidão
        
        #Salvar estado do jogo e o tamanho do cromossomo
        state_full = np.asarray(self.mario.game_area())
        np.append(state_full,self.mario.level_progress) 
        self.state_size = state_full.size     
              
    def reset(self):
        self.mario.reset_game() #Volta para o último estado salvo
        self.done = False
        self.pyboy.tick()
        assert self.mario.lives_left == 2
        self.position = self.mario.level_progress
        state_full = np.asarray(self.mario.game_area())
        np.append(state_full,self.mario.level_progress)
        
        return state_full

    #Ações que o Mario está restrito
    def step(self,action, reduced_action):
        if not reduced_action:
            if action == 0: # Pular       
                self.pyboy.send_input(WindowEvent.RELEASE_ARROW_DOWN)
                self.pyboy.send_input(WindowEvent.PRESS_BUTTON_A)
                self.time = 2

            elif action == 1: # Soltar Pular
                self.pyboy.send_input(WindowEvent.RELEASE_ARROW_DOWN)
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
        else:
            if action == 0: # Pular       
                self.pyboy.send_input(WindowEvent.PRESS_BUTTON_A)
                self.time = 2

            elif action == 1: # Soltar Pular
                self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_A)
                self.time = 2

            elif action == 2: # Andar para direita
                self.pyboy.send_input(WindowEvent.PRESS_ARROW_RIGHT)
                self.time = 2

            elif action == 3: # Parar de andar para direita
                self.pyboy.send_input(WindowEvent.RELEASE_ARROW_RIGHT)
                self.time = 2
                 
        return action, self.time

class Network():
    def __init__(self, cromossome_size, reduced_action):
        # Gera cromossomo com sequência aleatória de ações
        if reduced_action:
            self.actions = np.random.randint(0, 4, size=cromossome_size).tolist()
        else:
            self.actions = np.random.randint(0, 10, size=cromossome_size).tolist()
        #self.actions = [4]*100+[0]*3+[1]+[5]+[2]*15+[3]+[0]+[5]*400
        self.generation = 0
        self.death_iteration = 0

        self.lucro = 0 #armazena o fitness

    def get_actions(self):
        return self.actions

    def set_actions(self,actions,lucro):
        self.actions = actions
        self.lucro = lucro
        return self.lucro

def init_networks(population, cromossome_size, reduced_action):
    return [Network(cromossome_size, reduced_action) for _ in range(population)]

def fitness(networks, reduced_action):
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
        action_counter = 0

        print(env.mario)

        for act in actions:
            t_act_ini = time.time()
            if action_counter == len(actions)-1:
                print(f"Ultima ação {action_counter}!!!")

            try:
                filteredMario = [x for x in list(state[0]) if (x>10 and x<30)]
                index_mario = list(state[0]).index(filteredMario[0])
                feet_val = state[0][index_mario + 20]
            except:
                pass
                #print(iterator)
                #break

            act, tempo = env.step(act, reduced_action)
            
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
                    env.pyboy.tick() # Avança um frame
                
            t = 0.0167 * tempo
            time.sleep(t)

            fitness = env.mario.fitness
            
            fitness = env.mario.fitness
            
            if env.mario.lives_left < vidas:
                print(env.mario)
                network.death_iteration = action_counter
                done = True
                break
            
            vidas = env.mario.lives_left

            action_counter+=1
        network.lucro = fitness

        env.pyboy.stop()
    return networks

def selection(networks, population, selection_percentage):
    # Ordena os membros da população com base em seu lucro
    networks = sorted(networks, key=lambda network: network.lucro, reverse=True)
    # Seleciona os selection_percentage % melhores daquela população
    networks = networks[:int(selection_percentage * population)]

    return networks

#Cruzamento sobre um ponto
def crossover(networks, self_crossover, population, selection_percentage, cromossome_size, death_mutation, reduced_action):
    children = []
    if death_mutation:
        num_pares_filhos = int(population*((1-selection_percentage)/4))
        for _ in range(num_pares_filhos):
            parent1 = random.choice(networks)
            parent2 = random.choice(networks)

            if not self_crossover:
                while parent1==parent2:
                    parent1 = random.choice(networks)
                    parent2 = random.choice(networks)

            child1 = Network(cromossome_size, reduced_action)
            child2 = Network(cromossome_size, reduced_action)
            child3 = copy.deepcopy(parent1)
            child4 = copy.deepcopy(parent2)

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
            children.append(child3)
            children.append(child4)
    else:
        num_pares_filhos = int(population*((1-selection_percentage)/2))
        for _ in range(num_pares_filhos):
            parent1 = random.choice(networks)
            parent2 = random.choice(networks)

            if not self_crossover:
                while parent1==parent2:
                    parent1 = random.choice(networks)
                    parent2 = random.choice(networks)

            child1 = Network(cromossome_size, reduced_action)
            child2 = Network(cromossome_size, reduced_action)

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

def mutate(networks, mutation_rate, mutation_probability, population, selection_percentage, cromossome_size, death_mutation, reduced_action_set):
    # Mutação
    num_old_members = int(population*selection_percentage)
    #Caso ocorra mutação antes da próxima run
    if death_mutation:
        rng = default_rng()
        child1 = list(range(num_old_members,population,4))
        child2 = list(range(num_old_members+1,population,4))
        child1.extend(child2)
        child1.sort()
        child3 = list(range(num_old_members+2,population,4))
        child4 = list(range(num_old_members+3,population,4))
        child3.extend(child4)
        child3.sort()
        #Seleção da Network que sofrerá mutação 
        mutated_networks = rng.choice(child1, size=int(len(child1)*mutation_probability), replace=False).tolist()
        for network_idx in mutated_networks:
            #Seleção das ações que sofrerão mutação
            num_genes_mutados = int(cromossome_size*mutation_rate)
            network = networks[network_idx]

            #limiar de mutação antes da próxima run
            death_threshold = int(network.death_iteration*0.9)
            genes_in_death_threshold = np.arange(cromossome_size)[death_threshold:]
            genes_sorteados = rng.choice(genes_in_death_threshold, size=num_genes_mutados, replace=False)
            for gene in genes_sorteados:
                if reduced_action_set:
                    network.actions[gene] = np.random.randint(4)
                else:
                    network.actions[gene] = np.random.randint(10)

        mutated_networks_two = rng.choice(child3, size=int(len(child3)*mutation_probability), replace=False).tolist()
        for network_idx in mutated_networks_two:
            num_genes_mutados = int(cromossome_size*mutation_rate)
            genes_sorteados = rng.choice(cromossome_size, size=num_genes_mutados, replace=False)
            for gene in genes_sorteados:
                network = networks[network_idx]
                if reduced_action_set:
                    network.actions[gene] = np.random.randint(4)
                else:
                    network.actions[gene] = np.random.randint(10)
    else:
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

def run(iteration, run_name, population, generations, self_crossover, mutation_rate, mutation_probability, selection_percentage, cromossome_size, death_mutation, reduced_action):
    print(f"Run {iteration}")
    print("-"*10)

    lucro_nets = []
    best_lucro_nets = []
    lucro_nets_media = []
    
    networks = init_networks(population, cromossome_size, reduced_action)
    
    writer = SummaryWriter(log_dir='runs/'+run_name)
    t0=time.time()

    for gen in range(generations):
        
        #print (f'Generation {gen+1}')

        #action
        networks = fitness(networks, reduced_action)
             
        network_lucro = []
        for network in networks:
            network.generation = gen
            network_lucro.append(network.lucro)
            lucro_nets.append(network.lucro)

        #Genetic
        networks = selection(networks, population, selection_percentage)
        networks = crossover(networks, self_crossover, population, selection_percentage, cromossome_size, death_mutation, reduced_action)
        networks = mutate(networks, mutation_rate, mutation_probability, population, selection_percentage, cromossome_size, death_mutation, reduced_action)
                
        max_lucro = np.max(network_lucro)
        media = np.average(network_lucro)
        writer.add_scalar('Lucro/Máximo', max_lucro, gen+1)
        writer.add_scalar('Lucro/Média', media, gen+1)
        best_lucro_nets.append(max_lucro)
        print (f"Best Fitness: {max_lucro}. Generation: {gen}")
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
    if iteration==0:
        grid_search_df.to_csv("grid_search.csv",index=False)
    else:
        file_df = pd.read_csv("grid_search.csv")
        file_df = file_df.append(grid_search_df)
        file_df.to_csv("grid_search.csv",index=False)

    writer.close()
    writer.flush()

    try:
        with open('genetic_best_network_mario.json') as json_file:
            data = json.load(json_file)
        lucro = int(data.get('lucro'))
    except:
        print('Error to load')

    if lucro<networks[0].lucro:
        best_net = {'actions': networks[0].get_actions(),'lucro':networks[0].lucro, 'generation':networks[0].generation, 'grid_search_iteration': iteration}
    
        with open('genetic_best_network_mario.json', 'w') as json_file:
            json.dump(best_net, json_file)
        json_file.close()

def main():
    population_values = [25] #50   10 -> mt pouco
    generations_values = [10] #100
    self_crossover_values = [False] 
    mutation_rate_values = [0.1, 0.05] #0.001 e 0.01 travou em ótimo local
    mutation_probability_values = [1] # 0.1, 0 muito baixos, não chegou a mutacionar mt #falta 0.5 e False p/ 0.05 e 0.1 m_r
    selection_percentage_values = [0.2]  #porcentagem dos melhores membros da população que irão pra próxima geração
    cromossome_size_values = [8000] #5000, 10000      #tamanho do cromossomo (numero de ações)
    
    population = 20
    generations = 10
    self_crossover = False
    mutation_rate = 0.05
    mutation_probability = 0.5
    selection_percentage = 0.2
    cromossome_size = 8000
    i=0
    #death_mutation = True
    #reduced_action_set = False
    #run(i, run_name, population, generations, self_crossover, mutation_rate, mutation_probability, selection_percentage, cromossome_size, death_mutation, reduced_action_set)
    #death_mutation = False
    #reduced_action_set = True
    #run_name = "run"+str(i)
    #run(i, run_name, population, generations, self_crossover, mutation_rate, mutation_probability, selection_percentage, cromossome_size, death_mutation, reduced_action_set)
    i=2
    run_name = "run"+str(i)
    death_mutation = False
    reduced_action_set = False
    run(i, run_name, population, generations, self_crossover, mutation_rate, mutation_probability, selection_percentage, cromossome_size, death_mutation, reduced_action_set)
    

if __name__ == '__main__':
    main()