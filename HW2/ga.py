import numpy as np
import random
from typing import List, Tuple, Optional

class GeneticAlgorithm:
    def __init__(
        self,
        pop_size: int,      # Population size
        generations: int,   # Number of generations for the algorithm
        mutation_rate: float,  # Gene mutation rate
        crossover_rate: float,  # Gene crossover rate
        tournament_size: int,  # Tournament size for selection
        elitism: bool,         # Whether to apply elitism strategy
        random_seed: Optional[int],  # Random seed for reproducibility
    ):
        # Students need to set the algorithm parameters
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.elitism = elitism

        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

    def _init_population(self, M: int, N: int) -> List[List[int]]:
        """
        Initialize the population and generate random individuals, ensuring that every student is assigned at least one task.
        :param M: Number of students
        :param N: Number of tasks
        :return: Initialized population
        """
        # TODO: Initialize individuals based on the number of students M and number of tasks N
        population = []
        for _ in range(self.pop_size):
            # 先確保每位學生都會分配到工作
            individual = list(range(1, M + 1))

            # 剩餘的工作用隨機分配給學生
            remaining_tasks = N - M
            for _ in range(remaining_tasks):
                individual.append(random.randint(1, M))

            # 將整個 list 打亂
            random.shuffle(individual)

            population.append(individual)

        return population
    

    def _fitness(self, individual: List[int], student_times: np.ndarray) -> float:
        """
        Fitness function: calculate the fitness value of an individual.
        :param individual: Individual
        :param student_times: Time required for each student to complete each task
        :return: Fitness value
        """
        # TODO: Design a fitness function to compute the fitness value of the allocation plan
        sum = 0

        for i in range(len(individual)):
            sum += float(student_times[individual[i]-1][i])

        return sum
    
    def _selection(self, population: List[List[int]], fitness_scores: List[float]) -> List[int]:
        """
        Use tournament selection to choose parents for crossover.
        :param population: Current population
        :param fitness_scores: Fitness scores for each individual
        :return: Selected parent
        """
        # TODO: Use tournament selection to choose parents based on fitness scores

        selected_parents = []

        for _ in range(self.pop_size):
            # 隨機抽取出 self.tournament_size 這麼多個 individual 的 index
            tournament_indices = np.random.choice(len(population), size=self.tournament_size, replace=False)
            
            # 計算抽取出的 individual 之 fitness scores
            fitness_tournament_scores = np.array([fitness_scores[i] for i in tournament_indices])
            
            # 從中選擇出最好的 fitness scores 之 individual
            best_index = tournament_indices[np.argmin(fitness_tournament_scores)]
            
            # 加入倒 paranet 這個 list
            selected_parents.append(population[best_index])

        return selected_parents

    def _crossover(self, parent1: List[int], parent2: List[int], M: int) -> Tuple[List[int], List[int]]:
        """
        Crossover: generate two offspring from two parents.
        :param parent1: Parent 1
        :param parent2: Parent 2
        :param M: Number of students
        :return: Generated offspring
        """
        # TODO: Complete the crossover operation to generate two offspring
        
        # Two-Point Crossover

        # 是否要 crossover，透過 crossover rate 決定
        if np.random.random() < self.crossover_rate:
            # 用隨機數決定出 Two-Point Crossover 要切的兩個位置
            point1 = np.random.randint(0, M - 1)  # 0 to M-2 to ensure at least one element on both sides
            point2 = np.random.randint(point1 + 1, M)  # point2 must be greater than point1

            # 組合切片後的基因片段
            offspring1 = parent1[:point1+1] + parent2[point1+1:point2+1] + parent1[point2+1:]
            offspring2 = parent2[:point1+1] + parent1[point1+1:point2+1] + parent2[point2+1:]

        else:
            # No crossover, offspring are identical to parents
            offspring1, offspring2 = parent1.copy(), parent2.copy()

        return offspring1, offspring2
    
    def _ensure_valid_assignment(self, individual: List[int], M: int) -> List[int]:
        """
        Ensure that the individual's task assignments are valid using NumPy.
        :param individual: Individual with potential duplicate assignments
        :param M: Number of students
        :return: Valid individual with no duplicates
        """
        individual = np.array(individual)
        

        counts = np.bincount(individual, minlength=M+1)[1:]
        
        # Find students who are not assigned any tasks (counts == 0)
        no_tasks_students = np.where(counts == 0)[0] + 1  # Add 1 to match student numbers (1 to M)
        
        # Find students who have the most tasks assigned
        while len(no_tasks_students) > 0:
            # 找出目前最多任務的學生
            max_task_student = np.argmax(counts) + 1
            
            # 找到該學生在 individual 中分配的任務
            max_task_indices = np.where(individual == max_task_student)[0]
            
            # 隨機選取其中一個任務來替換成沒有任務的學生
            replace_index = np.random.choice(max_task_indices)
            individual[replace_index] = no_tasks_students[0]
            
            # 更新 counts：減少一個任務給最多任務的學生，增加一個任務給沒有任務的學生
            counts[max_task_student - 1] -= 1
            counts[no_tasks_students[0] - 1] += 1
            
            # 刪除已經被分配任務的學生
            no_tasks_students = np.delete(no_tasks_students, 0)

        return individual.tolist()
        
    def _mutate(self, individual: List[int], M: int) -> List[int]:
        """
        Mutation operation: randomly change some genes (task assignments) of the individual using NumPy.
        :param individual: Individual (task assignments)
        :param M: Number of students
        :return: Mutated individual
        """
        # Perform mutation with a probability of mutation_rate
        if np.random.random() < self.mutation_rate:
            # Convert individual to a NumPy array for mutation
            individual = np.array(individual)
            
            # Select two random positions to swap
            pos1, pos2 = np.random.choice(len(individual), size=2, replace=False)
            
            # Swap the tasks assigned at pos1 and pos2
            individual[[pos1, pos2]] = individual[[pos2, pos1]]
            return individual.tolist()

        else:
            return individual
        

    def _elitism(self, population: List[List[int]], fitness_scores: List[float], elite_size: int = 1) -> List[List[int]]:
        """
        Apply elitism: retain the top elite_size individuals based on fitness scores.
        :param population: Current population
        :param fitness_scores: Fitness scores of the current population
        :param elite_size: Number of elite individuals to retain
        :return: List of elite individuals
        """
        # 排序 population based on fitness scores (越小越好)
        sorted_indices = np.argsort(fitness_scores)[:elite_size]
        elites = [population[i] for i in sorted_indices]
        return elites

    def __call__(self, M: int, N: int, student_times: np.ndarray) -> Tuple[List[int], int]:
        """
        Execute the genetic algorithm and return the optimal solution (allocation plan) and its total time cost.
        :param M: Number of students
        :param N: Number of tasks
        :param student_times: Time required for each student to complete each task
        :return: Optimal allocation plan and total time cost
        """
        # TODO: Complete the genetic algorithm process, including initialization, selection, crossover, mutation, and elitism strategy
        population = self._init_population(M, N)
        fitness_scores = []

        for generation in range(self.generations):
            # Step 1: 計算每個 individual 的 fitness 值
            fitness_scores = [self._fitness(individual, student_times) for individual in population]
            
            # Step 2: 選擇下一代的父母
            selected_parents = self._selection(population, fitness_scores)
            
            # Step 3: 交叉與變異生成後代
            offspring = []
            for i in range(0, self.pop_size, 2):
                parent1, parent2 = selected_parents[i], selected_parents[i+1]
                offspring1, offspring2 = self._crossover(parent1, parent2, M)
                offspring.append(offspring1)
                offspring.append(offspring2)

            # Step 4: 變異操作並確保每個學生都有分配任務
            for i in range(self.pop_size):
                offspring[i] = self._mutate(offspring[i], M)
                offspring[i] = self._ensure_valid_assignment(offspring[i], M)
            
            # Step 5: 如果啟用 elitism，保留最佳個體
            if self.elitism:
                elites = self._elitism(population, fitness_scores, elite_size=1)
                # 隨機替換 offspring 中的一些個體為 elites
                offspring[:len(elites)] = elites

            population = offspring
        # 最終世代的 fitness 與基因
        final_fitness_scores = [self._fitness(individual, student_times) for individual in population]
        best_index = np.argmin(final_fitness_scores)
        best_individual = population[best_index]
        best_fitness = final_fitness_scores[best_index]

        return best_individual, best_fitness
            
if __name__ == "__main__":
    def write_output_to_file(problem_num: int, total_time: int, filename: str = "results.txt") -> None:
        """
        Write results to a file and check if the format is correct.
        """
        print(f"Problem {problem_num}: Total time = {total_time}")

        if not isinstance(total_time, int) :
            raise ValueError(f"Invalid format for problem {problem_num}. Total time should be an integer.")
        
        with open(filename, 'a') as file:
            file.write(f"Total time = {total_time}\n")

    # TODO: Define multiple test problems based on the examples and solve them using the genetic algorithm
    # Example problem 1 (define multiple problems based on the given example format)
    # M, N = 2, 3
    # student_times = [[3, 8, 6],
    #                  [5, 2, 7]]

    M1, N1 = 2, 3
    cost1 = [[3, 2, 4],
             [4, 3, 2]]
    
    # M2, N2 = int, int
    # cost2 = List[List[int]]
    
    M3, N3 = 8, 9
    cost3 = [[90, 100, 60, 5, 50, 1, 100, 80, 70],
        [100, 5, 90, 100, 50, 70, 60, 90, 100],
        [50, 1, 100, 70, 90, 60, 80, 100, 4],
        [60, 100, 1, 80, 70, 90, 100, 50, 100],
        [70, 90, 50, 100, 100, 4, 1, 60, 80],
        [100, 60, 100, 90, 80, 5, 70, 100, 50],
        [100, 4, 80, 100, 90, 70, 50, 1, 60],
        [1, 90, 100, 50, 60, 80, 100, 70, 5]]
    
    # M4, N4 = int, int
    # cost4 = List[List[int]]
    
    # M5, N5 = int, int
    # cost5 = List[List[int]]
    
    # M6, N6 = int, int
    # cost6 = List[List[int]]
    
    # M7, N7 = int, int
    # cost7 = List[List[int]]
    
    # M8, N8 = int, int
    # cost8 = List[List[int]]
    
    # M9, N9 = int, int
    # cost9 = List[List[int]]
    
    # M10, N10 = int, int
    # cost10 = List[List[int]]

    # problems = [(M1, N1, np.array(cost1)),
    #             (M2, N2, np.array(cost2)),
    #             (M3, N3, np.array(cost3)),
    #             (M4, N4, np.array(cost4)),
    #             (M5, N5, np.array(cost5)),
    #             (M6, N6, np.array(cost6)),
    #             (M7, N7, np.array(cost7)),
    #             (M8, N8, np.array(cost8)),
    #             (M9, N9, np.array(cost9)),
    #             (M10, N10, np.array(cost10))]
    problems = [(M1, N1, np.array(cost1)),
                (M3, N3, np.array(cost3))]

    # Example for GA execution:
    # TODO: Please set the parameters for the genetic algorithm
    ga = GeneticAlgorithm(
        pop_size=100,
        generations=100,
        mutation_rate=0.05,
        crossover_rate=0.8,
        tournament_size=20,
        elitism=True,
        random_seed=10
    )

    # Solve each problem and immediately write the results to the file
    for i, (M, N, student_times) in enumerate(problems, 1):
        best_allocation, total_time = ga(M=M, N=N, student_times=student_times)
        write_output_to_file(i, int(total_time))

    print("Results have been written to results.txt")
