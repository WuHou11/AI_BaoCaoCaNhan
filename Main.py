import pygame
import time
import random
import heapq
import math
from collections import deque, defaultdict
import platform
import asyncio
import numpy as np

class Base:
    class EightPuzzle:
        def __init__(self, initial, goal):
            self.initial = tuple(map(tuple, initial))
            self.goal = tuple(map(tuple, goal))
            self.goal_state_index = {self.goal[i][j]: (i, j) for i in range(3) for j in range(3)}

        def is_solvable(self):
            flat_state = [num for row in self.initial for num in row if num != 0]
            inversions = 0
            for i in range(len(flat_state)):
                for j in range(i + 1, len(flat_state)):
                    if flat_state[i] > flat_state[j]:
                        inversions += 1
            return inversions % 2 == 0
        
        def find_blank(self, state):
            for i, row in enumerate(state):
                if 0 in row:
                    return i, row.index(0)

        def get_neighbors(self, state):
            row, col = self.find_blank(state)
            moves = []
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            state_list = [list(row) for row in state]

            for dr, dc in directions:
                new_row, new_col = row + dr, col + dc
                if 0 <= new_row < 3 and 0 <= new_col < 3:
                    new_state = [row[:] for row in state_list]
                    new_state[row][col], new_state[new_row][new_col] = new_state[new_row][new_col], new_state[row][col]
                    moves.append(tuple(map(tuple, new_state)))
            return moves
        
        def manhattan_distance(self, state):
            distance = 0
            for i in range(3):
                for j in range(3):
                    if state[i][j] != 0:
                        goal_x, goal_y = self.goal_state_index[state[i][j]]
                        distance += abs(i - goal_x) + abs(j - goal_y)
            return distance
        
        def bfs(self):
            queue = deque([(self.initial, [])])
            visited = {self.initial}
            while queue:
                state, path = queue.popleft()
                if state == self.goal:
                    return path + [state]
                for neighbor in self.get_neighbors(state):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, path + [state]))
            return None
        
        def dfs(self):
            stack = [(self.initial, [])]
            visited = {self.initial}
            while stack:
                state, path = stack.pop()
                if state == self.goal:
                    return path + [state]
                for neighbor in self.get_neighbors(state):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        stack.append((neighbor, path + [state]))
            return None
        
        def ucs(self):
            pq = []
            heapq.heappush(pq, (0, self.initial, []))
            visited = set()
            while pq:
                cost, state, path = heapq.heappop(pq)
                if state == self.goal:
                    return path + [state]
                if state not in visited:
                    visited.add(state)
                    for neighbor in self.get_neighbors(state):
                        if neighbor not in visited:
                            new_cost = len(path) + 1
                            heapq.heappush(pq, (new_cost, neighbor, path + [state]))
            return None
        
        def iterative_deepening_search(self):
            def dls(state, path, depth, visited):
                if state == self.goal:
                    return path + [state]
                if depth <= 0:
                    return None
                for neighbor in self.get_neighbors(state):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        result = dls(neighbor, path + [state], depth - 1, visited)
                        if result:
                            return result
                return None

            depth = 0
            while True:
                visited = {self.initial}
                result = dls(self.initial, [], depth, visited)
                if result:
                    return result
                depth += 1
        
        def greedy_search(self):
            pq = []
            heapq.heappush(pq, (0, self.initial, []))
            visited = set()
            while pq:
                _, state, path = heapq.heappop(pq)
                if state == self.goal:
                    return path + [state]
                if state not in visited:
                    visited.add(state)
                    for neighbor in self.get_neighbors(state):
                        if neighbor not in visited:
                            h = self.manhattan_distance(neighbor)
                            heapq.heappush(pq, (h, neighbor, path + [state]))
            return None
        
        def a_star_search(self):
            pq = []
            heapq.heappush(pq, (0, self.initial, []))
            visited = set()
            while pq:
                cost, state, path = heapq.heappop(pq)
                if state == self.goal:
                    return path + [state]
                visited.add(state)
                for neighbor in self.get_neighbors(state):
                    if neighbor not in visited:
                        g = len(path) + 1
                        h = self.manhattan_distance(neighbor)
                        heapq.heappush(pq, (g + h, neighbor, path + [state]))
            return None
        
        def ida_star(self):
            def search(state, g, threshold, path, visited):
                h = self.manhattan_distance(state)
                f = g + h
                if f > threshold:
                    return f, None
                if state == self.goal:
                    return f, path + [state]
                min_threshold = float('inf')
                for neighbor in self.get_neighbors(state):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        new_threshold, result = search(neighbor, g + 1, threshold, path + [state], visited)
                        visited.remove(neighbor)
                        if result:
                            return new_threshold, result
                        min_threshold = min(min_threshold, new_threshold)
                return min_threshold, None

            threshold = self.manhattan_distance(self.initial)
            while True:
                visited = {self.initial}
                new_threshold, result = search(self.initial, 0, threshold, [], visited)
                if result:
                    return result
                if new_threshold == float('inf'):
                    return None
                threshold = new_threshold

        def simple_hill_climbing(self):
            current = self.initial
            path = [current]
            visited = {current}
            
            while current != self.goal:
                neighbors = self.get_neighbors(current)
                best_neighbor = None
                best_h = float('inf')
                
                for neighbor in neighbors:
                    if neighbor not in visited:
                        h = self.manhattan_distance(neighbor)
                        if h < best_h:
                            best_h = h
                            best_neighbor = neighbor
                
                if best_neighbor is None or best_h >= self.manhattan_distance(current):
                    return None 
                current = best_neighbor
                visited.add(current)
                path.append(current)
            
            return path

        def steepest_ascent_hill_climbing(self):
            current = self.initial
            path = [current]
            visited = {current}
            
            while current != self.goal:
                neighbors = self.get_neighbors(current)
                best_neighbor = None
                best_h = float('inf')
                
                for neighbor in neighbors:
                    if neighbor not in visited:
                        h = self.manhattan_distance(neighbor)
                        if h < best_h:
                            best_h = h
                            best_neighbor = neighbor
                
                if best_neighbor is None:
                    return None  
                current = best_neighbor
                visited.add(current)
                path.append(current)
            
            return path

        def stochastic_hill_climbing(self):
            current = self.initial
            path = [current]
            visited = {current}
            max_steps = 1000  
            
            for _ in range(max_steps):
                if current == self.goal:
                    return path
                neighbors = [n for n in self.get_neighbors(current) if n not in visited]
                if not neighbors:
                    return None

                better_neighbors = [(n, self.manhattan_distance(n)) for n in neighbors 
                                  if self.manhattan_distance(n) < self.manhattan_distance(current)]
                if better_neighbors:
                    current = random.choice(better_neighbors)[0]
                else:
                    current = random.choice(neighbors) 
                
                visited.add(current)
                path.append(current)
            
            return None  

        def simulated_annealing(self):
            current = self.initial
            path = [current]
            temperature = 1000.0
            cooling_rate = 0.995
            min_temperature = 0.001
            max_steps = 100000
            
            current_cost = self.manhattan_distance(current)
            
            while temperature > min_temperature and len(path) < max_steps:
                neighbors = self.get_neighbors(current)
                if not neighbors:
                    return None
                    
                next_state = random.choice(neighbors)
                next_cost = self.manhattan_distance(next_state)
                
                cost_diff = next_cost - current_cost
                
                if cost_diff < 0 or random.random() < math.exp(-cost_diff / temperature):
                    current = next_state
                    current_cost = next_cost
                    path.append(current)
                    
                    if current == self.goal:
                        return path
                        
                temperature *= cooling_rate
            
            return None if current != self.goal else path

        def beam_search(self, beam_width=3):
            beam = [(self.manhattan_distance(self.initial), self.initial, [self.initial])]
            visited = {self.initial}
            
            while beam:
                new_beam = []
                
                for _, state, path in beam:
                    if state == self.goal:
                        return path
                    
                    for neighbor in self.get_neighbors(state):
                        if neighbor not in visited:
                            visited.add(neighbor)
                            h = self.manhattan_distance(neighbor)
                            new_beam.append((h, neighbor, path + [neighbor]))
                
                new_beam.sort(key=lambda x: x[0])
                beam = new_beam[:beam_width]
                
                if not beam:
                    return None
            
            return None
        
        def genetic_algorithm(self, population_size=100, generations=1000, mutation_rate=0.1):
            def generate_random_state():
                flat_state = [0, 1, 2, 3, 4, 5, 6, 7, 8]
                random.shuffle(flat_state)
                return tuple(tuple(flat_state[i * 3:(i + 1) * 3]) for i in range(3))

            def fitness(state):
                return -self.manhattan_distance(state)

            def crossover(parent1, parent2):
                p1 = [num for row in parent1 for num in row]
                p2 = [num for row in parent2 for num in row]
                point1, point2 = sorted(random.sample(range(9), 2))
                child = p1[:point1] + p2[point1:point2] + p1[point2:]
                missing = set(range(9)) - set(child)
                seen = set()
                for i in range(9):
                    if child[i] in seen:
                        child[i] = missing.pop()
                    else:
                        seen.add(child[i])
                return tuple(tuple(child[i * 3:(i + 1) * 3]) for i in range(3))

            def mutate(state, rate):
                if random.random() < rate:
                    flat_state = [num for row in state for num in row]
                    i, j = random.sample(range(9), 2)
                    flat_state[i], flat_state[j] = flat_state[j], flat_state[i]
                    return tuple(tuple(flat_state[i * 3:(i + 1) * 3]) for i in range(3))
                return state

            population = [generate_random_state() for _ in range(population_size - 1)]
            population.append(self.initial)
            path = [self.initial]

            for _ in range(generations):
                fitness_scores = [(fitness(state), state) for state in population]
                fitness_scores.sort(reverse=True)
                
                best_state = fitness_scores[0][1]
                if best_state not in path:
                    path.append(best_state)
                
                if best_state == self.goal:
                    return path
                
                new_population = [best_state]
                while len(new_population) < population_size:
                    parent1, parent2 = random.choices([state for _, state in fitness_scores[:10]], k=2)
                    child = crossover(parent1, parent2)
                    child = mutate(child, mutation_rate)
                    new_population.append(child)
                
                population = new_population[:population_size]
            
            return None
        
        def and_or_tree_search(self):
            def or_node(belief, path, visited):
                if frozenset(belief) in visited:
                    return None
                visited.add(frozenset(belief))

                for state in belief:
                    if state == self.goal:
                        return path

                actions = set()
                for state in belief:
                    row, col = self.find_blank(state)
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        new_row, new_col = row + dr, col + dc
                        if 0 <= new_row < 3 and 0 <= new_col < 3:
                            actions.add((dr, dc))
                            
                for action in actions:
                    new_belief = set()
                    for state in belief:
                        row, col = self.find_blank(state)
                        new_row, new_col = row + action[0], col + action[1]
                        if 0 <= new_row < 3 and 0 <= new_col < 3:
                            state_list = [list(row) for row in state]
                            state_list[row][col], state_list[new_row][new_col] = (
                                state_list[new_row][new_col],
                                state_list[row][col]
                            )
                            new_state = tuple(map(tuple, state_list))
                            new_belief.add(new_state)

                    if not new_belief:
                        continue

                    result = and_node(new_belief, path + [new_belief], visited.copy())
                    if result is not None:
                        return result

                return None

            def and_node(belief, path, visited):
                return or_node(belief, path, visited)

            initial_belief = {self.initial}
            visited = set()
            result = or_node(initial_belief, [], visited)

            if result:
                state_path = [self.initial]
                for belief in result:
                    state = min(belief, key=self.manhattan_distance)
                    state_path.append(state)
                return state_path

            return None
        
        def q_learning(self, episodes=1000, alpha=0.1, gamma=0.9, epsilon=0.1):
            if not self.is_solvable():
                return None

            Q = {}
            
            def state_to_tuple(state):
                return tuple(tuple(row) for row in state)

            def get_valid_actions(self, state):
                row, col = self.find_blank(state)
                valid_actions = []
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    new_row, new_col = row + dr, col + dc
                    if 0 <= new_row < 3 and 0 <= new_col < 3:
                        valid_actions.append((dr, dc))
                return valid_actions

            def perform_action(self, state, action):
                row, col = self.find_blank(state)
                new_row, new_col = row + action[0], col + action[1]
                if not (0 <= new_row < 3 and 0 <= new_col < 3):
                    return state
                state_list = [list(row) for row in state]
                state_list[row][col], state_list[new_row][new_col] = state_list[new_row][new_col], state_list[row][col]
                return tuple(map(tuple, state_list))

            def get_reward(self, state):
                if state == self.goal:
                    return 100
                return -1 - self.manhattan_distance(state) / 10

            for episode in range(episodes):
                current_state = self.initial
                while current_state != self.goal:
                    current_state_tuple = state_to_tuple(current_state)
                    if current_state_tuple not in Q:
                        Q[current_state_tuple] = {action: 0 for action in get_valid_actions(self, current_state)}

                    valid_actions = get_valid_actions(self, current_state)
                    if not valid_actions:
                        break
                    if random.random() < epsilon:
                        action = random.choice(valid_actions)
                    else:
                        action_scores = {a: Q[current_state_tuple][a] for a in valid_actions}
                        action = max(action_scores, key=action_scores.get)

                    next_state = perform_action(self, current_state, action)
                    next_state_tuple = state_to_tuple(next_state)
                    if next_state_tuple not in Q:
                        Q[next_state_tuple] = {action: 0 for action in get_valid_actions(self, next_state)}

                    reward = get_reward(self, next_state)
                    best_next_action = max(Q[next_state_tuple].values()) if Q[next_state_tuple] else 0
                    Q[current_state_tuple][action] += alpha * (
                        reward + gamma * best_next_action - Q[current_state_tuple][action]
                    )
                    current_state = next_state

            path = [self.initial]
            current_state = self.initial
            visited = set()
            max_steps = 1000

            while current_state != self.goal and state_to_tuple(current_state) not in visited and len(path) < max_steps:
                visited.add(state_to_tuple(current_state))
                current_state_tuple = state_to_tuple(current_state)
                if current_state_tuple not in Q:
                    return None
                valid_actions = get_valid_actions(self, current_state)
                if not valid_actions:
                    return None
                action_scores = {a: Q[current_state_tuple][a] for a in valid_actions}
                action = max(action_scores, key=action_scores.get)
                current_state = perform_action(self, current_state, action)
                path.append(current_state)

            if current_state == self.goal:
                return path
            return None

    def draw_grid(self, screen, state, tile_size, offset_x, offset_y):
        WHITE, BLACK = (255, 255, 255), (0, 0, 0)
        pygame.draw.rect(screen, WHITE, (offset_x, offset_y, 240, 240))
        font = pygame.font.Font(None, 50)
        for i in range(3):
            for j in range(3):
                if state[i][j] != 0:
                    pygame.draw.rect(screen, (0, 100, 0), (offset_x + j * tile_size, offset_y + i * tile_size, tile_size, tile_size))
                    text = font.render(str(state[i][j]), True, BLACK)
                    screen.blit(text, (offset_x + j * tile_size + 35, offset_y + i * tile_size + 25))
        for i in range(4):
            pygame.draw.line(screen, BLACK, (offset_x, offset_y + i * tile_size), (offset_x + 240, offset_y + i * tile_size), 2)
            pygame.draw.line(screen, BLACK, (offset_x + i * tile_size, offset_y), (offset_x + i * tile_size, offset_y + 240), 2)

    def print_solution(self, solution):
        if solution is None:
            print("Không tìm thấy đường đi đến trạng thái mục tiêu!")
            return
        for step, state in enumerate(solution):
            print(f"Bước {step}:")
            for row in state:
                print(row)
            print()
        print("Hoàn thành!")

    def parse_input(self, text):
        try:
            numbers = [int(x) for x in text.replace(',', ' ').split() if x.strip().isdigit()]
            if len(numbers) != 9 or set(numbers) != set(range(9)):
                return None
            return [[numbers[i * 3 + j] for j in range(3)] for i in range(3)]
        except:
            return None

    async def run(self):
        pygame.init()
        screen = pygame.display.set_mode((800, 700))
        pygame.display.set_caption("8-Puzzle Solver (Program A)")
        clock = pygame.time.Clock()
        
        #initial_state = [[1, 2, 3], [4, 0, 6], [7, 5, 8]]
        initial_state = [[1, 2, 0], [5, 6, 3], [4, 7, 8]]
        goal_state = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
        puzzle = self.EightPuzzle(initial_state, goal_state)
        
        buttons = [
            ("BFS", (20, 160)),
            ("DFS", (20, 210)),
            ("UCS", (20, 260)),
            ("IDS", (20, 310)),
            ("Greedy Search", (20, 360)),
            ("A*", (20, 410)),
            ("AND-OR Search", (20, 460)),
            ("Q-Learning", (20, 510)),
            ("IDA*", (230, 160)),
            ("Simple HC", (230, 210)),
            ("Steepest HC", (230, 260)),
            ("Stochastic HC", (230, 310)),
            ("S A", (230, 360)),
            ("Beam Search", (230, 410)),
            ("GA", (230, 460)),
            ("Reset", (230, 510)),
            ("Back", (230, 560)),
        ]
        
        font = pygame.font.Font(None, 36)
        input_font = pygame.font.Font(None, 30)
        
        initial_input = ""
        goal_input = ""
        active_input = None
        
        initial_rect = pygame.Rect(20, 50, 410, 40)
        goal_rect = pygame.Rect(20, 110, 410, 40)
        
        solution = None
        step = 0
        elapsed_time = None
        error_message = ""
        
        running = True
        while running:
            screen.fill((255, 255, 255))
            
            pygame.draw.rect(screen, (0, 0, 0), initial_rect, 2)
            pygame.draw.rect(screen, (0, 0, 0), goal_rect, 2)
            
            initial_label = input_font.render("Initial state (e.g., 1 2 3 4 0 6 7 5 8):", True, (0, 0, 0))
            goal_label = input_font.render("Goal state (e.g., 1 2 3 4 5 6 7 8 0):", True, (0, 0, 0))
            screen.blit(initial_label, (20, 30))
            screen.blit(goal_label, (20, 90))
            
            initial_text = input_font.render(initial_input, True, (0, 0, 0))
            goal_text = input_font.render(goal_input, True, (0, 0, 0))
            screen.blit(initial_text, (initial_rect.x + 5, initial_rect.y + 10))
            screen.blit(goal_text, (goal_rect.x + 5, goal_rect.y + 10))
            
            if error_message:
                error_text = input_font.render(error_message, True, (255, 0, 0))
                screen.blit(error_text, (20, 150))
            
            for text, pos in buttons:
                rect = pygame.draw.rect(screen, (139, 0, 0), (*pos, 200, 40))
                button_width, button_height = 200, 40
                label = font.render(text, True, (0, 0, 0))
                text_width, text_height = label.get_size()
                text_x = pos[0] + (button_width - text_width) // 2
                text_y = pos[1] + (button_height - text_height) // 2
                screen.blit(label, (text_x, text_y))
            
            solution_label = font.render("Initial_state", True, (0, 0, 0))
            screen.blit(solution_label, (500, 10))
            if solution:
                self.draw_grid(screen, solution[step], 80, 500, 50)
                if step < len(solution) - 1:
                    step += 1
                    await asyncio.sleep(0.01)
            else:
                self.draw_grid(screen, puzzle.initial, 80, 500, 50)

            goal_label = font.render("Goal_state", True, (0, 0, 0))
            screen.blit(goal_label, (500, 310))
            self.draw_grid(screen, puzzle.goal, 80, 500, 350)
                
            if elapsed_time is not None:
                time_text = font.render(f"Thoi gian thuc thi: {elapsed_time:.4f}s", True, (0, 0, 0))
                screen.blit(time_text, (450, 600))
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = event.pos
                    if initial_rect.collidepoint(x, y):
                        active_input = "initial"
                    elif goal_rect.collidepoint(x, y):
                        active_input = "goal"
                    else:
                        active_input = None
                    
                    start_time = time.time()
                    if 20 <= x <= 220 and 160 <= y <= 200:  # BFS
                        solution = puzzle.bfs()
                        step = 0
                        end_time = time.time()
                        elapsed_time = end_time - start_time
                        print("\nKết quả của thuật toán BFS:")
                        print(f"Thời gian thực thi: {elapsed_time:.10f} giây")
                        self.print_solution(solution)
                    elif 20 <= x <= 220 and 210 <= y <= 250:  # DFS
                        solution = puzzle.dfs()
                        step = 0
                        end_time = time.time()
                        elapsed_time = end_time - start_time
                        print("\nKết quả của thuật toán DFS:")
                        print(f"Thời gian thực thi: {elapsed_time:.10f} giây")
                        self.print_solution(solution)
                    elif 20 <= x <= 220 and 260 <= y <= 300:  # UCS
                        solution = puzzle.ucs()
                        step = 0
                        end_time = time.time()
                        elapsed_time = end_time - start_time
                        print("\nKết quả của thuật toán UCS:")
                        print(f"Thời gian thực thi: {elapsed_time:.10f} giây")
                        self.print_solution(solution)
                    elif 20 <= x <= 220 and 310 <= y <= 350:  # IDS
                        solution = puzzle.iterative_deepening_search()
                        step = 0
                        end_time = time.time()
                        elapsed_time = end_time - start_time
                        print("\nKết quả của thuật toán Iterative Deepening Search:")
                        print(f"Thời gian thực thi: {elapsed_time:.10f} giây")
                        self.print_solution(solution)
                    elif 20 <= x <= 220 and 360 <= y <= 400:  # Greedy Search
                        solution = puzzle.greedy_search()
                        step = 0
                        end_time = time.time()
                        elapsed_time = end_time - start_time
                        print("\nKết quả của thuật toán Greedy Search:")
                        print(f"Thời gian thực thi: {elapsed_time:.10f} giây")
                        self.print_solution(solution)
                    elif 20 <= x <= 220 and 410 <= y <= 450:  # A*
                        solution = puzzle.a_star_search()
                        step = 0
                        end_time = time.time()
                        elapsed_time = end_time - start_time
                        print("\nKết quả của thuật toán A*:")
                        print(f"Thời gian thực thi: {elapsed_time:.10f} giây")
                        self.print_solution(solution)
                    elif 20 <= x <= 220 and 460 <= y <= 500:  # AND-OR Search
                        solution = puzzle.and_or_tree_search()
                        step = 0
                        end_time = time.time()
                        elapsed_time = end_time - start_time
                        print("\nKết quả của thuật toán AND-OR Tree Search:")
                        print(f"Thời gian thực thi: {elapsed_time:.10f} giây")
                        self.print_solution(solution)
                    elif 20 <= x <= 220 and 510 <= y <= 550:  # Q-Learning
                        solution = puzzle.q_learning()
                        step = 0
                        end_time = time.time()
                        elapsed_time = end_time - start_time
                        print("\nKết quả của thuật toán Q-Learning:")
                        print(f"Thời gian thực thi: {elapsed_time:.10f} giây")
                        self.print_solution(solution)
                    elif 230 <= x <= 430 and 160 <= y <= 200:  # IDA*
                        solution = puzzle.ida_star()
                        step = 0
                        end_time = time.time()
                        elapsed_time = end_time - start_time
                        print("\nKết quả của thuật toán IDA*:")
                        print(f"Thời gian thực thi: {elapsed_time:.10f} giây")
                        self.print_solution(solution)
                    elif 230 <= x <= 430 and 210 <= y <= 250:  # Simple HC
                        solution = puzzle.simple_hill_climbing()
                        step = 0
                        end_time = time.time()
                        elapsed_time = end_time - start_time
                        print("\nKết quả của thuật toán Simple Hill Climbing:")
                        print(f"Thời gian thực thi: {elapsed_time:.10f} giây")
                        self.print_solution(solution)
                    elif 230 <= x <= 430 and 260 <= y <= 300:  # Steepest HC
                        solution = puzzle.steepest_ascent_hill_climbing()
                        step = 0
                        end_time = time.time()
                        elapsed_time = end_time - start_time
                        print("\nKết quả của thuật toán Steepest Ascent Hill Climbing:")
                        print(f"Thời gian thực thi: {elapsed_time:.10f} giây")
                        self.print_solution(solution)
                    elif 230 <= x <= 430 and 310 <= y <= 350:  # Stochastic HC
                        solution = puzzle.stochastic_hill_climbing()
                        step = 0
                        end_time = time.time()
                        elapsed_time = end_time - start_time
                        print("\nKết quả của thuật toán Stochastic Hill Climbing:")
                        print(f"Thời gian thực thi: {elapsed_time:.10f} giây")
                        self.print_solution(solution)
                    elif 230 <= x <= 430 and 360 <= y <= 400:  # Simulated Annealing
                        solution = puzzle.simulated_annealing()
                        step = 0
                        end_time = time.time()
                        elapsed_time = end_time - start_time
                        print("\nKết quả của thuật toán Simulated Annealing:")
                        print(f"Thời gian thực thi: {elapsed_time:.10f} giây")
                        self.print_solution(solution)
                    elif 230 <= x <= 430 and 410 <= y <= 450:  # Beam Search
                        solution = puzzle.beam_search()
                        step = 0
                        end_time = time.time()
                        elapsed_time = end_time - start_time
                        print("\nKết quả của thuật toán Beam Search:")
                        print(f"Thời gian thực thi: {elapsed_time:.10f} giây")
                        self.print_solution(solution)
                    elif 230 <= x <= 430 and 460 <= y <= 500:  # GA
                        solution = puzzle.genetic_algorithm()
                        step = 0
                        end_time = time.time()
                        elapsed_time = end_time - start_time
                        print("\nKết quả của thuật toán Genetic Algorithm:")
                        print(f"Thời gian thực thi: {elapsed_time:.10f} giây")
                        self.print_solution(solution)
                    elif 230 <= x <= 430 and 510 <= y <= 550:  # Reset
                        solution = None
                        step = 0
                        elapsed_time = None
                        continue
                    elif 20 <= x <= 220 and 220 <= y <= 260:  # Back
                        running = False

                elif event.type == pygame.KEYDOWN and active_input:
                    if event.key == pygame.K_BACKSPACE:
                        if active_input == "initial":
                            initial_input = initial_input[:-1]
                        elif active_input == "goal":
                            goal_input = goal_input[:-1]
                    elif event.key == pygame.K_RETURN:
                        new_initial = self.parse_input(initial_input)
                        new_goal = self.parse_input(goal_input)
                        if new_initial and new_goal:
                            initial_state = new_initial
                            goal_state = new_goal
                            puzzle = self.EightPuzzle(initial_state, goal_state)
                            error_message = ""
                            solution = None
                            step = 0
                            initial_input = ""
                            goal_input = ""
                        else:
                            error_message = "Invalid input! Use 9 unique digits (0-8), e.g., '1 2 3 4 0 6 7 5 8'"
                    elif event.unicode.isprintable():
                        if active_input == "initial":
                            initial_input += event.unicode
                        elif active_input == "goal":
                            goal_input += event.unicode
            
            pygame.display.flip()
            clock.tick(60)
            await asyncio.sleep(1.0 / 60)
        
        pygame.quit()

class CSPs:
    class EightPuzzle:
        def __init__(self, initial, goal):
            self.initial = tuple(map(tuple, initial))
            self.goal = tuple(map(tuple, goal))

        def is_valid_assignment(self, state, pos, value):
            i, j = pos
            for r in range(3):
                for c in range(3):
                    if (r, c) != pos and state[r][c] == value:
                        return False
            if j > 0 and state[i][j - 1] != 0 and value != 0 and state[i][j - 1] != value - 1:
                return False
            if j < 2 and value != 0 and state[i][j + 1] != 0 and state[i][j + 1] != value + 1:
                return False
            if i > 0 and state[i - 1][j] != 0 and value != 0 and state[i - 1][j] != value - 3:
                return False
            if i < 2 and value != 0 and state[i + 1][j] != 0 and state[i + 1][j] != value + 3:
                return False
            return True

        def is_solvable(self, state):
            flat = [state[i][j] for i in range(3) for j in range(3) if state[i][j] != 0]
            inversions = 0
            for i in range(len(flat)):
                for j in range(i + 1, len(flat)):
                    if flat[i] > flat[j]:
                        inversions += 1
            return inversions % 2 == 0

        def count_conflicts(self, state):
            conflicts = 0
            value_counts = defaultdict(int)

            for i in range(3):
                for j in range(3):
                    val = state[i][j]
                    if val != 0:
                        value_counts[val] += 1
                        if value_counts[val] > 1:
                            conflicts += 2 * (value_counts[val] - 1)

            for i in range(3):
                for j in range(2):
                    if state[i][j] != 0 and state[i][j + 1] != 0:
                        if state[i][j + 1] != state[i][j] + 1:
                            conflicts += 1

            for j in range(3):
                for i in range(2):
                    if state[i][j] != 0 and state[i + 1][j] != 0:
                        if state[i + 1][j] != state[i][j] + 3:
                            conflicts += 1

            if not self.is_solvable(state):
                conflicts += 10

            return conflicts

        def backtracking_search(self):
            visited = set()
            path = []

            def backtrack(state, assigned, pos_index):
                if pos_index == 9:
                    state_tuple = tuple(tuple(row) for row in state)
                    if state_tuple == self.goal and self.is_solvable(state):
                        path.append(state_tuple)
                        return path
                    return None

                i, j = divmod(pos_index, 3)
                state_tuple = tuple(tuple(row) for row in state)
                if state_tuple in visited:
                    return None
                visited.add(state_tuple)
                path.append(state_tuple)

                for value in [1, 2, 3, 4, 5, 6, 7, 8, 0]:
                    if value not in assigned:
                        if self.is_valid_assignment(state, (i, j), value):
                            new_state = [row[:] for row in state]
                            new_state[i][j] = value
                            new_assigned = assigned | {value}
                            result = backtrack(new_state, new_assigned, pos_index + 1)
                            if result is not None:
                                return result
                path.pop()
                return None

            empty_state = [[0 for _ in range(3)] for _ in range(3)]
            result = backtrack(empty_state, set(), 0)
            return result

        def forward_checking_search(self):
            visited = set()
            path = []

            def get_domain(state, pos, assigned):
                i, j = pos
                domain = []
                for value in [0, 1, 2, 3, 4, 5, 6, 7, 8]:
                    if value not in assigned and self.is_valid_assignment(state, (i, j), value):
                        domain.append(value)
                return domain

            def forward_check(state, pos, value, domains, assigned):
                i, j = pos
                new_domains = {k: v[:] for k, v in domains.items()}
                success = True
                for r in range(3):
                    for c in range(3):
                        if (r, c) not in assigned and (r, c) != pos:
                            if (r, c) in new_domains:
                                new_domain = new_domains[(r, c)][:]
                                if value != 0 and value in new_domain:
                                    if r == i:
                                        if c < j and value - 1 in new_domain:
                                            new_domain = [v for v in new_domain if v == 0 or v == value - 1]
                                        elif c > j and value + 1 in new_domain:
                                            new_domain = [v for v in new_domain if v == 0 or v == value + 1]
                                    if c == j:
                                        if r < i and value - 3 in new_domain:
                                            new_domain = [v for v in new_domain if v == 0 or v == value - 3]
                                        elif r > i and value + 3 in new_domain:
                                            new_domain = [v for v in new_domain if v == 0 or v == value + 3]
                                    if value in new_domain:
                                        new_domain.remove(value)
                                new_domains[(r, c)] = new_domain
                                if not new_domain:
                                    success = False
                return success, new_domains

            def select_mrv_variable(positions, domains):
                min_domain_size = float('inf')
                selected_pos = None
                for pos in positions:
                    if pos in domains:
                        domain_size = len(domains[pos])
                        if domain_size < min_domain_size:
                            min_domain_size = domain_size
                            selected_pos = pos
                return selected_pos

            def select_lcv_value(pos, domain, state, domains, assigned):
                value_scores = []
                for value in domain:
                    temp_state = [row[:] for row in state]
                    temp_state[pos[0]][pos[1]] = value
                    _, new_domains = forward_check(temp_state, pos, value, domains, assigned)
                    remaining = sum(len(new_domains[p]) for p in new_domains if p != pos)
                    if value == self.goal[pos[0]][pos[1]]:
                        remaining += 10
                    value_scores.append((-remaining, value))
                value_scores.sort()
                return [value for _, value in value_scores]

            def backtrack_with_fc(state, assigned, positions, domains):
                if len(assigned) == 9:
                    state_tuple = tuple(tuple(row) for row in state)
                    if state_tuple == self.goal and self.is_solvable(state):
                        path.append(state_tuple)
                        return path
                    return None

                pos = select_mrv_variable(positions, domains)
                if pos is None:
                    return None

                domain = get_domain(state, pos, set(assigned.values()))
                if not domain:
                    return None

                sorted_values = select_lcv_value(pos, domain, state, domains, assigned)

                state_tuple = tuple(tuple(row) for row in state)
                if state_tuple in visited:
                    return None
                visited.add(state_tuple)
                path.append(state_tuple)

                for value in sorted_values:
                    new_state = [row[:] for row in state]
                    new_state[pos[0]][pos[1]] = value
                    new_assigned = assigned.copy()
                    new_assigned[pos] = value
                    new_positions = [p for p in positions if p != pos]
                    success, new_domains = forward_check(new_state, pos, value, domains, new_assigned)
                    if success:
                        result = backtrack_with_fc(new_state, new_assigned, new_positions, new_domains)
                        if result is not None:
                            return result
                    new_domains.pop(pos, None)
                path.pop()
                return None

            empty_state = [[0 for _ in range(3)] for _ in range(3)]
            positions = [(i, j) for i in range(3) for j in range(3)]
            domains = {(i, j): [0, 1, 2, 3, 4, 5, 6, 7, 8] for i in range(3) for j in range(3)}
            assigned = {}
            result = backtrack_with_fc(empty_state, assigned, positions, domains)
            return result

        def min_conflicts_search(self, max_steps=1000):
            def find_blank(state):
                for i in range(3):
                    for j in range(3):
                        if state[i][j] == 0:
                            return i, j
                return None

            def get_neighbors(i, j):
                neighbors = []
                directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
                for di, dj in directions:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < 3 and 0 <= nj < 3:
                        neighbors.append((ni, nj))
                return neighbors

            def count_conflicts(state):
                conflicts = 0
                goal_positions = {self.goal[i][j]: (i, j) for i in range(3) for j in range(3) if self.goal[i][j] != 0}
                for i in range(3):
                    for j in range(3):
                        if state[i][j] != 0:
                            goal_i, goal_j = goal_positions.get(state[i][j], (i, j))
                            conflicts += abs(i - goal_i) + abs(j - goal_j)
                return conflicts

            def is_valid_state(state):
                flat = [state[i][j] for i in range(3) for j in range(3)]
                return sorted(flat) == [0, 1, 2, 3, 4, 5, 6, 7, 8]

            if is_valid_state(self.initial) and self.is_solvable(self.initial):
                current_state = [row[:] for row in self.initial]
            else:
                numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8]
                random.shuffle(numbers)
                current_state = [[0 for _ in range(3)] for _ in range(3)]
                idx = 0
                for i in range(3):
                    for j in range(3):
                        current_state[i][j] = numbers[idx]
                        idx += 1
                while not self.is_solvable(current_state):
                    random.shuffle(numbers)
                    idx = 0
                    for i in range(3):
                        for j in range(3):
                            current_state[i][j] = numbers[idx]
                            idx += 1

            path = [tuple(tuple(row) for row in current_state)]
            steps = 0
            visited = set([tuple(tuple(row) for row in current_state)])

            while steps < max_steps:
                current_state_tuple = tuple(tuple(row) for row in current_state)
                if current_state_tuple == self.goal:
                    print(f"Solution found after {steps} iterations")
                    return path

                blank_i, blank_j = find_blank(current_state)
                if blank_i is None:
                    print("Invalid state: No blank tile found")
                    return None

                neighbors = get_neighbors(blank_i, blank_j)
                if not neighbors:
                    print("No valid moves available")
                    return None

                best_conflicts = float('inf')
                best_states = []

                for ni, nj in neighbors:
                    temp_state = [row[:] for row in current_state]
                    temp_state[blank_i][blank_j], temp_state[ni][nj] = temp_state[ni][nj], temp_state[blank_i][blank_j]
                    temp_state_tuple = tuple(tuple(row) for row in temp_state)
                    if temp_state_tuple in visited:
                        continue
                    conflicts = count_conflicts(temp_state)
                    if conflicts < best_conflicts:
                        best_conflicts = conflicts
                        best_states = [(temp_state, (ni, nj))]
                    elif conflicts == best_conflicts:
                        best_states.append((temp_state, (ni, nj)))

                if not best_states:
                    ni, nj = random.choice(neighbors)
                    temp_state = [row[:] for row in current_state]
                    temp_state[blank_i][blank_j], temp_state[ni][nj] = temp_state[ni][nj], temp_state[blank_i][blank_j]
                    best_states = [(temp_state, (ni, nj))]

                best_state, _ = random.choice(best_states)
                current_state = best_state
                current_state_tuple = tuple(tuple(row) for row in current_state)
                visited.add(current_state_tuple)
                path.append(current_state_tuple)

                steps += 1

            print("No solution found within max_steps")
            return None
        

    def draw_grid(self, screen, state, tile_size, offset_x, offset_y):
        WHITE, BLACK = (255, 255, 255), (0, 0, 0)
        pygame.draw.rect(screen, WHITE, (offset_x, offset_y, 300, 300))
        font = pygame.font.Font(None, 50)
        for i in range(3):
            for j in range(3):
                pygame.draw.rect(screen, WHITE, (offset_x + j * tile_size, offset_y + i * tile_size, tile_size, tile_size))
                if state[i][j] != 0:
                    pygame.draw.rect(screen, (0, 100, 0), (offset_x + j * tile_size, offset_y + i * tile_size, tile_size, tile_size))
                    text = font.render(str(state[i][j]), True, BLACK)
                    screen.blit(text, (offset_x + j * tile_size + 35, offset_y + i * tile_size + 25))
        for i in range(4):
            pygame.draw.line(screen, BLACK, (offset_x, offset_y + i * tile_size), (offset_x + 300, offset_y + i * tile_size), 2)
            pygame.draw.line(screen, BLACK, (offset_x + i * tile_size, offset_y), (offset_x + i * tile_size, offset_y + 300), 2)
        pygame.display.flip()

    def print_solution(self, solution):
        if solution is None or len(solution) == 0:
            print("Không tìm thấy đường đi đến trạng thái mục tiêu!")
            return
        for step, state in enumerate(solution):
            print(f"Bước {step}:")
            for row in state:
                print(row)
            print()
        print("Hoàn thành!")

    def run(self):
        pygame.init()
        screen = pygame.display.set_mode((600, 400))
        pygame.display.set_caption("8-Puzzle Solver (Program B)")
        clock = pygame.time.Clock()

        initial_state = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        goal_state = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
        puzzle = self.EightPuzzle(initial_state, goal_state)

        buttons = [
            ("Backtracking", (20, 20)),
            ("Forward Checking", (20, 70)),
            ("Min-Conflicts", (20, 120)),
            ("Reset", (20, 170)),
        ]
        font = pygame.font.Font(None, 36)
        running = True
        solution = None
        step = 0
        elapsed_time = None
        algorithm_run = False

        while running:
            screen.fill((255, 255, 255))
            
            for text, pos in buttons:
                pygame.draw.rect(screen, (139, 0, 0), (*pos, 200, 40))
                button_width, button_height = 200, 40
                label = font.render(text, True, (0, 0, 0))
                text_width, text_height = label.get_size()
                text_x = pos[0] + (button_width - text_width) // 2
                text_y = pos[1] + (button_height - text_height) // 2
                screen.blit(label, (text_x, text_y))
            
            if solution and algorithm_run and step < len(solution):
                self.draw_grid(screen, solution[step], 100, 250, 50)
                if step < len(solution) - 1:
                    step += 1
                    time.sleep(0.1)
            else:
                self.draw_grid(screen, puzzle.initial, 100, 250, 50)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.MOUSEBUTTONDOWN and not algorithm_run:
                    x, y = event.pos
                    start_time = time.time()
                    
                    if 20 <= x <= 220 and 20 <= y <= 60:  # Backtracking
                        solution = puzzle.backtracking_search()
                        step = 0
                        end_time = time.time()
                        elapsed_time = end_time - start_time
                        print("\nKết quả của thuật toán Backtracking:")
                        self.print_solution(solution)
                        print(f"Thời gian thực thi: {elapsed_time:.10f} giây")
                        algorithm_run = True
                    elif 20 <= x <= 220 and 70 <= y <= 110:  # Forward Checking
                        solution = puzzle.forward_checking_search()
                        step = 0
                        end_time = time.time()
                        elapsed_time = end_time - start_time
                        print("\nKết quả của thuật toán Forward Checking:")
                        self.print_solution(solution)
                        print(f"Thời gian thực thi: {elapsed_time:.10f} giây")
                        algorithm_run = True
                    elif 20 <= x <= 220 and 120 <= y <= 160:  # Min-Conflicts
                        solution = puzzle.min_conflicts_search()
                        step = 0
                        end_time = time.time()
                        elapsed_time = end_time - start_time
                        print("\nKết quả của thuật toán Min-Conflicts:")
                        self.print_solution(solution)
                        print(f"Thời gian thực thi: {elapsed_time:.10f} giây")
                        algorithm_run = True
                    elif 20 <= x <= 220 and 170 <= y <= 210: 
                        solution = [initial_state]  
                        step = 0
                        elapsed_time = None
                        algorithm_run = False
                        puzzle = self.EightPuzzle(initial_state, goal_state)  
                        self.draw_grid(screen, initial_state, 100, 250, 50)  
                        pygame.display.flip()

            pygame.display.flip()
            clock.tick(60)

        pygame.quit()

class Belief:
    class EightPuzzle:
        def __init__(self, initial, goals):
            self.initial = tuple(map(tuple, initial))
            self.goals = [tuple(map(tuple, goal)) for goal in goals]
            self.goal_state_indices = []
            for goal in self.goals:
                self.goal_state_indices.append({goal[i][j]: (i, j) for i in range(3) for j in range(3)})

        def find_blank(self, state):
            for i, row in enumerate(state):
                if 0 in row:
                    return i, row.index(0)

        def get_neighbors(self, state):
            row, col = self.find_blank(state)
            moves = []
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            state_list = [list(row) for row in state]

            for dr, dc in directions:
                new_row, new_col = row + dr, col + dc
                if 0 <= new_row < 3 and 0 <= new_col < 3:
                    new_state = [row[:] for row in state_list]
                    new_state[row][col], new_state[new_row][new_col] = new_state[new_row][new_col], new_state[row][col]
                    moves.append(tuple(map(tuple, new_state)))
            return moves

        def manhattan_distance(self, state, goal_idx):
            distance = 0
            for i in range(3):
                for j in range(3):
                    if state[i][j] != 0:
                        goal_x, goal_y = self.goal_state_indices[goal_idx][state[i][j]]
                        distance += abs(i - goal_x) + abs(j - goal_y)
            return distance

        def get_observation(self, state):
            for i in range(3):
                for j in range(3):
                    if state[i][j] == 1:
                        return (i, j)
            return None

        def generate_fixed_belief_states(self):
            return [
                tuple(map(tuple, [[1, 2, 3], [4, 5, 6], [7, 0, 8]])),
                tuple(map(tuple, [[1, 2, 3], [4, 0, 6], [7, 5, 8]])),
                tuple(map(tuple, [[1, 2, 3], [0, 5, 6], [4, 7, 8]])),
                tuple(map(tuple, [[1, 2, 0], [4, 5, 3], [7, 8, 6]])),
            ]

        def belief_state_search(self, max_states=4):
            initial_belief = set(self.generate_fixed_belief_states())
            pq = []
            heapq.heappush(pq, (0, 0, initial_belief, [list(initial_belief)[0]], [initial_belief.copy()]))
            visited = {}

            while pq:
                time.sleep(0.1)  
                f, g, belief, path, belief_history = heapq.heappop(pq)
                belief_key = frozenset(belief)
                if belief_key in visited and visited[belief_key] <= f:
                    continue
                visited[belief_key] = f

                for state in belief:
                    for goal in self.goals:
                        if state == goal:
                            return path, list(belief), belief_history

                new_belief = set()
                for state in belief:
                    new_belief.update(self.get_neighbors(state))

                if len(new_belief) > max_states:
                    new_belief = set(sorted(new_belief, key=lambda s: min(self.manhattan_distance(s, i) for i in range(len(self.goals))) + g)[:max_states])

                if not new_belief:
                    continue

                new_g = g + 1
                min_h = min(min(self.manhattan_distance(s, i) for i in range(len(self.goals))) for s in new_belief)
                new_f = new_g + min_h

                best_state = min(new_belief, key=lambda s: min(self.manhattan_distance(s, i) for i in range(len(self.goals))))
                heapq.heappush(pq, (new_f, new_g, new_belief, path + [best_state], belief_history + [new_belief.copy()]))

            return None, [], []

        def partial_observable_search(self, max_states=4, max_steps=1000):
            initial_belief = set(self.generate_fixed_belief_states())
            pq = []
            representative_state = min(initial_belief, key=lambda s: min(self.manhattan_distance(s, i) for i in range(len(self.goals))))
            initial_h = min(self.manhattan_distance(representative_state, i) for i in range(len(self.goals)))
            heapq.heappush(pq, (initial_h, 0, initial_belief, [representative_state], [initial_belief.copy()]))
            visited = {}
            steps = 0

            while pq and steps < max_steps:
                time.sleep(0.1)  # Add 10ms delay per iteration
                f, g, belief, path, belief_history = heapq.heappop(pq)
                belief_key = frozenset(belief)
                if belief_key in visited and visited[belief_key] <= f:
                    continue
                visited[belief_key] = f
                steps += 1

                for state in belief:
                    for goal in self.goals:
                        if state == goal:
                            return path, list(belief), belief_history

                observation = self.get_observation(representative_state)
                if observation is None:
                    continue

                filtered_belief = set()
                for state in belief:
                    if self.get_observation(state) == observation:
                        filtered_belief.add(state)
                if not filtered_belief:
                    filtered_belief = belief

                new_belief = set()
                for state in filtered_belief:
                    new_belief.update(self.get_neighbors(state))

                if len(new_belief) > max_states:
                    new_belief = set(sorted(new_belief, key=lambda s: min(self.manhattan_distance(s, i) for i in range(len(self.goals))) + g)[:max_states])

                if not new_belief:
                    continue

                new_g = g + 1
                representative_state = min(new_belief, key=lambda s: min(self.manhattan_distance(s, i) for i in range(len(self.goals))))
                min_h = min(self.manhattan_distance(representative_state, i) for i in range(len(self.goals)))
                new_f = new_g + min_h

                if random.random() < 0.1:
                    representative_state = random.choice(list(new_belief))

                heapq.heappush(pq, (new_f, new_g, new_belief, path + [representative_state], belief_history + [new_belief.copy()]))

            return None, [], []

    def draw_grid(self, screen, state, tile_size, offset_x, offset_y):
        WHITE, BLACK = (255, 255, 255), (0, 0, 0)
        pygame.draw.rect(screen, WHITE, (offset_x, offset_y, 240, 240))
        font = pygame.font.Font(None, 50)
        for i in range(3):
            for j in range(3):
                if state[i][j] != 0:
                    pygame.draw.rect(screen, (0, 100, 0), (offset_x + j * tile_size, offset_y + i * tile_size, tile_size, tile_size))
                    text = font.render(str(state[i][j]), True, BLACK)
                    screen.blit(text, (offset_x + j * tile_size + 35, offset_y + i * tile_size + 25))
        for i in range(4):
            pygame.draw.line(screen, BLACK, (offset_x, offset_y + i * tile_size), (offset_x + 240, offset_y + i * tile_size), 2)
            pygame.draw.line(screen, BLACK, (offset_x + i * tile_size, offset_y), (offset_x + i * tile_size, offset_y + 240), 2)

    def print_solution(self, solution, belief_states, elapsed_time, belief_history=None):
        if solution is None:
            print("Không tìm thấy đường đi đến trạng thái mục tiêu!")
            return
        print("Đường đi:")
        for step, state in enumerate(solution):
            print(f"Bước {step}:")
            for row in state:
                print(row)
            print()
        print("Các trạng thái niềm tin cuối cùng:")
        for i, state in enumerate(belief_states):
            print(f"Trạng thái niềm tin {i+1}:")
            for row in state:
                print(row)
            print()
        if belief_history:
            print(f"Số bước trong lịch sử niềm tin: {len(belief_history)}")
        print(f"Thời gian thực thi: {elapsed_time:.8f}s")
        print("Hoàn thành!")

    async def run(self):
        pygame.init()
        screen = pygame.display.set_mode((1200, 700))
        pygame.display.set_caption("8-Puzzle Solver - Belief State & Partial Observable Search")
        clock = pygame.time.Clock()

        initial_state = [[1, 2, 3], [5, 0, 6], [7, 4, 8]]
        goal_states = [
            [[1, 2, 3], [4, 5, 6], [7, 8, 0]],
            [[1, 2, 3], [4, 5, 6], [0, 7, 8]],
            [[1, 2, 3], [0, 4, 6], [7, 5, 8]]
        ]
        puzzle = self.EightPuzzle(initial_state, goal_states)

        font = pygame.font.Font(None, 36)
        belief_button_rect = pygame.Rect(20, 50, 200, 40)
        partial_button_rect = pygame.Rect(230, 50, 250, 40)

        belief_states = puzzle.generate_fixed_belief_states()
        belief_history = []
        solution = []
        step = 0
        elapsed_time = None
        running = True
        animating = False
        last_step_time = time.time()

        while running:
            screen.fill((255, 255, 255))

            pygame.draw.rect(screen, (139, 0, 0), belief_button_rect)
            label = font.render("Belief State", True, (0, 0, 0))
            text_width, text_height = label.get_size()
            text_x = belief_button_rect.x + (belief_button_rect.width - text_width) // 2
            text_y = belief_button_rect.y + (belief_button_rect.height - text_height) // 2
            screen.blit(label, (text_x, text_y))

            pygame.draw.rect(screen, (0, 0, 139), partial_button_rect)
            label = font.render("Partial Observable", True, (0, 0, 0))
            text_width, text_height = label.get_size()
            text_x = partial_button_rect.x + (partial_button_rect.width - text_width) // 2
            text_y = partial_button_rect.y + (partial_button_rect.height - text_height) // 2
            screen.blit(label, (text_x, text_y))

            if belief_history and (animating or step > 0):
                current_beliefs = list(belief_history[min(step, len(belief_history) - 1)])
            else:
                current_beliefs = belief_states

            for i in range(min(4, len(current_beliefs))):
                belief_label = font.render(f"Belief {i+1}", True, (0, 0, 0))
                screen.blit(belief_label, (20 + i * 300, 100))
                self.draw_grid(screen, current_beliefs[i], 80, 20 + i * 300, 140)

            for i in range(3):
                goal_label = font.render(f"Goal {i+1}", True, (0, 0, 0))
                screen.blit(goal_label, (20 + i * 300, 400))
                self.draw_grid(screen, puzzle.goals[i], 80, 20 + i * 300, 440)

            if animating and time.time() - last_step_time > 1.0:
                step += 1
                last_step_time = time.time()
                if step >= len(belief_history):
                    animating = False
                    step = len(belief_history) - 1

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if belief_button_rect.collidepoint(event.pos):
                        start_time = time.time()
                        solution, belief_states, belief_history = puzzle.belief_state_search(max_states=4)
                        elapsed_time = time.time() - start_time
                        step = 0
                        last_step_time = time.time()
                        animating = True
                        print("\nKết quả của thuật toán Belief State Search:")
                        self.print_solution(solution, belief_states, elapsed_time, belief_history)
                        await asyncio.sleep(0.5)
                    elif partial_button_rect.collidepoint(event.pos):
                        start_time = time.time()
                        solution, belief_states, belief_history = puzzle.partial_observable_search(max_states=4)
                        elapsed_time = time.time() - start_time
                        step = 0
                        last_step_time = time.time()
                        animating = True
                        print("\nKết quả của thuật toán Partial Observable Search:")
                        self.print_solution(solution, belief_states, elapsed_time, belief_history)
                        await asyncio.sleep(0.5)

            pygame.display.flip()
            clock.tick(60)
            await asyncio.sleep(1.0 / 60)

        pygame.quit()

async def main():
    pygame.init()
    screen = pygame.display.set_mode((400, 400)) 
    pygame.display.set_caption("8-Puzzle Solver Selector")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 36)

    buttons = [
        ("Run Base", (100, 50)),
        ("Run CSPS", (100, 150)),
        ("Run Belief", (100, 250)),  
    ]

    running = True
    while running:
        screen.fill((255, 255, 255))

        for text, pos in buttons:
            pygame.draw.rect(screen, (139, 0, 0), (*pos, 200, 40))
            button_width, button_height = 200, 40
            label = font.render(text, True, (0, 0, 0))
            text_width, text_height = label.get_size()
            text_x = pos[0] + (button_width - text_width) // 2
            text_y = pos[1] + (button_height - text_height) // 2
            screen.blit(label, (text_x, text_y))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                if 100 <= x <= 300 and 50 <= y <= 90:  
                    program_a = Base()
                    await program_a.run()
                elif 100 <= x <= 300 and 150 <= y <= 190:  
                    program_b = CSPs()
                    program_b.run()
                elif 100 <= x <= 300 and 250 <= y <= 290:  
                    program_c = Belief()
                    await program_c.run()

        pygame.display.flip()
        clock.tick(60)
        await asyncio.sleep(1.0 / 60)

    pygame.quit()

if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    if __name__ == "__main__":
        asyncio.run(main())
