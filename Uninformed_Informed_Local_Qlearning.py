import pygame
import heapq
import time
import random
import math
from collections import deque
import platform
import asyncio
import random
import numpy as np

class EightPuzzle:
    def __init__(self, initial, goal):
        self.initial = tuple(map(tuple, initial))
        self.goal = tuple(map(tuple, goal))
        self.goal_state_index = {self.goal[i][j]: (i, j) for i in range(3) for j in range(3)}

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
    
    def manhattan_distance(self, state):
        distance = 0
        for i in range(3):
            for j in range(3):
                if state[i][j] != 0:
                    goal_x, goal_y = self.goal_state_index[state[i][j]]
                    distance += abs(i - goal_x) + abs(j - goal_y)
        return distance

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

        # Khởi tạo bảng Q
        Q = {}
        
        def state_to_tuple(state):
            return tuple(tuple(row) for row in state)

        def get_valid_actions(self, state):
            row, col = self.find_blank(state)
            valid_actions = []
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # Lên, xuống, trái, phải
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

        # Huấn luyện Q-Learning
        for episode in range(episodes):
            current_state = self.initial
            while current_state != self.goal:
                current_state_tuple = state_to_tuple(current_state)
                if current_state_tuple not in Q:
                    Q[current_state_tuple] = {action: 0 for action in get_valid_actions(self, current_state)}

                # Chọn hành động theo chính sách epsilon-greedy
                valid_actions = get_valid_actions(self, current_state)
                if not valid_actions:
                    break
                if random.random() < epsilon:
                    action = random.choice(valid_actions)
                else:
                    action_scores = {a: Q[current_state_tuple][a] for a in valid_actions}
                    action = max(action_scores, key=action_scores.get)

                # Thực hiện hành động
                next_state = perform_action(self, current_state, action)
                next_state_tuple = state_to_tuple(next_state)
                if next_state_tuple not in Q:
                    Q[next_state_tuple] = {action: 0 for action in get_valid_actions(self, next_state)}

                # Tính phần thưởng
                reward = get_reward(self, next_state)

                # Cập nhật Q-value
                best_next_action = max(Q[next_state_tuple].values()) if Q[next_state_tuple] else 0
                Q[current_state_tuple][action] += alpha * (
                    reward + gamma * best_next_action - Q[current_state_tuple][action]
                )

                current_state = next_state

        # Truy xuất đường đi từ trạng thái ban đầu
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
    

def draw_grid(screen, state, tile_size, offset_x, offset_y):
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

def print_solution(solution):
    if solution is None:
        print("Không tìm thấy đường đi đến trạng thái mục tiêu!")
        return
    for step, state in enumerate(solution):
        print(f"Bước {step}:")
        for row in state:
            print(row)
        print()
    print("Hoàn thành!")

def parse_input(text):
    try:
        numbers = [int(x) for x in text.replace(',', ' ').split() if x.strip().isdigit()]
        if len(numbers) != 9 or set(numbers) != set(range(9)):
            return None
        return [[numbers[i * 3 + j] for j in range(3)] for i in range(3)]
    except:
        return None

async def run_pygame():
    pygame.init()
    screen = pygame.display.set_mode((800, 700))
    pygame.display.set_caption("8-Puzzle Solver")
    clock = pygame.time.Clock()
    
    initial_state = [[1, 2, 3], [4, 0, 6], [7, 5, 8]]
    # initial_state = [[1, 2, 3], [5, 0, 6], [7, 4, 8]]
    # initial_state = [
    # [1, 2, 0],
    # [5, 6, 3],
    # [4, 7, 8]
    # ]
    # initial_state = [
    #     [8, 6, 7],
    #     [2, 5, 4],
    #     [3, 0, 1]
    # ]
    goal_state = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
    puzzle = EightPuzzle(initial_state, goal_state)
    
    buttons = [
    ("BFS", (20, 160)),               # Cột 1, hàng 1
    ("DFS", (20, 210)),               # Cột 1, hàng 2
    ("UCS", (20, 260)),               # Cột 1, hàng 3
    ("IDS", (20, 310)),               # Cột 1, hàng 4
    ("Greedy Search", (20, 360)),     # Cột 1, hàng 5
    ("A*", (20, 410)),                # Cột 1, hàng 6
    ("AND-OR Search", (20, 460)),     # Cột 1, hàng 7
    ("Q-Learning", (20, 510)),        # Cột 1, hàng 8
    ("IDA*", (230, 160)),             # Cột 2, hàng 1
    ("Simple HC", (230, 210)),        # Cột 2, hàng 2
    ("Steepest HC", (230, 260)),      # Cột 2, hàng 3
    ("Stochastic HC", (230, 310)),    # Cột 2, hàng 4
    ("S A", (230, 360)),              # Cột 2, hàng 5
    ("Beam Search", (230, 410)),      # Cột 2, hàng 6
    ("GA", (230, 460)),               # Cột 2, hàng 7
    ("Reset", (230, 510)),            # Cột 2, hàng 8
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
        
        # Vẽ ô nhập liệu
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
        
        # Hiển thị thông báo lỗi
        if error_message:
            error_text = input_font.render(error_message, True, (255, 0, 0))
            screen.blit(error_text, (20, 150))
        
        # Vẽ các nút bấm
        for text, pos in buttons:
            rect = pygame.draw.rect(screen, (139, 0, 0), (*pos, 200, 40))
            button_width, button_height = 200, 40
            label = font.render(text, True, (0, 0, 0))
            text_width, text_height = label.get_size()
            text_x = pos[0] + (button_width - text_width) // 2
            text_y = pos[1] + (button_height - text_height) // 2
            screen.blit(label, (text_x, text_y))
        
        # Vẽ bảng trạng thái ban đầu
        solution_label = font.render("Initial_state", True, (0, 0, 0))
        screen.blit(solution_label, (500, 10))
        if solution:
            draw_grid(screen, solution[step], 80, 500, 50)
            if step < len(solution) - 1:
                step += 1
                await asyncio.sleep(0.01)
        else:
            draw_grid(screen, puzzle.initial, 80, 500, 50)

        # Vẽ bảng trạng thái đích
        goal_label = font.render("Goal_state", True, (0, 0, 0))
        screen.blit(goal_label, (500, 310))
        draw_grid(screen, puzzle.goal, 80, 500, 350)
            
        if elapsed_time is not None:
            time_text = font.render(f"Thoi gian thuc thi: {elapsed_time:.4f}s", True, (0, 0, 0))
            screen.blit(time_text, (450, 600))
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                # Xử lý nhập liệu trạng thái ban đầu và mục tiêu
                if initial_rect.collidepoint(x, y):
                    active_input = "initial"
                elif goal_rect.collidepoint(x, y):
                    active_input = "goal"
                else:
                    active_input = None
                
                start_time = time.time()
                # Xử lý các nút thuật toán
                if 20 <= x <= 220 and 160 <= y <= 200:  # BFS
                    solution = puzzle.bfs()
                    step = 0
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    print("\nKết quả của thuật toán BFS:")
                    print(f"Thời gian thực thi: {elapsed_time:.10f} giây")
                    print_solution(solution)
                    algorithm_run = True
                elif 20 <= x <= 220 and 210 <= y <= 250:  # DFS
                    solution = puzzle.dfs()
                    step = 0
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    print("\nKết quả của thuật toán DFS:")
                    print(f"Thời gian thực thi: {elapsed_time:.10f} giây")
                    print_solution(solution)
                    algorithm_run = True
                elif 20 <= x <= 220 and 260 <= y <= 300:  # UCS
                    solution = puzzle.ucs()
                    step = 0
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    print("\nKết quả của thuật toán UCS:")
                    print(f"Thời gian thực thi: {elapsed_time:.10f} giây")
                    print_solution(solution)
                    algorithm_run = True
                elif 20 <= x <= 220 and 310 <= y <= 350:  # IDS
                    solution = puzzle.iterative_deepening_search()
                    step = 0
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    print("\nKết quả của thuật toán Iterative Deepening Search:")
                    print(f"Thời gian thực thi: {elapsed_time:.10f} giây")
                    print_solution(solution)
                    algorithm_run = True
                elif 20 <= x <= 220 and 360 <= y <= 400:  # Greedy Search
                    solution = puzzle.greedy_search()
                    step = 0
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    print("\nKết quả của thuật toán Greedy Search:")
                    print(f"Thời gian thực thi: {elapsed_time:.10f} giây")
                    print_solution(solution)
                    algorithm_run = True
                elif 20 <= x <= 220 and 410 <= y <= 450:  # A*
                    solution = puzzle.a_star_search()
                    step = 0
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    print("\nKết quả của thuật toán A*:")
                    print(f"Thời gian thực thi: {elapsed_time:.10f} giây")
                    print_solution(solution)
                    algorithm_run = True
                elif 20 <= x <= 220 and 460 <= y <= 500:  # AND-OR Search
                    solution = puzzle.and_or_tree_search()
                    step = 0
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    print("\nKết quả của thuật toán AND-OR Tree Search:")
                    print(f"Thời gian thực thi: {elapsed_time:.10f} giây")
                    print_solution(solution)
                    algorithm_run = True
                elif 20 <= x <= 220 and 510 <= y <= 550:  # Q-Learning
                    solution = puzzle.q_learning()
                    step = 0
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    print("\nKết quả của thuật toán Q-Learning:")
                    print(f"Thời gian thực thi: {elapsed_time:.10f} giây")
                    print_solution(solution)
                    algorithm_run = True
                elif 230 <= x <= 430 and 160 <= y <= 200:  # IDA*
                    solution = puzzle.ida_star()
                    step = 0
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    print("\nKết quả của thuật toán IDA*:")
                    print(f"Thời gian thực thi: {elapsed_time:.10f} giây")
                    print_solution(solution)
                    algorithm_run = True
                elif 230 <= x <= 430 and 210 <= y <= 250:  # Simple HC
                    solution = puzzle.simple_hill_climbing()
                    step = 0
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    print("\nKết quả của thuật toán Simple Hill Climbing:")
                    print(f"Thời gian thực thi: {elapsed_time:.10f} giây")
                    print_solution(solution)
                    algorithm_run = True
                elif 230 <= x <= 430 and 260 <= y <= 300:  # Steepest HC
                    solution = puzzle.steepest_ascent_hill_climbing()
                    step = 0
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    print("\nKết quả của thuật toán Steepest Ascent Hill Climbing:")
                    print(f"Thời gian thực thi: {elapsed_time:.10f} giây")
                    print_solution(solution)
                    algorithm_run = True
                elif 230 <= x <= 430 and 310 <= y <= 350:  # Stochastic HC
                    solution = puzzle.stochastic_hill_climbing()
                    step = 0
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    print("\nKết quả của thuật toán Stochastic Hill Climbing:")
                    print(f"Thời gian thực thi: {elapsed_time:.10f} giây")
                    print_solution(solution)
                    algorithm_run = True
                elif 230 <= x <= 430 and 360 <= y <= 400:  # Simulated Annealing
                    solution = puzzle.simulated_annealing()
                    step = 0
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    print("\nKết quả của thuật toán Simulated Annealing:")
                    print(f"Thời gian thực thi: {elapsed_time:.10f} giây")
                    print_solution(solution)
                    algorithm_run = True
                elif 230 <= x <= 430 and 410 <= y <= 450:  # Beam Search
                    solution = puzzle.beam_search()
                    step = 0
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    print("\nKết quả của thuật toán Beam Search:")
                    print(f"Thời gian thực thi: {elapsed_time:.10f} giây")
                    print_solution(solution)
                    algorithm_run = True
                elif 230 <= x <= 430 and 460 <= y <= 500:  # GA
                    solution = puzzle.genetic_algorithm()
                    step = 0
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    print("\nKết quả của thuật toán Genetic Algorithm:")
                    print(f"Thời gian thực thi: {elapsed_time:.10f} giây")
                    print_solution(solution)
                    algorithm_run = True
                elif 230 <= x <= 430 and 510 <= y <= 550:  # Reset
                    solution = None
                    step = 0
                    elapsed_time = None
                    algorithm_run = False
                    continue

            elif event.type == pygame.KEYDOWN and active_input:
                if event.key == pygame.K_BACKSPACE:
                    if active_input == "initial":
                        initial_input = initial_input[:-1]
                    elif active_input == "goal":
                        goal_input = goal_input[:-1]
                elif event.key == pygame.K_RETURN:
                    new_initial = parse_input(initial_input)
                    new_goal = parse_input(goal_input)
                    if new_initial and new_goal:
                        initial_state = new_initial
                        goal_state = new_goal
                        puzzle = EightPuzzle(initial_state, goal_state)
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

if platform.system() == "Emscripten":
    asyncio.ensure_future(run_pygame())
else:
    if __name__ == "__main__":
        asyncio.run(run_pygame())