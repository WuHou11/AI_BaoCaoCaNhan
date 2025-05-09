import pygame
import heapq
import time
import platform
import asyncio
import random

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
            tuple(map(tuple, [[1, 2, 3], [4, 5, 6], [7, 0, 8]])),  # State 1
            tuple(map(tuple, [[1, 2, 3], [4, 0, 6], [7, 5, 8]])),  # State 2
            tuple(map(tuple, [[1, 2, 3], [0, 5, 6], [4, 7, 8]])),  # State 3
            tuple(map(tuple, [[1, 2, 0], [4, 5, 3], [7, 8, 6]])),  # State 4
        ]

    def belief_state_search(self, max_states=4):
        initial_belief = set(self.generate_fixed_belief_states())
        pq = []
        heapq.heappush(pq, (0, 0, initial_belief, [list(initial_belief)[0]], [initial_belief.copy()]))
        visited = {}

        while pq:
            time.sleep(0.1)  # Thêm độ trễ 10ms mỗi lần lặp
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
            time.sleep(0.1)  # Thêm độ trễ 10ms mỗi lần lặp
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

def print_solution(solution, belief_states, elapsed_time, belief_history=None):
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

async def run_pygame():
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
    puzzle = EightPuzzle(initial_state, goal_states)

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
            draw_grid(screen, current_beliefs[i], 80, 20 + i * 300, 140)

        for i in range(3):
            goal_label = font.render(f"Goal {i+1}", True, (0, 0, 0))
            screen.blit(goal_label, (20 + i * 300, 400))
            draw_grid(screen, puzzle.goals[i], 80, 20 + i * 300, 440)

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
                    print_solution(solution, belief_states, elapsed_time, belief_history)
                    await asyncio.sleep(0.5)
                elif partial_button_rect.collidepoint(event.pos):
                    start_time = time.time()
                    solution, belief_states, belief_history = puzzle.partial_observable_search(max_states=4)
                    elapsed_time = time.time() - start_time
                    step = 0
                    last_step_time = time.time()
                    animating = True
                    print("\nKết quả của thuật toán Partial Observable Search:")
                    print_solution(solution, belief_states, elapsed_time, belief_history)
                    await asyncio.sleep(0.5)

        pygame.display.flip()
        clock.tick(60)
        await asyncio.sleep(0)

    pygame.quit()

if platform.system() == "Emscripten":
    asyncio.ensure_future(run_pygame())
else:
    if __name__ == "__main__":
        asyncio.run(run_pygame())