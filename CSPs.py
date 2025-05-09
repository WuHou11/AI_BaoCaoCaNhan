import pygame
import time
import random
from collections import defaultdict

class EightPuzzle:
    def __init__(self, initial, goal):
        # Khởi tạo trạng thái ban đầu và trạng thái mục tiêu dưới dạng tuple
        self.initial = tuple(map(tuple, initial))
        self.goal = tuple(map(tuple, goal))

    def is_valid_assignment(self, state, pos, value):
        # Kiểm tra xem việc gán giá trị cho vị trí pos có hợp lệ không
        i, j = pos
        # Đảm bảo mỗi số chỉ xuất hiện một lần
        for r in range(3):
            for c in range(3):
                if (r, c) != pos and state[r][c] == value:
                    return False
        # Kiểm tra ràng buộc hàng: nếu value không phải 0, kiểm tra các ô lân cận
        if j > 0 and state[i][j - 1] != 0 and value != 0 and state[i][j - 1] != value - 1:
            return False
        if j < 2 and value != 0 and state[i][j + 1] != 0 and state[i][j + 1] != value + 1:
            return False
        # Kiểm tra ràng buộc cột: nếu value không phải 0, kiểm tra các ô lân cận
        if i > 0 and state[i - 1][j] != 0 and value != 0 and state[i - 1][j] != value - 3:
            return False
        if i < 2 and value != 0 and state[i + 1][j] != 0 and state[i + 1][j] != value + 3:
            return False
        return True

    def is_solvable(self, state):
        # Kiểm tra xem trạng thái có thể giải được hay không
        flat = [state[i][j] for i in range(3) for j in range(3) if state[i][j] != 0]
        inversions = 0
        for i in range(len(flat)):
            for j in range(i + 1, len(flat)):
                if flat[i] > flat[j]:
                    inversions += 1
        return inversions % 2 == 0

    def count_conflicts(self, state):
        # Đếm số lượng xung đột trong trạng thái
        conflicts = 0
        value_counts = defaultdict(int)

        # Kiểm tra số lần xuất hiện của mỗi giá trị
        for i in range(3):
            for j in range(3):
                val = state[i][j]
                if val != 0:
                    value_counts[val] += 1
                    if value_counts[val] > 1:
                        conflicts += 2 * (value_counts[val] - 1)

        # Kiểm tra xung đột trên hàng
        for i in range(3):
            for j in range(2):
                if state[i][j] != 0 and state[i][j + 1] != 0:
                    if state[i][j + 1] != state[i][j] + 1:
                        conflicts += 1

        # Kiểm tra xung đột trên cột
        for j in range(3):
            for i in range(2):
                if state[i][j] != 0 and state[i + 1][j] != 0:
                    if state[i + 1][j] != state[i][j] + 3:
                        conflicts += 1

        # Thêm xung đột nếu trạng thái không thể giải được
        if not self.is_solvable(state):
            conflicts += 10

        return conflicts

    def backtracking_search(self):
        # Tìm kiếm giải pháp bằng thuật toán quay lui
        visited = set()
        path = []

        def backtrack(state, assigned, pos_index):
            # Nếu đã gán hết 9 ô, kiểm tra xem có phải trạng thái mục tiêu không
            if pos_index == 9:
                state_tuple = tuple(tuple(row) for row in state)
                if state_tuple == self.goal and self.is_solvable(state):
                    path.append(state_tuple)
                    return path
                return None

            i, j = divmod(pos_index, 3)
            state_tuple = tuple(tuple(row) for row in state)
            # Nếu trạng thái đã được thăm, bỏ qua
            if state_tuple in visited:
                return None
            visited.add(state_tuple)
            path.append(state_tuple)

            # Thử gán các giá trị từ 0 đến 8
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

        # Bắt đầu với trạng thái rỗng
        empty_state = [[0 for _ in range(3)] for _ in range(3)]
        result = backtrack(empty_state, set(), 0)
        return result

    def forward_checking_search(self):
        # Tìm kiếm giải pháp bằng thuật toán kiểm tra tiến
        visited = set()
        path = []

        def get_domain(state, pos, assigned):
            # Lấy danh sách giá trị khả thi cho vị trí pos
            i, j = pos
            domain = []
            for value in [0, 1, 2, 3, 4, 5, 6, 7, 8]:
                if value not in assigned and self.is_valid_assignment(state, (i, j), value):
                    domain.append(value)
            return domain

        def forward_check(state, pos, value, domains, assigned):
            # Cập nhật miền giá trị sau khi gán giá trị
            i, j = pos
            new_domains = {k: v[:] for k, v in domains.items()}
            success = True
            # Cập nhật miền cho các vị trí chưa gán
            for r in range(3):
                for c in range(3):
                    if (r, c) not in assigned and (r, c) != pos:
                        if (r, c) in new_domains:
                            new_domain = new_domains[(r, c)][:]
                            # Loại giá trị đã gán nếu có xung đột
                            if value != 0 and value in new_domain:
                                # Kiểm tra ràng buộc hàng
                                if r == i:
                                    if c < j and value - 1 in new_domain:
                                        new_domain = [v for v in new_domain if v == 0 or v == value - 1]
                                    elif c > j and value + 1 in new_domain:
                                        new_domain = [v for v in new_domain if v == 0 or v == value + 1]
                                # Kiểm tra ràng buộc cột
                                if c == j:
                                    if r < i and value - 3 in new_domain:
                                        new_domain = [v for v in new_domain if v == 0 or v == value - 3]
                                    elif r > i and value + 3 in new_domain:
                                        new_domain = [v for v in new_domain if v == 0 or v == value + 3]
                                # Loại giá trị đã gán
                                if value in new_domain:
                                    new_domain.remove(value)
                            new_domains[(r, c)] = new_domain
                            if not new_domain:
                                success = False
            return success, new_domains

        def select_mrv_variable(positions, domains):
            # Chọn biến có ít giá trị khả thi nhất
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
            # Chọn giá trị ít gây ràng buộc nhất
            value_scores = []
            for value in domain:
                temp_state = [row[:] for row in state]
                temp_state[pos[0]][pos[1]] = value
                _, new_domains = forward_check(temp_state, pos, value, domains, assigned)
                # Đếm số giá trị còn lại trong các miền khác
                remaining = sum(len(new_domains[p]) for p in new_domains if p != pos)
                # Ưu tiên giá trị khớp với trạng thái mục tiêu
                if value == self.goal[pos[0]][pos[1]]:
                    remaining += 10
                value_scores.append((-remaining, value))  # Sắp xếp theo số giá trị còn lại
            value_scores.sort()
            return [value for _, value in value_scores]

        def backtrack_with_fc(state, assigned, positions, domains):
            # Quay lui với kiểm tra tiến
            if len(assigned) == 9:
                state_tuple = tuple(tuple(row) for row in state)
                if state_tuple == self.goal and self.is_solvable(state):
                    path.append(state_tuple)
                    return path
                return None

            # Chọn biến có ít giá trị khả thi nhất
            pos = select_mrv_variable(positions, domains)
            if pos is None:
                return None

            # Lấy danh sách giá trị khả thi
            domain = get_domain(state, pos, set(assigned.values()))
            if not domain:
                return None

            # Sắp xếp giá trị theo mức độ ràng buộc
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

        # Khởi tạo trạng thái rỗng và miền giá trị
        empty_state = [[0 for _ in range(3)] for _ in range(3)]
        positions = [(i, j) for i in range(3) for j in range(3)]
        domains = {(i, j): [0, 1, 2, 3, 4, 5, 6, 7, 8] for i in range(3) for j in range(3)}
        assigned = {}
        result = backtrack_with_fc(empty_state, assigned, positions, domains)
        return result

    def min_conflicts_search(self, max_steps=1000):
        # Tìm kiếm giải pháp bằng thuật toán tối thiểu xung đột

        def find_blank(state):
            # Tìm vị trí của ô trống (0) trong trạng thái
            for i in range(3):
                for j in range(3):
                    if state[i][j] == 0:
                        return i, j
            return None

        def get_neighbors(i, j):
            # Lấy các vị trí lân cận hợp lệ cho ô trống
            neighbors = []
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Lên, Xuống, Trái, Phải
            for di, dj in directions:
                ni, nj = i + di, j + dj
                if 0 <= ni < 3 and 0 <= nj < 3:
                    neighbors.append((ni, nj))
            return neighbors

        def count_conflicts(state):
            # Tính số xung đột dựa trên khoảng cách Manhattan
            conflicts = 0
            goal_positions = {self.goal[i][j]: (i, j) for i in range(3) for j in range(3) if self.goal[i][j] != 0}
            for i in range(3):
                for j in range(3):
                    if state[i][j] != 0:
                        goal_i, goal_j = goal_positions.get(state[i][j], (i, j))
                        conflicts += abs(i - goal_i) + abs(j - goal_j)
            return conflicts

        def is_valid_state(state):
            # Kiểm tra tính hợp lệ của trạng thái
            flat = [state[i][j] for i in range(3) for j in range(3)]
            return sorted(flat) == [0, 1, 2, 3, 4, 5, 6, 7, 8]

        # Khởi tạo trạng thái: Sử dụng trạng thái ban đầu nếu hợp lệ, nếu không tạo trạng thái ngẫu nhiên có thể giải được
        if is_valid_state(self.initial) and self.is_solvable(self.initial):
            current_state = [row[:] for row in self.initial]
        else:
            # Tạo trạng thái ngẫu nhiên có thể giải được
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
            # Kiểm tra xem trạng thái hiện tại có phải là mục tiêu không
            if current_state_tuple == self.goal:
                print(f"Tìm thấy giải pháp sau {steps} lần lặp")
                return path

            # Tìm ô trống
            blank_i, blank_j = find_blank(current_state)
            if blank_i is None:
                print("Trạng thái không hợp lệ: Không tìm thấy ô trống")
                return None

            # Lấy các nước đi khả thi (các ô lân cận để hoán đổi với ô trống)
            neighbors = get_neighbors(blank_i, blank_j)
            if not neighbors:
                print("Không có nước đi hợp lệ")
                return None

            # Đánh giá xung đột cho mỗi nước đi
            best_conflicts = float('inf')
            best_states = []

            for ni, nj in neighbors:
                # Tạo bản sao của trạng thái hiện tại
                temp_state = [row[:] for row in current_state]
                # Hoán đổi ô trống với ô lân cận
                temp_state[blank_i][blank_j], temp_state[ni][nj] = temp_state[ni][nj], temp_state[blank_i][blank_j]
                temp_state_tuple = tuple(tuple(row) for row in temp_state)
                # Bỏ qua nếu trạng thái đã được thăm
                if temp_state_tuple in visited:
                    continue
                # Tính xung đột cho trạng thái mới
                conflicts = count_conflicts(temp_state)
                if conflicts < best_conflicts:
                    best_conflicts = conflicts
                    best_states = [(temp_state, (ni, nj))]
                elif conflicts == best_conflicts:
                    best_states.append((temp_state, (ni, nj)))

            # Nếu không tìm thấy nước đi tốt hơn, chọn ngẫu nhiên một ô lân cận
            if not best_states:
                ni, nj = random.choice(neighbors)
                temp_state = [row[:] for row in current_state]
                temp_state[blank_i][blank_j], temp_state[ni][nj] = temp_state[ni][nj], temp_state[blank_i][blank_j]
                best_states = [(temp_state, (ni, nj))]

            # Chọn ngẫu nhiên một trạng thái tốt nhất
            best_state, _ = random.choice(best_states)
            current_state = best_state
            current_state_tuple = tuple(tuple(row) for row in current_state)
            visited.add(current_state_tuple)
            path.append(current_state_tuple)

            steps += 1

        print("Không tìm thấy giải pháp trong giới hạn max_steps")
        return None

def draw_grid(screen, state, tile_size, offset_x, offset_y):
    # Vẽ lưới trạng thái lên màn hình
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

def print_solution(solution):
    # In ra các bước của giải pháp
    if solution is None or len(solution) == 0:
        print("Không tìm thấy đường đi đến trạng thái mục tiêu!")
        return
    for step, state in enumerate(solution):
        print(f"Bước {step}:")
        for row in state:
            print(row)
        print()
    print("Hoàn thành!")

def run_pygame(puzzle):
    # Chạy giao diện pygame để hiển thị trò chơi
    pygame.init()
    screen = pygame.display.set_mode((600, 400))
    pygame.display.set_caption("8-Puzzle Solver")
    clock = pygame.time.Clock()

    # Danh sách các nút bấm
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
        
        # Vẽ các nút bấm
        for text, pos in buttons:
            pygame.draw.rect(screen, (139, 0, 0), (*pos, 200, 40))
            button_width, button_height = 200, 40
            label = font.render(text, True, (0, 0, 0))
            text_width, text_height = label.get_size()
            text_x = pos[0] + (button_width - text_width) // 2
            text_y = pos[1] + (button_height - text_height) // 2
            screen.blit(label, (text_x, text_y))
        
        # Hiển thị giải pháp từng bước
        if solution and algorithm_run and step < len(solution):
            draw_grid(screen, solution[step], 100, 250, 50)
            if step < len(solution) - 1:
                step += 1
                time.sleep(0.01)  # Tạm dừng 0.01 giây giữa các bước
        else:
            draw_grid(screen, puzzle.initial, 100, 250, 50)

        # Hiển thị thời gian thực thi
        if elapsed_time is not None:
            time_text = font.render(f"Thời gian: {elapsed_time:.10f}s", True, (0, 0, 0))
            screen.blit(time_text, (250, 360))

        # Xử lý sự kiện
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN and not algorithm_run:
                x, y = event.pos
                start_time = time.time()
                
                # Xử lý nút Backtracking
                if 20 <= x <= 220 and 20 <= y <= 60:
                    solution = puzzle.backtracking_search()
                    step = 0
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    print("\nKết quả của thuật toán Backtracking:")
                    print_solution(solution)
                    print(f"Thời gian thực thi: {elapsed_time:.10f} giây")
                    algorithm_run = True
                # Xử lý nút Forward Checking
                elif 20 <= x <= 220 and 70 <= y <= 110:
                    solution = puzzle.forward_checking_search()
                    step = 0
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    print("\nKết quả của thuật toán Forward Checking:")
                    print_solution(solution)
                    print(f"Thời gian thực thi: {elapsed_time:.10f} giây")
                    algorithm_run = True
                # Xử lý nút Min-Conflicts
                elif 20 <= x <= 220 and 120 <= y <= 160:
                    solution = puzzle.min_conflicts_search()
                    step = 0
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    print("\nKết quả của thuật toán Min-Conflicts:")
                    print_solution(solution)
                    print(f"Thời gian thực thi: {elapsed_time:.10f} giây")
                    algorithm_run = True
                # Xử lý nút Reset
                elif 20 <= x <= 220 and 170 <= y <= 210:
                    solution = None
                    step = 0
                    elapsed_time = None
                    algorithm_run = False

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

# Trạng thái ban đầu và mục tiêu
initial_state = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0]
]
goal_state = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
puzzle = EightPuzzle(initial_state, goal_state)
run_pygame(puzzle)