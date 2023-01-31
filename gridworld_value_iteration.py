from enum import Enum
from typing import Final
import random as r


# 행동 정의
class Actions(Enum):
    U = 'U'
    D = 'D'
    L = 'L'
    R = 'R'
    DEFAULT = ' '


DELTA_THRESHOLD: Final[float] = 1e-3  #
GAMMA: Final[float] = 0.9  # 할인율

Pos = tuple[int, int]

class Grid:
    # (i, j) : 현재 플레이어의 좌표
    state: list[int, int]
    rewards: dict[Pos, int]
    actions: dict[Pos, list[Actions]]

    # 생성자
    def __init__(self, row_cnt, col_cnt, start_pos: Pos):
        self.row_cnt = row_cnt
        self.col_cnt = col_cnt
        self.state = list(start_pos)

    # 상태 공간과 행동 공간 정의
    def set(self, rewards: dict[Pos, int], actions: dict[Pos, list[Actions]]):
        self.rewards = rewards
        self.actions = actions

    def is_terminal(self, s: Actions):
        # self.actions는 각 상태에서 할 수 있는 행동들을 dict화해둔 것 -> 이에 없다면 판을 벗어난 것 or 갇힌 것
        return s not in self.actions

    # 결정된 행동을 바탕으로 플레이어를 움직이는 함수
    def move(self, action: Actions) -> int:
        if action in self.actions[self.get_state()]:
            if action == Actions.U:
                self.state[0] -= 1
            elif action == Actions.D:
                self.state[0] += 1
            elif action == Actions.R:
                self.state[1] += 1
            elif action == Actions.L:
                self.state[1] -= 1

        return self.rewards.get(self.get_state(), 0)

    def get_all_states(self):
        return set(self.actions.keys()) | set(self.rewards.keys())

    def get_state(self) -> tuple[int, int]:
        return tuple(self.state)

    def set_state(self, new_state):
        self.state = list(new_state)

def standard_grid() -> Grid:
    grid = Grid(3, 4, (2, 0))
    rewards = {(0, 3): 1, (1, 3): -1}
    actions = {
        (0, 0): [Actions.D, Actions.R],
        (0, 1): [Actions.L, Actions.R],
        (0, 2): [Actions.L, Actions.D, Actions.R],
        (1, 0): [Actions.U, Actions.D],
        (1, 2): [Actions.U, Actions.D, Actions.R],
        (2, 0): [Actions.U, Actions.R],
        (2, 1): [Actions.L, Actions.R],
        (2, 2): [Actions.L, Actions.R, Actions.U],
        (2, 3): [Actions.L, Actions.U]
    }
    grid.set(rewards, actions)
    return grid


def print_values(V, grid):
    for i in range(grid.row_cnt):
        print("---------------------------------------")
        for j in range(grid.col_cnt):
            value = V.get((i, j), 0)
            if value >= 0:
                print("%.2f | " % value, end="")
            else:
                print("%.2f | " % value, end="")  # -ve sign takes up an extra space
        print("")


def print_policy(P, grid):
    for i in range(grid.row_cnt):
        print("---------------------------------------")
        for j in range(grid.col_cnt):
            action = P.get((i, j), Actions.DEFAULT)
            print(f"  {action.value}  |", end="")
        print()


if __name__ == '__main__':
    # 격자 공간 초기화
    grid = standard_grid()

    # 보상을 초기화
    print("\n보상: ")
    print_values(grid.rewards, grid)

    # 초기 정책은 각 상태에서 선택 가능한 행동을 무작위적으로 선택
    policy = {}
    for s in grid.actions.keys():
        policy[s] = r.choice(list(Actions))

    # 정책 입력
    print("\n초기 정책: ")
    print_policy(policy, grid)

    # 가치 함수 V(s) 초기화
    V = {}
    states = grid.get_all_states()
    for s in states:
        if s in grid.actions:
            # 보상을 알 수 없으므로 랜덤으로 두자
            V[s] = r.random()
        else:
            # 종단 상태이므로 그냥 0으로
            V[s] = 0

    # 수렴할 때까지 반복
    i:int = 0
    while True:
        maxChange = 0
        for s in states:
            oldValue = V[s]

            # 종단 상태가 아닌 상태에 대해서만 V(s)를 계산
            if s in policy:
                newValue = float('-inf')

                for a in Actions:
                    grid.set_state(s)
                    r = grid.move(a)

                    #벨만 방정식 계산
                    v = r + GAMMA * V[grid.get_state()]
                    if v > newValue:
                        newValue = v

                V[s] = newValue
                maxChange = max(maxChange, abs(oldValue - V[s]))

        print(f"\n{i}  번째 반복")
        print_values(V, grid)
        i += 1

        # 임계값 이하로 변화량이 떨어지면 수렴했다고 판단
        if maxChange < DELTA_THRESHOLD:
            break

    # 이제 보상함수를 바탕으로 최적 가치 찾는 함수 도출
    for s in policy.keys():
        bestAction = None
        bestValue = float('-inf')

        #가능한 모든 행동에 대해 반복
        for a in Actions:
            grid.set_state(s)
            r = grid.move(a)
            v = r + GAMMA * V[grid.get_state()]

            if v > bestValue:
                bestValue = v
                bestAction = a

        policy[s] = bestAction

    print("\n가치 함수: ")
    print_values(V, grid)

    print("\n정책: ")
    print_policy(policy, grid)