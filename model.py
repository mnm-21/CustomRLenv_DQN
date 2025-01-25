import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time
from tqdm import tqdm
style.use("ggplot")

SIZE = 15
EPISODES = 40000
MOVE_PENALTY = 1
ENEMY_PENALTY = 200
FOOD_REWARD = 175

epsilon = 0.9
EPS_DECAY = 0.9998
SHOW_EVERY = 5000

start_q_table = None # None or Filename

LEARNING_RATE = 0.1
DISCOUNT = 0.99

PLAYER_N = 1 # player key in dict
FOOD_N = 2 # food key in dict
ENEMY_N = 3 # enemy key in dict

# Color dict in BGR 
d = {1: (255, 175, 0),  # player
    2: (0, 255, 0),     # food
    3: (0, 0, 255)}     # enemy

class Blob:
    def __init__(self):
        self.x = np.random.randint(0, SIZE)
        self.y = np.random.randint(0, SIZE)

    def __str__(self):
        return f"{self.x}, {self.y}"
    
    def __sub__(self, other):
        return (self.x-other.x, self.y-other.y)

    def __eq__(self, other):
        if isinstance(other, Blob):
            return self.x == other.x and self.y == other.y
        return False
    
    def action(self, choice):
        if choice == 0:
            self.move(x=1, y=0) # right
        elif choice == 1:
            self.move(x=-1, y=0) # left
        elif choice == 2:
            self.move(x=0, y=1) # down    
        elif choice == 3:
            self.move(x=0, y=-1) # up   
        elif choice == 4:
            self.move(x=1, y=1) # down right
        elif choice == 5:
            self.move(x=-1, y=1) # down left
        elif choice == 6:
            self.move(x=1, y=-1) # up right
        elif choice == 7:
            self.move(x=-1, y=-1) # up left
        

    def move(self, x=False, y=False):
        if x:
            self.x += x
        if y:    
            self.y += y

        if self.x < 0:
            self.x = 0
        elif self.x > SIZE-1:
            self.x = SIZE-1

        if self.y < 0:  
            self.y = 0
        elif self.y > SIZE-1:
            self.y = SIZE-1

def distance(player, food):
    return abs(player.x - food.x) + abs(player.y - food.y)


if start_q_table is None:
    q_table = {}        # (x1, y1) - distance blob to food blob, (x2, y2) - distance blob to enemy blob
    for x1 in range(-SIZE+1, SIZE):
        for y1 in range(-SIZE+1, SIZE):
            for x2 in range(-SIZE+1, SIZE):
                for y2 in range(-SIZE+1, SIZE):
                    q_table[((x1, y1), (x2, y2))] = [np.random.uniform(-7, 0) for i in range(8)]
else:
    with open(start_q_table, "rb") as f:
        q_table = pickle.load(f)    

episode_rewards = []

for episode in tqdm(range(EPISODES), ascii=True, unit='episodes'):
    # make sure the blobs are not in the same position
    player = Blob()
    # food not on player
    food = Blob()
    while food == player:
        food = Blob()
    # enemy not on player or food    
    enemy = Blob()
    while enemy == player or enemy == food:
        enemy = Blob()

    if episode % SHOW_EVERY == 0 and episode != 0:
        print(f"\n Episode #{episode}")
        print(f"player mean {np.mean(episode_rewards[-SHOW_EVERY:])}")
        show = True 
    else:
        show = False

    episode_reward = 0
    for i in range(200):
        prev_distance = distance(player, food)
        obs = (player-food, player-enemy)

        # Player's action based on Q-learning
        if np.random.random() > epsilon:
            action = np.argmax(q_table[obs])
        else:
            action = np.random.randint(0, 8)
        player.action(action)

        # Enemy moves randomly
        enemy.action(np.random.randint(0, 8))

        new_distance = distance(player, food)
        proximity_reward = 0
        if new_distance < prev_distance:
            proximity_reward = 2  # Reward for getting closer
        else:
            proximity_reward = -5

        if player == food:
            food = Blob()
            while food == player:
                food = Blob()
            reward = FOOD_REWARD

        elif player == enemy:
            enemy = Blob()
            while enemy == player or enemy == food:
                enemy = Blob()
            reward = -ENEMY_PENALTY

        else:
            reward = proximity_reward - MOVE_PENALTY

        new_obs = (player-food, player-enemy)
        max_future_q = np.max(q_table[new_obs])
        current_q = q_table[obs][action]
        new_q = (1-LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
        q_table[obs][action] = new_q

        episode_reward += reward

        if show:
            env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
            env[food.y][food.x] = d[FOOD_N]
            env[player.y][player.x] = d[PLAYER_N]
            env[enemy.y][enemy.x] = d[ENEMY_N]
            img = Image.fromarray(env, 'RGB')
            img = img.resize((300, 300), resample=Image.BOX)
            cv2.imshow("image", np.array(img))
            if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:
                if cv2.waitKey(500) & 0xFF == ord('q'):
                    break
            else:
                if cv2.waitKey(200) & 0xFF == ord('q'):
                    break

        if reward == -ENEMY_PENALTY:
            break

    episode_rewards.append(episode_reward)

    epsilon *= EPS_DECAY

moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,))/SHOW_EVERY, mode='valid')

# Plot rewards for the player
plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.title('Player Rewards')
plt.show()

with open(f"qtable-{int(time.time())}.pickle", "wb") as f:
    pickle.dump(q_table, f)
