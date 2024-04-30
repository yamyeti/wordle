import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Embedding, Flatten, Concatenate
from tensorflow.keras.optimizers import legacy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
import random
from collections import deque
from game.wordle import Wordle, Letter



class WordleEnv:

    def __init__(self):
        self.game = Wordle()
        self.all_words = self.game.get_wordle_words()
        self.reset()

    def reset(self):
        self.game.random_word_of_the_day()
        self.tries_remaining = 6
        self.state = np.zeros((self.tries_remaining, 5, 3))
        self.guess_history = []  # track words guessed
        self.guess_indices = []  # track indices of guessed words
        self.feedback_history = []
        self.possible_words = self.all_words.copy()  # reset possible words to full list
        self.previous_possible_words_stack = []  # save previous states
        return self.state.flatten()

    def step(self, action):
        if action < 0 or action >= len(self.possible_words):
            print("Invalid guess: heavy penalty applied.")
            reward = -1
            self.tries_remaining -= 1
            done = self.tries_remaining == 0
            return self.state.flatten(), reward, done, None  # Return None for guess_word when invalid

        guess_word = self.possible_words[action]
        # Execute the guess to get feedback
        feedback = self.game.guess(guess_word)
        self.feedback_history.append(feedback)
        done = False
        reward = self.calculate_feedback_reward(feedback)

        # Update the game state based on feedback
        self.update_state_with_feedback(feedback)

        # Determine if the game is done based on the feedback
        if all(fb == Letter.GREEN for fb in feedback):
            done = True  # Correct guess signals game completion
            reward += 1  # Assign additional reward for guessing correctly

        self.tries_remaining -= 1
        if self.tries_remaining == 0:
            done = True  # No tries left also signals game completion

        # Remove the guessed word from the list of possible words
        self.possible_words.remove(guess_word)

        # Apply feedback to further filter the possible words
        self.filter_possible_words(guess_word, feedback)

        feedback_symbols = ''.join(['G' if fb == Letter.GREEN else 'Y' if fb == Letter.YELLOW else '_' for fb in feedback])
        print(f"Try #{6 - self.tries_remaining}: Guess='{guess_word}', Feedback='{feedback_symbols}', Reward={reward}")

        return self.state.flatten(), reward, done, guess_word



    def get_possible_words_count(self):
        return len(self.possible_words)

    def calculate_feedback_reward(self, feedback):
        reward = sum([0.5 if fb == Letter.GREEN else 0.2 if fb == Letter.YELLOW else -0.1 for fb in feedback])
        if not self.is_progress(feedback):
            reward -= 0.5
        return reward

    def filter_possible_words(self, guess, feedback):
        # Save current state before filtering
        self.previous_possible_words_stack.append(self.possible_words.copy())

        new_possible_words = []
        for word in self.possible_words:
            if self.is_word_possible(guess, feedback, word):
                new_possible_words.append(word)

        # fallback logic: revert to the previous state if overfiltering occurs
        if not new_possible_words and self.previous_possible_words_stack:
            #print("Fallback: Reverting to previous list of possible words due to overfiltering.")
            self.possible_words = self.previous_possible_words_stack.pop()
        else:
            self.possible_words = new_possible_words


    def is_word_possible(self, guess, feedback, possible_word):
        yellow_letters = {}  # track positions where each letter was marked yellow

        for i, (g, f) in enumerate(zip(guess, feedback)):
            if f == Letter.GREEN and g != possible_word[i]:
                return False
            elif f == Letter.YELLOW:
                # collect positions of yellow-marked letters
                if g in yellow_letters:
                    yellow_letters[g].add(i)
                else:
                    yellow_letters[g] = {i}
            elif f == Letter.GRAY:
                # check for non-repeated gray letters that appear in possible word
                if guess.count(g) == 1 and g in possible_word:
                    return False

        for yl, positions in yellow_letters.items():
            if yl not in possible_word:
                return False
            for pos in positions:
                if possible_word[pos] == yl:  # yellow letter should not be in the same position
                    return False
                if guess.count(yl) == 1 and possible_word.count(yl) == 1 and yl in possible_word[pos]:
                    return False

        return True



    def update_state_with_feedback(self, feedback):
        try_index = 6 - self.tries_remaining
        for i, fb in enumerate(feedback):
            self.state[try_index, i, :] = 0
            self.state[try_index, i, fb.value + 1] = 1

    def is_progress(self, feedback):
        # convert feedback to numeric values for comparison
        numeric_feedback = [self.letter_to_numeric(fb) for fb in feedback]

        if not self.feedback_history:
            return True  # True for first guess or if there's no history to compare against

        prev_numeric_feedback = [self.letter_to_numeric(fb) for fb in self.feedback_history[-1]]

        # compare current feedback with previous to determine improvement
        improvement = any(curr > prev for curr, prev in zip(numeric_feedback, prev_numeric_feedback))
        return improvement

    def letter_to_numeric(self, letter):
        # convert letter enum to numeric value
        if letter == Letter.GREEN:
            return 2
        elif letter == Letter.YELLOW:
            return 1
        else:
            return 0

def build_model(state_size, action_size, vocab_size=26, embedding_dim=5, word_length=5):
    # State input: expects a flat vector of shape (90,)
    state_input = Input(shape=(state_size,), name='state_input')

    # Word input: expects integer indices of shape (word_length,)
    word_input = Input(shape=(word_length,), dtype='int32', name='word_input')

    # Embedding layer: outputs a 3D tensor of shape (batch_size, word_length, embedding_dim)
    embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=word_length)(word_input)

    # Flatten the output of the embedding layer to shape (batch_size, word_length*embedding_dim)
    flat_embedding = Flatten()(embedding)

    # Concatenate the flattened embeddings with the state vector
    concatenated = Concatenate()([state_input, flat_embedding])

    # Dense layers following the concatenated inputs
    x = Dense(128, activation='relu')(concatenated)
    x = Dropout(0.2)(x)
    x = Dense(256, activation='relu')(x)
    output = Dense(action_size, activation='linear')(x)

    # Compile the model
    model = Model(inputs=[state_input, word_input], outputs=output)
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))

    return model


class DQNAgent:
    def __init__(self, state_size, action_size, env, start_word_index, char_to_index, vocab_size=26, embedding_dim=5, word_length=5):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.env = env
        self.start_word_index = start_word_index
        self.char_to_index = char_to_index  # Store the mapping in the agent

        # Build the model with the correct dimensions
        self.model = build_model(state_size, action_size, vocab_size, embedding_dim, word_length)

    def encode_word(self, word):
        if word is None:
            return np.zeros((5,), dtype=int)  # Assuming word length is 5 and 0 is a placeholder index
        return np.array([self.char_to_index.get(char, 0) for char in word.lower()])


    def act(self, state, use_start_word=False):
        if len(self.env.possible_words) == 0:
            print("No possible words left to guess.")
            return None, None  # No action, no indices

        # Always initialize word_indices_input to ensure it's defined
        word_indices_input = None

        # Preparing state input
        state_input = np.array([state]).reshape(1, -1)  # Reshape state to 2D: (1, state_size)

        if use_start_word and not self.env.guess_history:
            action = self.start_word_index
            word_indices = self.encode_word(self.env.all_words[action])
            word_indices_input = np.array([word_indices]).reshape(1, -1)  # Ensure this is the correct shape
        else:
            if np.random.rand() <= self.epsilon:
                # Randomly select an action if below epsilon
                action = np.random.choice(len(self.env.possible_words))
                word_indices = self.encode_word(self.env.possible_words[action])
                word_indices_input = np.array([word_indices]).reshape(1, -1)
            else:
                # Prepare word indices for all possible words
                word_indices_input = np.array([self.encode_word(word) for word in self.env.possible_words])
                # Replicate state input to match the number of words being evaluated
                state_input = np.tile(state_input, (len(self.env.possible_words), 1))

                # Predict Q-values using the model and select the action with the highest Q-value
                q_values = self.model.predict([state_input, word_indices_input])
                action = np.argmax(q_values)

        if action >= len(self.env.possible_words):
            #print(f"Action {action} is out of range. Adjusting action to valid range.")
            action = action % len(self.env.possible_words)  # Simple modulo to ensure the action is valid

        word_indices = self.encode_word(self.env.possible_words[action])
        return action, word_indices

    def remember(self, state, word_indices, action, reward, next_state, next_word_indices, done):
        self.memory.append((state, word_indices, action, reward, next_state, next_word_indices, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, word_indices, action, reward, next_state, next_word_indices, done in minibatch:
            # Prepare inputs for the model
            state_input = np.array([state]).reshape(1, -1)
            word_indices_input = np.array([word_indices]).reshape(1, -1)

            # Debugging print statements to verify shapes
            #print("State input shape:", state_input.shape)  # Expected: (1, 90)
            #print("Word indices input shape:", word_indices_input.shape)  # Expected: (1, 5)

            # Predict Q-values for the current state
            current_q_values = self.model.predict([state_input, word_indices_input])
            target = reward

            # If the game is not done, predict future Q-values for updating the target
            if not done:
                next_state_input = np.array([next_state]).reshape(1, -1)
                next_word_indices_input = np.array([next_word_indices]).reshape(1, -1)
                next_q_values = self.model.predict([next_state_input, next_word_indices_input])
                target += self.gamma * np.max(next_q_values[0])

            # Update current Q-values with the new target
            target_f = current_q_values.copy()
            target_f[0][action] = target  # Update the action with new Q-value

            # Fit the model (this may be batched for efficiency outside the loop)
            self.model.fit([state_input, word_indices_input], target_f, epochs=1, verbose=0)

            # Epsilon decay (if applicable)
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

# load word list and initialize environment

word_list_path = 'game/valid-wordle-words.txt'
with open(word_list_path, 'r') as file:
    word_list = [word.strip() for word in file.readlines()]


# set starting guess
start_word = 'slate'
if start_word in word_list:
    start_word_index = word_list.index(start_word)
else:
    raise ValueError(f"'{start_word}' is not in the word list.")


char_to_index = {chr(i): i - 97 for i in range(97, 123)}  # Create a mapping for 'a' to 'z'


env = WordleEnv()

state = env.reset()
state_size = len(state)
action_size = len(word_list)  # Assuming each action corresponds to a word in the list

agent = DQNAgent(state_size, action_size, env, start_word_index, char_to_index)

# Adjust epsilon_decay to ensure more gradual exploration reduction
agent.epsilon_decay = 0.995
agent.epsilon_min = 0.01

# Increase total_episodes for more training opportunities
total_episodes = 1000
episode_rewards = []


reset_epsilon_every = 100  # Reset epsilon to 1.0 every 100 episodes


# early stopping
best_model_path = 'best_wordle_model.h5'
best_average_reward = -float('inf')
patience = 20
min_improvement = 0.01
patience_counter = 0

for e in range(total_episodes):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    total_reward = 0
    done = False
    first_try = True  # Flag to indicate the first try in the episode

    while not done:
        if first_try:
            action, word_indices = agent.act(state, use_start_word=True)  # Use start word for the first try
            first_try = False  # Update the flag
        else:
            action, word_indices = agent.act(state)  # Let the agent decide the action

        if action is None:
            print("No valid actions possible; ending game.")
            break  # Exit the loop if no actions are possible

        next_state, reward, done, guess_word = env.step(action)

        if guess_word is not None:
            next_word_indices = agent.encode_word(guess_word)
        else:
            next_word_indices = np.zeros((5,), dtype=int)  # Use a default or error state

        total_reward += reward
        next_state = np.reshape(next_state, [1, agent.state_size])
        agent.remember(state, word_indices, action, reward, next_state, next_word_indices, done)
        state = next_state


    episode_rewards.append(total_reward)

    agent.replay(32)



    # Epsilon adjustment
    if (e + 1) % reset_epsilon_every == 0:
        agent.epsilon = 1.0
    else:
        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)

    # Logging
    if (e + 1) % 10 == 0:
        recent_average_reward = np.mean(episode_rewards[-10:])
        print(f"Episode: {e + 1}, Recent Average Reward: {recent_average_reward}, Epsilon: {agent.epsilon}")

    # early stopping and model checkpointing based on average reward
    if (e + 1) % 10 == 0:
        current_average_reward = np.mean(episode_rewards[-10:])
        if current_average_reward > best_average_reward + min_improvement:
            best_average_reward = current_average_reward
            agent.model.save(best_model_path)  # save current best model
            print(f"Saved improved model at episode {e + 1}.")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at episode {e + 1}. No improvement for {patience} episodes.")
                break  # Stop training


def evaluate_agent(env, agent, episodes=100):
    success_count = 0  # count of successfully finished games
    total_guesses_on_success = 0  # accumulate guesses on successful games

    for e in range(episodes):
        state = env.reset()  # reset environment for a new episode
        state = np.reshape(state, [1, agent.state_size])
        done = False
        guesses = 0
        last_reward = 0  # track last reward
        first_try = True  # Flag to indicate the first try in the episode
        total_reward = 0  # Initialize total_reward for the episode

        while not done:
            if first_try:
                action, word_indices = agent.act(state, use_start_word=True)  # Use start word for the first try
                first_try = False  # Update the flag
            else:
                action, word_indices = agent.act(state)  # Let the agent decide the action

            if action is None:
                print("No valid actions possible; ending game.")
                break  # Exit the loop if no actions are possible

            next_state, reward, done, guess_word = env.step(action)

            if guess_word is not None:
                next_word_indices = agent.encode_word(guess_word)
            else:
                next_word_indices = np.zeros((5,), dtype=int)  # Use a default or error state

            total_reward += reward
            guesses += 1
            last_reward = reward
            next_state = np.reshape(next_state, [1, agent.state_size])
            state = next_state

        # Evaluate success at the end of an episode
        if last_reward > 1:  # Check if the last reward was a win bonus
            success_count += 1
            total_guesses_on_success += guesses

    average_guesses_on_success = total_guesses_on_success / success_count if success_count > 0 else 0

    print(f"\nEvaluation Summary over {episodes} episodes:")
    print(f"Number of Successfully Finished Games: {success_count}")
    print(f"Average Number of Guesses per Successful Game: {average_guesses_on_success}")


# Set agent to evaluation mode (less exploratory)
agent.epsilon = 0.01
evaluate_agent(env, agent, episodes=100)
