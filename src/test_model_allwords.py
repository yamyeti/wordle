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
import time


class WordleEnv:

    def __init__(self):
        self.game = Wordle()
        self.all_words = self.game.get_wordle_words()
        self.reset()

    def reset(self, specific_word=None):
        if specific_word:
            if specific_word in self.game.wordle_words:
                self.game.word_of_the_day = specific_word
            else:
                raise ValueError("Word not in the list of valid Wordle words.")
        else:
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
        if action is None:
            print("No valid action provided to step function.")
            # Handle this scenario appropriately, e.g., end the game or take no action
            return self.state.flatten(), -1, True, None  # Assume -1 reward for no action and mark done as True

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
                # Additional check: if a letter is marked yellow, it means there should be another instance of the letter
                # either in a different position (if it's also present as green or yellow elsewhere)
                # or not at all (if it's only present as yellow and not found in the word in any position)
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
    # Define the total number of features for the state input
    total_state_features = 90  # This should include all features from your environment state

    # Inputs
    state_input = Input(shape=(total_state_features,), name='state_input')
    word_input = Input(shape=(word_length,), name='word_input')  # This is the input for word indices

    # Embedding for word input
    embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=word_length)(word_input)
    flat_embedding = Flatten()(embedding)

    # Concatenate state and word embeddings
    concatenated = Concatenate()([state_input, flat_embedding])

    # Dense layers
    x = Dense(128, activation='relu')(concatenated)
    x = Dropout(0.2)(x)
    x = Dense(256, activation='relu')(x)
    output = Dense(action_size, activation='linear')(x)  # Ensure action size matches the output layer

    model = Model(inputs=[state_input, word_input], outputs=output)
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
    return model




class DQNAgent:
    def __init__(self, state_size, action_size, env, start_word_index, char_to_index, model):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = model
        self.env = env
        self.start_word_index = start_word_index
        self.char_to_index = char_to_index  # Store the mapping in the agent

    def encode_word(self, word):
        if word is None:
            return np.zeros((5,), dtype=int)  # Assuming word length is 5 and 0 is a placeholder index
        return np.array([self.char_to_index.get(char, 0) for char in word.lower()])


    def act(self, state, use_start_word=False):
        if len(self.env.possible_words) == 0:
            print("No possible words left to guess.")
            return None, None  # No action, no indices

        if use_start_word and not self.env.guess_history:
            action = self.start_word_index
            word_indices = self.encode_word(self.env.all_words[action])
            return action, word_indices

        if np.random.rand() <= self.epsilon:
            action = random.choice(range(len(self.env.possible_words)))
        else:
            state_input = np.array([state]).reshape(1, -1)
            word_embeddings = np.array([self.encode_word(word) for word in self.env.possible_words]).reshape(-1, 5)
            state_input_expanded = np.repeat(state_input, len(self.env.possible_words), axis=0)

            q_values = self.model.predict([state_input_expanded, word_embeddings])
            action = np.argmax(q_values[0])

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
        for item in minibatch:
            state, word_indices, action, reward, next_state, next_word_indices, done = item

            # Reshape inputs to match the input requirement of the model
            state_array = np.array(state).reshape(1, -1)  # Reshape state to be 2D
            word_indices_array = np.array(word_indices).reshape(1, -1)  # Reshape word indices to be 2D

            # Prepare next state and word indices for prediction
            next_state_array = np.array(next_state).reshape(1, -1)
            next_word_indices_array = np.array(next_word_indices).reshape(1, -1)

            # debug print statements
            #print("Shape of state_array:", state_array.shape)
            #print("Shape of word_indices_array:", word_indices_array.shape)
            #print("Shape of next_state_array:", next_state_array.shape)
            #print("Shape of next_word_indices_array:", next_word_indices_array.shape)

            # Predict the next Q-values
            if not done:
                next_q_values = self.model.predict([next_state_array, next_word_indices_array])
                target = reward + self.gamma * np.max(next_q_values[0])
            else:
                target = reward

            # Current Q-values prediction for updating
            target_f = self.model.predict([state_array, word_indices_array])
            target_f[0][action] = target  # Update the target for the performed action

            # Fit the model to the target
            self.model.fit([state_array, word_indices_array], target_f, epochs=1, verbose=0)

            # Epsilon decay
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

        # Print epsilon to track its decay
        print("Current epsilon:", self.epsilon)



# load word list and initialize environment

word_list_path = 'game/valid-wordle-words.txt'
with open(word_list_path, 'r') as file:
    word_list = [word.strip() for word in file.readlines()]


char_to_index = {chr(i): i - 97 for i in range(97, 123)}  # Create a mapping for 'a' to 'z'


env = WordleEnv()

state = env.reset()
state_size = len(state)
action_size = len(word_list)  # Assuming each action corresponds to a word in the list

def evaluate_agent(env, agent, output_file, use_start_word=False, start_word=None):
    # Ensure all words are sorted alphabetically
    all_words_sorted = sorted(env.all_words)
    all_letters = sorted(set(word[0] for word in all_words_sorted))

    with open(output_file, "w") as file:
        file.write("Evaluation of each starting letter:\n\n")
        if use_start_word and start_word:
            file.write(f"Using '{start_word}' as the start word for the first guess in each game.\n\n")

        for letter in all_letters:
            words = [word for word in all_words_sorted if word.startswith(letter)]
            file.write(f"Words starting with '{letter}':\n")

            success_count = 0
            total_guesses = 0
            total_time = 0

            for word in words:
                start_time = time.time()  # Start time measurement for this word
                state = env.reset(specific_word=word)  # Reset environment with a specific word
                state = np.reshape(state, [1, agent.state_size])

                done = False
                guesses = 0
                last_reward = 0

                # Check if start_word should be used for the first guess
                if use_start_word and start_word:
                    action = env.all_words.index(start_word) # Get the action index for the start word
                    state, reward, done, guess_word = env.step(action)
                    state = np.reshape(state, [1, agent.state_size])
                    guesses += 1

                while not done:
                    action, word_indices = agent.act(state)
                    next_state, reward, done, guess_word = env.step(action)
                    guesses += 1
                    last_reward = reward
                    next_state = np.reshape(next_state, [1, agent.state_size])
                    state = next_state

                end_time = time.time()
                word_time = end_time - start_time
                total_guesses += guesses
                total_time += word_time

                # Only count as success if the last reward was exactly 3.0
                if last_reward == 3.0:
                    success_count += 1

            average_guesses_per_word = total_guesses / len(words) if words else 0
            average_time_per_word = total_time / len(words) if words else 0
            success_percentage = (success_count / len(words)) * 100 if words else 0

            file.write(f"Number of words:\t\t{len(words)}\n")
            file.write(f"Number of words successfully solved:\t{success_count}\n")
            file.write(f"Percentage of words successfully solved:\t{success_percentage:.2f}%\n")
            file.write(f"Total time taken:\t\t{total_time:.4f}s\n")
            file.write(f"Average tries per word:\t{average_guesses_per_word:.3f}\n")
            file.write(f"Average time per word:\t{average_time_per_word:.4f}s\n\n")


if __name__ == "__main__":
    file_name = 'game/valid-wordle-words.txt'
    env = WordleEnv()
    model_path = 'wordle_model.h5'
    start_word = 'slate'
    if start_word in env.all_words:
        start_word_index = env.all_words.index(start_word)
    else:
        raise ValueError(f"'{start_word}' is not in the word list.")
    model = load_model(model_path)
    agent = DQNAgent(state_size, action_size, env, start_word_index, char_to_index, model)
    output_file_path = "evaluation_results_all.txt"
    agent.epsilon = 0.01
    evaluate_agent(env, agent, output_file_path, use_start_word=True, start_word=start_word)
