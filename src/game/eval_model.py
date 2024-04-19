import numpy as np
from tensorflow.keras.models import load_model
from wordle import Wordle, Letter
import random

class WordleEnv:

    def __init__(self):
        file_name = 'valid-wordle-words.txt'
        with open(file_name, 'r') as file:
            self.all_words = [word.strip() for word in file.readlines()]
        self.game = Wordle(file_name)
        self.reset()

    def reset(self):
        self.game.set_word_of_the_day()
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
            return self.state.flatten(), reward, done, {}

        guess_word = self.possible_words[action]
        #print(f"Attempting guess with word: {guess_word}")

        # Execute the guess to get feedback
        feedback = self.game.guess(guess_word)
        self.feedback_history.append(feedback)
        done = False
        reward = self.calculate_feedback_reward(feedback)

        # update the game state based on feedback
        self.update_state_with_feedback(feedback)

        # Determine if the game is done based on the feedback
        if all(fb == Letter.GREEN for fb in feedback):
            done = True  # Correct guess signals game completion
            reward += 1  # Assign additional reward for guessing correctly

        self.tries_remaining -= 1
        if self.tries_remaining == 0:
            done = True  # No tries left also signals game completion

        # Remove guessed word from the list of possible words
        self.possible_words.remove(guess_word)
        #print(f"Removed '{guess_word}' from possible words.")

        # Apply feedback to further filter the possible words
        self.filter_possible_words(guess_word, feedback)

        #print(f"Remaining possible words: {len(self.possible_words)}")

        feedback_symbols = ''.join(['G' if fb == Letter.GREEN else 'Y' if fb == Letter.YELLOW else '_' for fb in feedback])
        print(f"Try #{6 - self.tries_remaining}: Guess='{guess_word}', Feedback='{feedback_symbols}', Reward={reward}")

        return self.state.flatten(), reward, done, {}


    def get_possible_words_count(self):
        return len(self.possible_words)

    def calculate_feedback_reward(self, feedback):
        reward = sum([0.5 if fb == Letter.GREEN else 0.2 if fb == Letter.YELLOW else -0.1 for fb in feedback])
        if not self.is_progress(feedback):
            reward -= 0.5  # Adjust this value based on your penalty preference
        return reward

    def filter_possible_words(self, guess, feedback):
        # Save the current state before filtering
        self.previous_possible_words_stack.append(self.possible_words.copy())

        new_possible_words = []
        for word in self.possible_words:
            if self.is_word_possible(guess, feedback, word):
                new_possible_words.append(word)

        # Implement fallback logic: revert to the previous state if overfiltering occurs
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
            return True  # True for the first guess or if there's no history to compare against

        prev_numeric_feedback = [self.letter_to_numeric(fb) for fb in self.feedback_history[-1]]

        # compare current feedback with previous to determine improvement
        improvement = any(curr > prev for curr, prev in zip(numeric_feedback, prev_numeric_feedback))
        return improvement

    def letter_to_numeric(self, letter):
        # Convert letter enum to numeric value
        if letter == Letter.GREEN:
            return 2
        elif letter == Letter.YELLOW:
            return 1
        else:
            return 0

class SimplifiedDQNAgent:
    def __init__(self, model, env):
        self.gamma = 0.95
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = model
        self.env = env
        self.state_size = len(env.reset())
        self.action_size = len(env.possible_words)
        self.epsilon = 0.01  # Set low exploration rate for evaluation

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            # Randomly select action
            available_actions = [i for i in range(len(self.env.possible_words)) if i not in self.env.guess_indices]
            action = random.choice(available_actions) if available_actions else random.randrange(self.action_size)
        else:
            # Predict action
            state = np.reshape(state, [1, self.state_size])
            act_values = self.model.predict(state)
            # invalidate actions corresponding to previously guessed words
            for i in range(self.action_size):
                if i in self.env.guess_indices or i >= len(self.env.possible_words):
                    act_values[0][i] = float('-inf')
            action = np.argmax(act_values[0])
        return action

def evaluate_agent(env, agent, episodes=100):
    success_count = 0
    total_guesses_on_success = 0

    for episode in range(episodes):
        state = env.reset()
        done = False
        guesses = 0
        last_reward = 0

        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            guesses += 1
            last_reward = reward

            if done and last_reward == 3.0:  # Successful guess
                success_count += 1
                total_guesses_on_success += guesses
                print(f"Episode {episode + 1}: Success with {guesses} guesses.")
            elif done:
                print(f"Episode {episode + 1}: Failed to solve after {guesses} guesses.")

    average_guesses_on_success = total_guesses_on_success / success_count if success_count > 0 else 0
    print(f"\nEvaluation Summary over {episodes} episodes:")
    print(f"Number of Successfully Finished Games: {success_count}")
    print(f"Average Number of Guesses per Successful Game: {average_guesses_on_success}")

if __name__ == "__main__":
    env = WordleEnv()
    model_path = 'best_wordle_model.h5'
    model = load_model(model_path)
    agent = SimplifiedDQNAgent(model, env)
    evaluate_agent(env, agent, episodes=100)
