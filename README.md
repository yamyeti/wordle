# Retrieval-with-Wordle
Contributors: Martinus Kleiweg, Shawn Kim, and Matthew Hernandez
> **Note** Access to our paper [here](dont-forget-this)

Last updated April 30th, 2024.

> **Note** The scope of this project involves the following points: indexing and retrieval, measuring performance, error analysis, and a proposed improved implementation.

This project was created for the purpose of applying techniques in Information Retrieval (IR) to develop a strategy to efficiently play Wordle.

## Table of Contents
- [Objective](#objective)
- [Instructions](#instructions)
- [See More](#see-more)

## Objective
In this repository we describe our end-to-end-implementation of the popular Wordle game, using various Information Retrieval (IR) techniques, together with Reinforcement Learning and large language models. The goal of the game is to guess the word-of-the-day under six attempts with the help of feedback in the form of colored tiles. The system operates on algorithms that index five-letter words and perform a boolean search over them. We present analysis over two popular starting words, and make use of an inverted-index to reduce the search after each guess.

## Instructions
> The code is intended to be run in the terminal. There are two main files to run: ```benchmark_inv_index_v2.py``` and ```prompt.py```. You will need a list of previous solutions to Wordle to use the prompt file. This can be found [here](https://wordfinder.yourdictionary.com/wordle/answers/).

In order to reproduce the results from the paper please follow these steps: 

1. Clone the Repository
```bash
git clone git@github.com:weezymatt/Retrieval-with-Wordle.git
```
2. Change Directory
  ```bash
  cd src/
  ```
3. Running the ```prompt.py``` file requires you to have an OpenAI account to be able to programatically run prompts with an API key as an environment variable. Skip if you are not interested. Unfortunately, you must fund your account ($5.00 minimum) even though you can run some free API calls. Sorry.
- Create an environment variable.
  ```bash
  nano .env
  OPENAI_API_KEY=<paste-your-openai-key-here>
  ```
- Run the prompt and use the previous solutions to Wordle.
  ```bash
  python3 prompt.py
  ```
- Read the prompt and use the recommended word in brackets.
  > To choose the most helpful starting word for the next game of Wordle, I will consider the previous solutions - "vapid," "gleam," and "prune.

  > Looking at these words, I see that they are quite diverse in terms of their starting letters and vowel/consonant distributions. To increase our chances of hitting      > the target word in the fewest guesses possible, I will go with a word that has a good mix of vowels and consonants, as well as a variety of starting letters.

  > Considering this information, a good starting word could be "charm" **[charm]**. This word has a nice balance of vowels and consonants, and the starting letter   > is different from the previous solutions. The variety in letters can help cover a wider range of possible words in the Wordle game.

4. The default code in ```benchmark_inv_index_v2.py``` is written such that it will be run against all the letters for the selected character. Run the benchmark file to test the main system. 
- Choose the starting letter for the word-of-the-day or a word that you want to test.
  ```python
  def main():
    guess = input("Provide your guess: ")
    b = BenchmarkInvIndex(guess.lower())
    # b.benchmark_alphabet()
    b.benchmark_words_starting_with('<LETTER>')
  ```
- Run the benchmark by providing a guess to initialize the intersection.
  > Tip: You may use adieu, slate, ChatGPT's recommendation, or your choice!
  ```bash
  python3 benchmark_inv_index_v2.py
  Provide your guess: <INSERT-YOUR-WORD>
  ```
5. A text file of statistics is printed in this directory and you can see how your guess faired against the word of the day. 

## see more?
