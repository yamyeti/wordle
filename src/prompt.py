import openai
from openai import OpenAI
from dotenv import load_dotenv
import os

class ChatGPTClient:
	def __init__(self, api_key):
		openai.api_key = api_key
		self.client = OpenAI()


	def get_completion(self, sys_prompt, *user_inputs, engine="gpt-3.5-turbo"):
		input_str = ', '.join([f'"{inp}' for inp in user_inputs])
		prompt = f"{sys_prompt}\n```{input_str}```"
		completion = self.client.chat.completions.create(
			model=engine,
			messages=[
				{"role": "system", "content": "You are a strategic assistant that loves words and probability."},
				{"role": "user", "content": prompt}
			]
		)
		return completion.choices[0].message.content

def main():
	load_dotenv()
	api_key = os.environ.get('OPENAI_API_KEY') # Retrieve the API key from the .env variable
	gpt = ChatGPTClient(api_key)
	prompt= "Given the previous Wordle solutions delimited by backticks\
			what is the most helpful starting word for the next game?\
			Please explain your decision and enclose the guess in\
			square brackets."
	prev_one=input("Enter yesterday's Wordle solution: ")
	prev_two=input("Enter the previous Wordle solution: ")
	prev_three=input("Enter the Wordle solution before: ")

	completion=gpt.get_completion(prompt, prev_three, prev_two, prev_one)
	print(completion)

if __name__ == '__main__':
	main()