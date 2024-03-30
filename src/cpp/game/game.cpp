#include <fstream>
// #include <iostream>

#include "game.hpp"

Game::Game(void)
{
	Game::read_ww();
}

void Game::read_ww(void)
{
	std::ifstream file("game/valid-wordle-words.txt");
	for (std::string word; std::getline(file, word);)
		Game::ww.push_back(word);
}

const std::vector<std::string> &Game::get_ww(void)
{
	return Game::ww;
}

void Game::set_wd(const std::string &w)
{
	Game::wd = w;
}

const std::vector<int> &Game::input(const std::string &guess)
{
	Game::res.clear();
	for (int i = 0; i < 5; i++) {
		Game::res.push_back(0);
		if (guess[i] == Game::wd[i]) {
			Game::wd[i]  = '_';
			Game::res[i] = 2;
		}
	}
	for (int i = 0; i < 5; i++) {
		if (Game::res[i])
			continue;
		for (int j = 0; j < 5; j++) {
			if (guess[i] == Game::wd[j]) {
				Game::res[i] = 1;
				Game::wd[j]  = '_';
				break;
			}
		}
	}
	return Game::res;
}