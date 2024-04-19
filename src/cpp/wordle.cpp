#include <iostream>

#include "game/game.hpp"

int main(void)
{
	Game game;
	auto ww = game.get_ww();
	// for (const auto& word : ww)
	// 	std::cout << word << '\n';
	game.set_wd("gleam");
	auto res = game.input("blame");
	for (const auto& r : res)
		std::cout << r << '\n';
}