#pragma once
#include <string>
#include <vector>

class Game {
    private:
	std::vector<std::string> ww;
	std::string		 wd;
	std::vector<int>	 res;
	void			 read_ww(void);

    public:
	Game(void);
	const std::vector<std::string> &get_ww(void);
	void				set_wd(const std::string &);
	const std::vector<int>	       &input(const std::string &);
};