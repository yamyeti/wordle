#include <string>
#include <vector>

class Game {
    private:
	std::vector<std::string> ww;
	std::string wd;

	void read_ww(void);

    public:
	Game(void);
	~Game(void);

	std::vector<std::string> get_ww(void);
	void set_wd(std::string);
};
