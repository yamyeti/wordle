#pragma once
#include <map>
#include <set>
#include <vector>

class InvertedIndex {
    private:
	std::set<std::string>				 keys;
	std::map<std::string, std::vector<std::string> > index;
	const std::vector<std::string>			&docs;
	void						 build(void);
	void						 find_indices(void);

    public:
	InvertedIndex(const std::vector<std::string> &);
	const std::map<std::string, std::vector<std::string> > &get_index(void);
};