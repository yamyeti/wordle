#include <string>

#include "inverted.hpp"

InvertedIndex::InvertedIndex(const std::vector<std::string> &docs)
	: docs(docs)
{
	InvertedIndex::build();
}

void InvertedIndex::build(void)
{
	InvertedIndex::find_indices();
	for (const auto &key : InvertedIndex::keys) {
		// InvertedIndex::index.insert(std::make_pair(int, int));
	}
}

void InvertedIndex::find_indices(void)
{
	InvertedIndex::keys.clear();
	for (const auto &doc : InvertedIndex::docs) {
		for (int i = 0; i < 5; i++) {
			InvertedIndex::keys.insert(doc[i] + std::to_string(i));
		}
		if (InvertedIndex::keys.size() == 130)
			break;
	}
}

const std::map<std::string, std::vector<std::string> > &
InvertedIndex::get_index(void)
{
	return InvertedIndex::index;
}