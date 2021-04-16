
CXXFLAGS += -Wall -Wextra -g -std=c++0x -O3 -DNDEBUG

.phony: all dither release

all: release

release: sequential.cpp
	g++ sequential.cpp -o dither $(CXXFLAGS)

clean:
	rm -f ./dither
	rm -f ./dither