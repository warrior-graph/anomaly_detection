CC = g++
CXXFLAGS = -Wall -g -O2
LIBS = -fopenmp -larmadillo `pkg-config --cflags --libs opencv`
SRCSDIR = srcs
SRCS = main.cpp SequentialBge.cpp SequentialBgeParams.cpp
OBDIR = code/objs
OBJS = $(addprefix $(OBDIR)/, $(SRCS:.cpp=.o))

EstimateBackground: $(OBJS)
		$(CC) $(CXXFLAGS) -o $@ $(OBJS) $(LIBS)
$(OBDIR)/main.o: code/srcs/main.cpp code/headers/inc.hpp
		$(CC) -c $(CXXFLAGS) $< -o $@
$(OBDIR)/SequentialBge.o: code/srcs/SequentialBge.cpp code/headers/SequentialBge.hpp code/headers/inc.hpp
		$(CC) -c $(CXXFLAGS) $< -o $@
$(OBDIR)/SequentialBgeParams.o: code/srcs/SequentialBgeParams.cpp code/headers/SequentialBgeParams.hpp code/headers/inc.hpp
		$(CC) -c $(CXXFLAGS) $< -o $@
clean:
		rm $(OBJS)
