CC = g++
CXXFLAGS = -Wall -g -O2
LIBS = -fopenmp -larmadillo `pkg-config --cflags --libs opencv`
SRCSDIR = srcs
SRCS = main.cpp ParameterInfo.cpp DisplayImage.cpp AnomalyDetection.cpp
OBDIR = code/objs
OBJS = $(addprefix $(OBDIR)/, $(SRCS:.cpp=.o))

AnomalyDetection: $(OBJS)
		$(CC) $(CXXFLAGS) -o $@ $(OBJS) $(LIBS)
$(OBDIR)/main.o: code/srcs/main.cpp code/headers/main.h code/headers/inc.h
		$(CC) -c $(CXXFLAGS) $< -o $@
$(OBDIR)/ParameterInfo.o: code/srcs/ParameterInfo.cpp code/headers/ParameterInfo.h code/headers/inc.h
		$(CC) -c $(CXXFLAGS) $< -o $@
$(OBDIR)/DisplayImage.o: code/srcs/DisplayImage.cpp code/headers/DisplayImage.h code/headers/inc.h
		$(CC) -c $(CXXFLAGS) $< -o $@
$(OBDIR)/AnomalyDetection.o: code/srcs/AnomalyDetection.cpp code/headers/AnomalyDetection.h code/headers/inc.h
		$(CC) -c $(CXXFLAGS) $< -o $@
clean:
		rm $(OBJS)
