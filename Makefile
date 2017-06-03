CC = g++
CXXFLAGS = -Wall -g -O2
LIBS = -fopenmp -larmadillo `pkg-config --cflags --libs opencv`
SRCSDIR = srcs
SRCS = main.cpp ParameterInfo.cpp DisplayImage.cpp AnomalyDetection.cpp
OBDIR = objs
OBJS = $(addprefix $(OBDIR)/, $(SRCS:.cpp=.o))

AnomalyDetection: $(OBJS)
		$(CC) $(CXXFLAGS) -o AnomalyDetection $(OBJS) $(LIBS)
$(OBDIR)/main.o: srcs/main.cpp headers/main.h headers/inc.h
		$(CC) -c $(CXXFLAGS) $< -o $@
$(OBDIR)/ParameterInfo.o: srcs/ParameterInfo.cpp headers/ParameterInfo.h headers/inc.h
		$(CC) -c $(CXXFLAGS) $< -o $@
$(OBDIR)/DisplayImage.o: srcs/DisplayImage.cpp headers/DisplayImage.h headers/inc.h
		$(CC) -c $(CXXFLAGS) $< -o $@
$(OBDIR)/AnomalyDetection.o: srcs/AnomalyDetection.cpp headers/AnomalyDetection.h headers/inc.h
		$(CC) -c $(CXXFLAGS) $< -o $@
clean:
		rm $(OBJS)
