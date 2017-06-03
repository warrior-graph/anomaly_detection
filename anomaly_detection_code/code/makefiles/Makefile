CC = g++
CXXFLAGS = -Wall -g -O2
LIBS = -fopenmp -larmadillo `pkg-config --cflags --libs opencv`
SRCS = main.cpp ParameterInfo.cpp DisplayImage.cpp AnomalyDetection.cpp
OBJS = main.o ParameterInfo.o DisplayImage.o AnomalyDetection.o

AnomalyDetection: main.o ParameterInfo.o DisplayImage.o AnomalyDetection.o
		$(CC) $(CXXFLAGS) -o AnomalyDetection $(OBJS) $(LIBS)
main.o: main.cpp main.h inc.h
		$(CC) $(CXXFLAGS) -c main.cpp
ParameterInfo.o: ParameterInfo.cpp ParameterInfo.h inc.h
		$(CC) $(CXXFLAGS) -c ParameterInfo.cpp
DisplayImage.o: DisplayImage.cpp DisplayImage.h inc.h
		$(CC) $(CXXFLAGS) -c DisplayImage.cpp
AnomalyDetection.o: AnomalyDetection.cpp AnomalyDetection.h inc.h
		$(CC) $(CXXFLAGS) -c AnomalyDetection.cpp $(LIBs)
clean:
		rm *.o
