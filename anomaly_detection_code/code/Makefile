CC = g++
CXXFLAGS = -stt=c++14 -Wall -g -O3
LIBS = -fopenmp -larmadillo `pkg-config --cflags --libs opencv`
SRCS = main.cpp ParameterInfo.cpp DisplayImage.cpp AnomalyDetection.cpp

anomalydetection.o :AnomalyDetection.cpp AnomalyDetection.h
	$(CC) $(CXXFLAGS)
displayimage.o :DisplayImage.cpp DisplayImage.h
	$(CC) $(CXXFLAGS)
parameterinfo.o :ParameterInfo.cpp ParameterInfo.h
	$(CC) $(CXXFLAGS)

AnomalyDetection: anomalydetection.o displayimage.o parameterinfo.o
	$(CC) $(OBJS) $(CXXFLAGS) $(LIBS)
