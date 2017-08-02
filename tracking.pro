TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt
CONFIG += link_pkgconfig

LIBS += -I /usr/include/eigen3/
PKGCONFIG += opencv
SOURCES += main.cpp
