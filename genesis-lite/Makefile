CC=g++
CFLAGS=-I./main -Wall -O3
TARGET=genesis_lite
SRCS=main/main.cpp

all: $(TARGET)

$(TARGET): $(SRCS) main/genesis_core.hpp
	$(CC) $(CFLAGS) $(SRCS) -o $(TARGET)

clean:
	rm -f $(TARGET)

run: all
	./$(TARGET)
