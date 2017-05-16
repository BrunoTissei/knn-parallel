CC := g++
SRCDIR := src
BUILDDIR := obj
TARGET := knn

SOURCES := $(shell find $(SRCDIR) -type f -name *.cpp)
OBJECTS := $(patsubst $(SRCDIR)/%, $(BUILDDIR)/%, $(SOURCES:.cpp=.o))

CFLAGS := -g -O3 -mavx -Wall -Wextra -fopenmp -std=c++11 -Isrc
LIB := -fopenmp

$(TARGET): $(OBJECTS)
	$(CC) $^ -o $(TARGET) $(LIB)

$(BUILDDIR)/%.o: $(SRCDIR)/%.cpp
	@mkdir -p $(BUILDDIR)
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	$(RM) -r $(BUILDDIR) $(TARGET)
