# Plain Makefile for machines without CMake. CMakeLists.txt builds the same targets.
CXX ?= c++
CXXFLAGS ?= -O2 -g
CXXFLAGS += -std=c++17 -Wall -Wextra -Wpedantic -Iinclude

BUILD := build
LIB_SRC := src/config.cpp src/decomposition.cpp src/domain.cpp src/particles.cpp src/stepper.cpp src/diagnostics.cpp src/wind.cpp src/urban.cpp
LIB_OBJ := $(LIB_SRC:src/%.cpp=$(BUILD)/obj/%.o)
HEADERS := $(wildcard include/lps/*.hpp)

all: $(BUILD)/lagrangian_solver $(BUILD)/lps_unit_tests

$(BUILD)/obj/%.o: src/%.cpp $(HEADERS)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD)/lagrangian_solver: $(BUILD)/obj/main.o $(LIB_OBJ)
	$(CXX) $(CXXFLAGS) $^ -o $@

$(BUILD)/lps_unit_tests: tests/unit_tests.cpp $(LIB_OBJ) $(HEADERS)
	$(CXX) $(CXXFLAGS) tests/unit_tests.cpp $(LIB_OBJ) -o $@

test: $(BUILD)/lps_unit_tests $(BUILD)/lps_urban_tests $(BUILD)/lps_decomposition_tests
	./$(BUILD)/lps_unit_tests
	./$(BUILD)/lps_urban_tests $(BUILD)/urban-fixture
	./$(BUILD)/lps_decomposition_tests

validate: all
	python3 scripts/validate.py --exe $(BUILD)/lagrangian_solver

clean:
	rm -rf $(BUILD)

.PHONY: all test validate clean

$(BUILD)/lps_decomposition_tests: tests/decomposition_tests.cpp $(LIB_OBJ) $(HEADERS)
	$(CXX) $(CXXFLAGS) tests/decomposition_tests.cpp $(LIB_OBJ) -o $@

$(BUILD)/lps_urban_tests: tests/urban_tests.cpp $(LIB_OBJ) $(HEADERS)
	$(CXX) $(CXXFLAGS) tests/urban_tests.cpp $(LIB_OBJ) -o $@

