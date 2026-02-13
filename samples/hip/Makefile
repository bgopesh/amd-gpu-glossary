HIPCC = hipcc
CXXFLAGS = -O2 -g

TARGETS = vector_add matrix_multiply

all: $(TARGETS)

vector_add: vector_add.cpp
	$(HIPCC) $(CXXFLAGS) -o $@ $<

matrix_multiply: matrix_multiply.cpp
	$(HIPCC) $(CXXFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)

.PHONY: all clean
