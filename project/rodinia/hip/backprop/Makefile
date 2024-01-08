# C compiler
CC = gcc
CC_FLAGS = -g  -O2

# HIP compiler
HIPCC = hipcc
HIPCC_FLAGS = --offload-arch=gfx90a

HIPLIB_DIR = /opt/rocm-5.0.2/hip/lib

backprop: backprop.o facetrain.o imagenet.o backprop_hip.o 
	$(HIPCC) $(HIPCC_FLAGS) backprop.o facetrain.o imagenet.o backprop_hip.o -o backprop 

%.o: %.[ch]
	$(CC) $(CC_FLAGS) $< -c

facetrain.o: facetrain.c backprop.h
	$(CC) $(CC_FLAGS) facetrain.c -c
	
backprop.o: backprop.c backprop.h
	$(CC) $(CC_FLAGS) backprop.c -c

backprop_hip.o: backprop_hip.cpp backprop.h
	hipcc --offload-arch=gfx90a -c -o backprop_hip.o backprop_hip.cpp
	#$(HIPCC) $(HIPCC_FLAGS) -c backprop_hip.cpp

imagenet.o: imagenet.c backprop.h
	$(CC) $(CC_FLAGS) imagenet.c -c

clean:
	rm -f *.o backprop
