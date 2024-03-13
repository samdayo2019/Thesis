CC=g++
HL=i++
CFLAGS=-g3
IFLAGS=-isystem /usr/local/systemc-2.3.2/include
LFLAGS=-lsystemc -lm -L/usr/local/systemc-2.3.2/lib-linux64
DEPS=system.h
DEVICE=Arria10
FILE=system

all: 

gpp: $(FILE).cpp $(DEPS)
	$(CC) $(CFLAGS) $(IFLAGS) -o $(FILE).out

emu: $(FILE).cpp $(DEPS)
	$(HL) $(CFLAGS) $(LFLAGS) $(FILE).cpp -o $(FILE)_emu.out

fpga: $(FILE).cpp $(DEPS)
	$(HL) $(CFLAGS) $(LFLAGS) $(FILE).cpp -o $(FILE)_fpga.out -march=$ (DEVICE)

