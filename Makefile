CC=g++
CFLAGS=-g3
IFLAGS=-isystem /usr/local/systemc-2.3.2/include
LFLAGS=-lsystemc -lm -L/usr/local/systemc-2.3.2/lib-linux64
DEPS=system.h
OBJS=system.o

%.o: %.cpp $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS) $(IFLAGS)

sad: $(OBJS)
	$(CC) -o $@ $^ $(CFLAGS) $(LFLAGS)

.PHONY: clean

clean:
	rm -f $(OBJS) sad
