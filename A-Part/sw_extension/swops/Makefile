CC = swgcc
CFLAGS = -msimd -I./ -mieee -w -fPIC
LIBFLAGS = -mdynamic

LIB = ./libswops.a

ALL: ${LIB}  

master.o: master.c
	$(CC) $(CFLAGS) -O3 -mhost -c $^ -o $@

slave.o: slave.c
	$(CC) $(CFLAGS) -O3 -mslave -c $^ -o $@

$(LIB): master.o slave.o
	sw9ar rcs $@ $^

.PHONY: clean
clean:
	rm -rf ${LIB} ./*.o 
