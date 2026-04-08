# C runtime makefile

CC = gcc
CFLAGS = -Wall -Wextra -O2 -I.
TARGET = yolo_npu_sim
SRCS = main.c npu_runtime.c npu_data.c
OBJS = $(SRCS:.c=.o)

all: $(TARGET)

run:
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $(OBJS)
	./$(TARGET)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean