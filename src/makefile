CC = gcc
CFLAGS = -Wall -O2 -pg -no-pie
LDFLAGS = -lm -pg -no-pie

EXE = nn.exe
SRC = nn.c

all: $(EXE)

$(EXE): $(SRC)
	$(CC) $(CFLAGS) -o $(EXE) $(SRC) $(LDFLAGS)

run: $(EXE)
	./$(EXE)

profile: $(EXE) run
	gprof $(EXE) gmon.out > profile_report.txt
	cat profile_report.txt | less

clean:
	rm -f $(EXE) gmon.out profile_report.txt
