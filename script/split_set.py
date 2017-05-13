import sys
import random

input_file = open(sys.argv[1], "r")
shuff = False
if len(sys.argv) > 2 and sys.argv[2] == "--random":
    shuff = True

training_file = open("data/training_set", "w")
testing_file = open("data/testing_set", "w")

n, d = map(int, input_file.readline().split())

lines = []
for i in xrange(n):
    lines.append(input_file.readline())

if shuff:
    random.shuffle(lines)

training_size = int(n * 0.6)
testing_size = n - training_size

training_file.write(str(training_size) + " " + str(d) + "\n")
for i in xrange(0, training_size):
    training_file.write(lines[i])

testing_file.write(str(testing_size) + " " + str(d) + "\n")
for i in xrange(training_size, training_size + testing_size):
    testing_file.write(lines[i])
