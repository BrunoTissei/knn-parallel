import sys

input_file = open(sys.argv[1], "r")
training_file = open("data/training_set", "w")
testing_file = open("data/testing_set", "w")

n, d = map(int, input_file.readline().split())
training_size = int(n * 0.6)

training_file.write(str(training_size) + " " + str(d) + "\n")
for i in xrange(training_size):
    training_file.write(input_file.readline())

testing_file.write(str(n - training_size) + " " + str(d) + "\n")
for i in xrange(n - training_size):
    testing_file.write(input_file.readline())
