import argparse
import struct
import numpy as np

class Match(object):
    def __init__(self, first, last, score, correct=None):
        self.first = first
        self.last = last
        self.score = score
        self.correct = correct

def create_ros_curve(predicted_matches, true_matches):
    tresholds = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900]
    predicted = []
    b = None
    with open(predicted_matches, "rb") as f:
        counter = 0
        b = f.read()

    with open(true_matches) as f:
        true_lines = f.readlines()
    true_matches = []
    last_id = None
    first = None
    last = None
    for i, line in enumerate(true_lines):
        id = int(line.split(" ")[0])
        if last_id == None:
            last_id = id
            first = i
        if last_id != id:
            true_matches.append((first, i-1))
            first = i
            last_id = id

    min_score = 10000
    max_score = 0


    tp = [0]*len(tresholds)
    tn = [0]*len(tresholds)
    fp = [0]*len(tresholds)
    fn = [0]*len(tresholds)
    for i in range(0, len(b), 12):
        f, s, score = struct.unpack("iif", b[i:i+12])
        true_match = False

        if (s >= true_matches[f][0] and s <= true_matches[f][1]): true_match = True
        for j, treshold in enumerate(tresholds):
            if score < treshold and true_match: tp[j] += 1
            if score < treshold and not true_match: fp[j] += 1
            if score > treshold and true_match: fn[j] += 1
            if score > treshold and not true_match: tn[j] += 1

        if (score > max_score): max_score = score
        if (score < min_score): min_score = score
        if (i % 1200000  == 0):
            print (i, "/", len(b))

    
    cmf = []
    imf = []

    for tp_, tn_, fp_, fn_ in zip(tp, tn, fp, fn):
        cmf.append(tp_/(tp_+fp_))
        imf.append(fp_/(tn_+fp_))
    
    print (cmf)
    print (imf)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--predicted-matches', required=True)
    parser.add_argument('-t', '--true-matches', required=True)
    args = parser.parse_args()
    create_ros_curve(**args.__dict__)
