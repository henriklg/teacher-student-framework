#!/usr/bin/python3

# Hvis du kjører 'pip install timecode" først
# Example "python kapseltid.py 00:00:00:28 00:00:32:24" etterpå for å finne riktig tidskode i taggeverktøyet

from timecode import *
import sys

def calc_time(start_time, capsule_time):
    start_time = '00:'+start_time
    capsule_time = '00:'+capsule_time
    
    s1 = Timecode(60, start_time)
    c = Timecode(60, capsule_time)

    v = c - s1

    frames = v.frames*2
    tc = Timecode(30, frames=frames)

    # print("Frames: %d, Timecode: %s" % (frames, tc))
    return int(frames)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Kjoring: %s start_tid kapsel_tid" % (sys.argv[0]))
        print("start_tid er tid til forste screenshot.")
        print("Eksempel: %s 00:00:00:28 00:00:03:27" % (sys.argv[0]))
        sys.exit(1)
        
    print (calc_time(sys.argv[1], sys.argv[2]))