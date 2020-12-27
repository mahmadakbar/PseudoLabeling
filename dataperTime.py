import numpy as np
import pandas as pd
import sched
import time
import subprocess
import sys

from subprocess import Popen, PIPE, STDOUT
from twisted.internet import task, reactor

timeout = 10.0 # one seconds

def do_something():
    ts = time.time()

    one = (ts)*1000
    time1 = int(one)

    two = (ts-10)*1000
    time2 = int(two)

    # print "time now = %s , and time 5 second pass = %s" % (one, two)
    print "time 10 second pass = %s , and time now = %s" % (time2, time1)
    subprocess.call(["mongoexport",
                     "--db",
                     "mnemosyne",
                     "--collection",
                     "session",
                     "--type=csv",
                     "--fields=_id,protocol,hpfeed_id,timestamp,source_ip,destination_ip,identifier,honeypot",
                     "-q", """{"timestamp":{$gt:new Date(%s),$lt:new Date(%s)}, "protocol" : "ICMP"}""" % (time2, time1),
                     "--out",
                     "output/%s.csv" % time1])
    pass

l = task.LoopingCall(do_something)
l.start(timeout) # call every ten seconds

reactor.run()
