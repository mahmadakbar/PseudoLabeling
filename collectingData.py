import numpy as np
import pandas as pd
import sched
import time
import subprocess
import sys

from subprocess import Popen, PIPE, STDOUT
from twisted.internet import task, reactor

def do_something():

    ts = time.time()

    subprocess.call(["mongoexport",
                     "--db",
                     "mnemosyne",
                     "--collection",
                     "session",
                     "--type=csv",
                     "--fields=_id,protocol,hpfeed_id,timestamp,source_ip,destination_ip,identifier,honeypot",
                     "-q", """{"protocol" : "ICMP"}""",
                     "--out",
                     "%s.csv" % ts])
    
    k1 = subprocess.Popen(["mongo","mnemosyne"], stdin=PIPE, stdout=PIPE)
    k1.communicate(input="db.session.remove({})")
    subprocess.Popen("exit 1", shell=True)
    k2 = subprocess.Popen(["mongo","mnemosyne"], stdin=PIPE, stdout=PIPE)
    k2.communicate(input="db.metadata.remove({})")
    subprocess.Popen("exit 1", shell=True)
    k3 = subprocess.Popen(["mongo","mnemosyne"], stdin=PIPE, stdout=PIPE)
    k3.communicate(input="db.counts.remove({})")
    subprocess.Popen("exit 1", shell=True)
    k4 = subprocess.Popen(["mongo","mnemosyne"], stdin=PIPE, stdout=PIPE)
    k4.communicate(input="db.file.remove({})")
    subprocess.Popen("exit 1", shell=True)
    k5 = subprocess.Popen(["mongo","mnemosyne"], stdin=PIPE, stdout=PIPE)
    k5.communicate(input="db.hpfeed.remove({})")
    subprocess.Popen("exit 1", shell=True)
    k6 = subprocess.Popen(["mongo","mnemosyne"], stdin=PIPE, stdout=PIPE)
    k6.communicate(input="db.dork.remove({})")
    subprocess.Popen("exit 1", shell=True)
    k7 = subprocess.Popen(["mongo","mnemosyne"], stdin=PIPE, stdout=PIPE)
    k7.communicate(input="db.url.remove({})")
    subprocess.Popen("exit 1", shell=True)
    k8 = subprocess.Popen(["mongo","mnemosyne"], stdin=PIPE, stdout=PIPE)
    k8.communicate(input="db.daily_stats.remove({})")
    subprocess.Popen("exit 1", shell=True)

    f = pd.read_csv('%s.csv' % ts)
    # print f

do_something()