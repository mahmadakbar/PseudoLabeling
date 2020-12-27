# -*- coding: utf-8 -*-

import requests
import json

import time
import zmq

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")

def getRyu(whoRyu):
	#get DPID from switches
	mapping = {}
	a = requests.get('http://192.168.3.25:8080/stats/switches')
	# print(a.json())
	switches = a.json()

	#get port description
	for i in switches:
		# print(i)
		command = 'http://192.168.3.25:8080/stats/portdesc/' + str(i)
		r = requests.get(command)
		temp = r.json()[str(i)]
		ports = []
		for b in temp:
			if b['port_no'] != 'LOCAL':
				ports.append(b['port_no'])
				# print("DPID:"+str(i)+";Port:"+str(b['port_no']))
		mapping[i] = ports

	print(mapping)

	#mitigating - Flow Rule
	for keys, values in mapping.items():
		for a in values:
			send = requests.post('http://192.168.3.25:8080/stats/flowentry/add', json={\
			"dpid": keys,\
			"cookie": 0,\
			"table_id": 0,\
			"hard_timeout": 60,\
			"priority": 11111,\
			"flags": 1,\
			"match":{"in_port": a,"eth_type": 0x0800,"ip_proto": 1},\
			"actions":[]\
			})
			print(send.status_code)

while True:
    #  Wait for next request from client
    message = socket.recv()
    # print("Received request: %s" % message)
    if message == "Dattaa":

		# getRyu()
		print "I revceived a message"

		socket.send(b"Done")


# curl -X POST -d '{
#     "dpid": 1,
#     "cookie": 0,
#     "table_id": 0,
#     "priority": 100,
#     "flags": 1,
#     "match":{
#         "in_port": 1,
# 		"eth_type": 0x0800,
# 		"ip_proto": 1
#     },
#     "actions":[
#     ]
#  }' http://localhost:8080/stats/flowentry/add
