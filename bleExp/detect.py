# Adam Rauff
# 6/19/2017
# Smart Door Project

import bluetooth
import time

BleAddr = []
# read list provided in DevList and store in python list

while True:
	print("Checking " + time.strftime("%a, %d %b %Y %H:%M:%S", time.gmtime()))
	result = bluetooth.lookup_name('',timeout=5)


	if (result != None):
		print('User Present')
	else:
		print('Out of range')

	time.sleep(7) 


