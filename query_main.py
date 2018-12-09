from tkinter import *
from os import system
import subprocess
import os
from datetime import date, datetime
from time import gmtime, strftime
import numpy as np
import msgpack
import msgpack_numpy as m
import pickle

m.patch()

#########################################################################
#                                                                       #
# Disclaimer: I am horribly awful at GUI programming and trying to do   #
# 			  anything involving networks. This code is riddled with    #
#			  bad practices and really hacky coding. Sorry. At least    #
#			  this comment is nicely formatted?						    #
#                                                                       #
#########################################################################

class Query(Frame):

	def __init__(self, master=None):
		Frame.__init__(self, master)
		self.grid()
		self.createWidgets()
		self.master.title("Connect to Kraken")
		self.master.maxsize(1200, 300)
		self.master.minsize(700, 150)
		self.message = Label(self.master)
		self.images = None

	def createWidgets(self):
		"""
		Populate the menu with buttons and fields
		"""

		self.username_label = Label(self.master, text="Username: ")
		self.username_label.grid(row = 0, column = 0, padx = (20, 0), pady = (20,10))

		self.username = Entry(self.master, width=12)
		self.username.delete(0, END)
		self.username.grid(row = 0, column = 1, padx = 0, pady = (20,10))

		self.password_label = Label(self.master, text="Password: ")
		self.password_label.grid(row = 1, column = 0, padx = (20, 0), pady = 0)

		self.password = Entry(self.master, show="*", width=12)
		self.password.delete(0, END)
		self.password.grid(row = 1, column = 1, padx = 0, pady = 0)

		self.connect_button = Button(self.master, text="Connect")
		self.connect_button["bg"]   = "green"
		self.connect_button["command"] = self.connect
		self.connect_button.grid(row = 99, column = 1, padx = 0, pady = (60,10))

		self.exit = Button(self.master, text="Exit")
		self.exit["bg"]   = "red"
		self.exit["command"] = self.quit
		self.exit.grid(row = 99, column = 0, padx = 0, pady = (60,10))

		self.screen = "connect"
		self.master.bind("<Return>", self.enter)

	def ssh_connection(self, file):
		"""
		Make a connection to kraken via ssh. Run the specified
		file on kraken if the connection is successful
		"""

		cmd = ("cat "+ file +" | sshpass -p "
			+self.pw+" ssh -q "
			+self.user+"@kraken.msu.montana.edu python3")
		success = system(cmd)

		if success==0:
			return True
		else:
			return False

	def enter(self, event):
		"""
		When the user presses enter, activate the appropriate button
		"""

		if self.screen == "connect":
			self.connect()
		elif self.screen == "submit":
			self.submit_query()

	def connect(self):
		"""
		Attempt to connect to Kraken using the user
		input username and password
		"""

		user = self.username.get()
		pw = self.password.get()
		msg = ""
		color = "red"
		attempt_connection = False

		if user == "" and pw == "":
			self.sad_message("Cannot connect: missing username and password!")

		elif user == "":
			self.sad_message("Cannot connect: missing username!")

		elif pw == "":
			self.sad_message("Cannot connect: missing password!")

		else:
			self.happy_message("Connecting...")
			attempt_connection = True

		run_db_command = False
		if attempt_connection:
			file = "test_connection.py"
			self.user = self.username.get()
			self.pw = self.password.get()
			run_db_command = self.ssh_connection(file)

			if run_db_command:
				self.happy_message("Connection successful.")
				self.query_menu()

			else:
				self.sad_message("Could not connect. Check username and password.")

	def clear_login(self):
		"""
		Remove the initial login widgets
		"""

		self.username_label.grid_forget()
		self.username.grid_forget()
		self.password_label.grid_forget()
		self.password.grid_forget()
		self.connect_button.grid_forget()
		self.exit.grid_forget()
		self.message.grid_forget()

	def query_menu(self):
		"""
		Clear the old menu, and populate the new menu with
		parameters for the database insertion
		"""

		self.clear_login()
		self.create_submission_menu()

	def create_submission_menu(self):
		"""
		Create a menu for submitting an image to the database
		with fields for the required parameters
		"""

		self.produce_label = Label(self.master, text='Produce Type: ')
		self.produce_label.grid(row=0, column=0)
		self.produce_entry = Entry(self.master)
		self.produce_entry.grid(row=0, column=1)

		self.age_label = Label(self.master, text='Days Old: ')
		self.age_label.grid(row=1, column=0)
		self.age_entry = Entry(self.master)
		self.age_entry.grid(row=1, column=1)

		self.days_left_label = Label(self.master, text='Days Until Bad: ')
		self.days_left_label.grid(row=2, column=0)
		self.days_left_entry = Entry(self.master)
		self.days_left_entry.grid(row=2, column=1)

		'''self.condition_label = Label(master, text='Condition: ')
		self.condition_label.grid(row=3, column=0)
		self.condition_entry = Entry(master)
		self.condition_entry.grid(row=3, column=1)'''

		self.cond = StringVar(self.master)
		self.cond.set('fresh')
		conditions = ['fresh', 'discount', 'old']
		self.condition_menu = OptionMenu(self.master, self.cond, *conditions)
		self.condition_menu.grid(row=3, column=0)

		self.exit_button = Button(self.master, fg='red', text="Exit", command=self.master.quit)
		self.exit_button.grid(row=4, column=0)

		self.submit_button = Button(self.master, fg = 'green', text="Submit", command=self.submit_query)
		self.submit_button.grid(row=4, column=1)

		self.screen = "submit"

	def happy_message(self, msg):
		"""
		Write a green message to the tkinter window
		"""

		self.message["text"] = msg
		self.message["fg"] = "green"
		self.message.grid(row = 99, column = 4, padx = 0, pady = (60,10), sticky=W)
		self.master.update()

	def sad_message(self, msg):
		"""
		Write a red message to the tkinter window
		"""

		self.message["text"] = msg
		self.message["fg"] = "red"
		self.message.grid(row = 99, column = 4, padx = 0, pady = (60,10), sticky=W)
		self.master.update()

	def submit_query(self):
		"""
		Submit the image to the database
		"""

		produce = self.produce_entry.get().lower().strip()
		if self.age_entry.get() != '':
			age = int(self.age_entry.get().strip())
		else:
			age = -999
		if self.days_left_entry.get() != '':
			days_left = int(self.days_left_entry.get().strip())
		else:
			days_left = -999
		condition = self.cond.get().strip().lower()

		if produce == "":
			self.sad_message("Cannot submit: no produce type listed.")
		else:
			#try:
			#Need to add .pgpass file to user's home directory on
			#Kraken with entry formatted as:
			#localhost:5432:produce:<username>:<password>
			#Then chmod 600 .pgpass
			cmd = ("sshpass -p "+self.pw+" ssh -q "
				+self.user+"@kraken.msu.montana.edu \"psql -U "+self.user
				+ " -w -d produce -c 'SELECT * FROM produce p join"
				+ " produce_properties pp ON p.id=pp.id'\"")

			a=subprocess.check_output(cmd, shell=True)
			a = str(a)
			i = 1
			x = 0
			raw_query = []
			while True:

				flag = True
				start = ' ' + str(i) + ' | {'
				x = a.find(start)
				end = ' ' + str(i+1) + ' | {'
				y = a.find(end)
				#if x == -1 and y == -1:
				#	break
				if x == -1 and y != -1: #deal with deleted indices
					i+=1
					continue
				elif y == -1: #deal with deleted indices
					flag = False
					for j in range(2,20): #if more than 20 consecutive indices missing, assume we're at end
						new_ind = i+j
						test = ' ' + str(new_ind) + ' | {'
						test_result = a.find(test)
						if test_result != -1:
							flag = True
							y = test_result
							i = new_ind-1
							break

				if not flag:
					raw_query.append(a[x:y].split('|'))
					break

				raw_query.append(a[x:y].split('|'))
				i+=1

			reduced_query = []
			for entry in raw_query:
				sub_entry = []
				#print ('--------------------')
				if (len(entry) > 1):
					sub_entry.append(entry[1].strip().replace('{', '').replace('}', '').split(',')) #Produce list
					sub_entry.append(entry[2].strip()) #Condition
					sub_entry.append(int(entry[6].strip())) #Age in days
					sub_entry.append(int(entry[7].strip())) #Days until old
					sub_entry.append(entry[4].strip()) #File path
					reduced_query.append(sub_entry)

			filtered_query = []
			for result in reduced_query:
				if produce in result[0] and len(result[0])==1:
					if condition==result[1]:
						if age==result[2] or age==-999:
							if days_left==result[3] or days_left==-999:
								filtered_query.append(result[4])

			images = []
			for i, file_path in enumerate(filtered_query):
				self.happy_message('Fetching image ' + str(i+1) + ' of ' + str(len(filtered_query)) + '...')
				cwd = os.getcwd()
				cmd = ("sshpass -p "+self.pw+" scp "
					+self.user+"@kraken.msu.montana.edu:"+file_path+" "+cwd)

				os.system(cmd)

				self.happy_message('Unpacking image ' + str(i+1) + ' of ' + str(len(filtered_query)) + '...')
				file = file_path[file_path.rfind('/')+1:] #Extact the file name from path
				o = open(file, 'rb')
				#self.happy_message("Unpacking")
				unpacker = msgpack.Unpacker(file_like = o)
				x_rec = unpacker.unpack()
				images.append(x_rec.T)
				#self.happy_message("Done.")
				os.system('rm '+ file)
			self.happy_message('Done fetching images! Close window to return to program.')

			#except:
			#	self.sad_message("Something went wrong. Query not executed.")

			self.images = images
			save_path = "produce.p"
			with open(os.path.join(os.getcwd(),save_path), 'wb') as f:
				pickle.dump(self.images, f, protocol=pickle.HIGHEST_PROTOCOL)


def get_images():
	root = Tk()
	menu = Query(master=root)
	menu.mainloop()
	return menu.images

if __name__=='__main__':
	root = Tk()
	menu = Query(master=root)
	menu.mainloop()

	#Closing the window using the 'x' button already calls destroy()
	try:
		root.destroy()
	except TclError:
		pass
