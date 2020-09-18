import os
from flask import Flask, render_template,request
app = Flask(__name__)

from temp import start_cam

@app.route('/',methods=['GET','POST'])
def hello_world():
	if request.method=='GET':
		return render_template('index.html')
	if request.method=='POST':
		try:
			start_cam()

		except Exception as e:
			print(e)
		

    

if __name__ == '__main__':
	app.run(debug=True,port=os.getenv('PORT',5000))