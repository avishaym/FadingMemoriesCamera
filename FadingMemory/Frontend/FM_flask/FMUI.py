from flask import Flask, escape, request

app = Flask(__name__)

@app.route('/')
def hello():
    return "<h1>Home Page</h1>"

@app.route('/about')
def about():
    return "<h1>About Page</h1>"

@app.route('/welcome')
def welcome():
    return "<h1>About Page</h1>"

@app.route('/approval')
def approval():
    return "<h1>Please approve saving you picture </h1>"

@app.route('/instructions')
def instructions():
    return "<h1>Instructions: 1. hit the button below 2. you have 10 seconds to stand infront of the camera 3. wait there for 5 seconds and come back to see the results 4. if you'd like - you can hit the 'mail it' button below the picture </h1>"

app.route('/displayimage')
def displayimage():
    return "<h1>here it is </h1>"

app.route('/mailittome')
def mailittome():
    return "<h1>Please provide your </h1>"

@app.route('/gallery')
def gallery():
    return "<h1>Image Slider</h1>"

if __name__ == '__name__':
	app.run(debug=True)
