# import the necessary packages   0.23.1
import dill
import pandas as pd
import os
dill._dill._reverse_typemap['ClassType'] = type
import flask
import logging
from logging.handlers import RotatingFileHandler
from time import strftime

# initialize our Flask application and the model
app = flask.Flask(__name__)
model = None

handler = RotatingFileHandler(filename='app.log', maxBytes=100000, backupCount=10)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

def load_model(model_path):
	# load the pre-trained model
	global model
	with open(model_path, 'rb') as f:
		model = dill.load(f)
	print(model)

modelpath = "./models/logreg_model.dill"
load_model(modelpath)

@app.route("/", methods=["GET"])
def general():
	return """Welcome to toxic comments prediction process. Please use 'http://<address>/predict' to POST"""

@app.route("/predict", methods=["POST"])
def predict():
	# initialize the data dictionary that will be returned
	data = {"success": False}
	dt = strftime("[%Y-%b-%d %H:%M:%S]")

	# ensure an post request was called
	if flask.request.method == "POST":

		comment_text = ""
		request_json = flask.request.get_json()
		print(request_json)
		if request_json['comment']:
			comment_text = request_json['comment']
		logger.info(f'{dt} Data: comment={comment_text}')

		try:
			data_in_pd = pd.DataFrame({"comment": [comment_text]})
			preds = model.predict(data_in_pd)
			preds_proba = model.predict_proba(data_in_pd)
			proba_threshold = model.get_probability_threshold()
		except AttributeError as e:
			logger.warning(f'{dt} Exception: {str(e)}')
			data['predictions'] = str(e)
			data['success'] = False
			return flask.jsonify(data)

		data["prediction"] = int(preds[0])
		data["prediction_probability"] = float(preds_proba[0])
		data["probability_threshold"] = float(proba_threshold)
		# indicate that the request was a success
		data["success"] = True

	# return the data dictionary as a JSON response
	return flask.jsonify(data)

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
	print(("* Loading the model and Flask starting server..."
		"please wait until server has fully started"))
	port = int(os.environ.get('PORT', 8180))
	app.run(host='0.0.0.0', debug=True, port=port)
