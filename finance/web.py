from flask import Flask, send_file
from werkzeug.contrib.cache import SimpleCache
from vis import vis
from io import BytesIO

app = Flask(__name__)
cache = SimpleCache()

@app.route("/")
def index(): return "index"

@app.route("/p/<ticker>")
def pics(ticker):
	#return send_file(vis(ticker), mimetype='image/png')
	#return send_file(BytesIO(vis(ticker)), mimetype='image/png')
	rv = cache.get(ticker)
	if rv: print 'cached'
	if rv is None:
		rv = BytesIO(vis(ticker))
		cache.set(ticker, rv, timeout=5 * 60)
	return send_file(rv, mimetype='image/png')

if __name__ == "__main__":
	app.run(debug='True')