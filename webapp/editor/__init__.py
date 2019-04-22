import os

from flask import Flask
from flask import (
    render_template
)
from flask_cors import CORS

def create_app(test_config=None):
    
    app = Flask(__name__)
    CORS(app)

    @app.route("/")
    def home():
        return render_template("index.html")


    from . import api   
    app.register_blueprint(api.bp)

    return app