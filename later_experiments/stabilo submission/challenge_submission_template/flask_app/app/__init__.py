from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
CORS(app) # enables openapi "try it now"

from app import views
