from flask import Blueprint
print("Starting routes/__init__.py")
# Initialize blueprints
index_bp = Blueprint('index_bp', __name__)
compare_bp = Blueprint('compare_bp', __name__)
upload_bp = Blueprint('upload_bp', __name__)
search_bp = Blueprint('search_bp', __name__)
quadtrees_bp = Blueprint('quadtrees_bp', __name__)
print("Have now initialized blueprints in routes/__init__.py")


# Import routes to register them with the blueprints
print("About to start importing all the routes in routes/__init__.py ")
from .index import *
from .compare import *
from .upload import *
from .search import *
from .quadtrees import *
