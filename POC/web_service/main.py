from flask import Flask
import os
import logging
from routes import index_bp, compare_bp, upload_bp, search_bp, quadtrees_bp
from logging.handlers import RotatingFileHandler


# If using Google Cloud Logging
if os.getenv('GAE_ENV', '').startswith('standard'):
    import google.cloud.logging
    google_cloud_client = google.cloud.logging.Client()
    google_cloud_client.setup_logging()


def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)#Cloud logging level

    # Create console handler with a higher log level
    c_handler = logging.StreamHandler()
    c_handler.setLevel(logging.DEBUG)

    # Create formatters and add them to handlers
    c_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(c_format)

    # Add handlers to the logger
    logger.addHandler(c_handler)

    # Only add file handler if not running on Google Cloud
    if not os.getenv('GAE_ENV', '').startswith('standard'):
        # Ensure log folder exists
        log_folder = os.path.join(os.path.dirname(__file__), 'logs')
        os.makedirs(log_folder, exist_ok=True)

        # Configure file handler with rotation
        log_file = os.path.join(log_folder, 'flask_app.log')
        f_handler = RotatingFileHandler(log_file, maxBytes=10000000, backupCount=5)  # Rotate after 10MB, keep 5 backups
        f_handler.setLevel(logging.DEBUG)
        f_handler.setFormatter(c_format)
        logger.addHandler(f_handler)

setup_logging()

print("Starting app.py")
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/tmp/uploads'
app.config['OUTPUT_FOLDER'] = '/tmp/output'
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

print("finished app config")
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
print("have created directories")
# Register blueprints
print("About to start registering blueprints")

app.register_blueprint(index_bp)
print("index_bp registered in main")
app.register_blueprint(compare_bp)
print("compare_bp registered in main")
app.register_blueprint(upload_bp)
app.register_blueprint(search_bp)
app.register_blueprint(quadtrees_bp)
print("all blueprints registered in main")

if __name__ == "__main__":
    app.run(debug=True)
