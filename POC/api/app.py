import os
import cv2
from flask import Flask, request, current_app, send_file, jsonify, send_from_directory
from flask_restx import Api, Resource
from werkzeug.utils import secure_filename
from parsers.parsers import encode_parser, compare_parser, search_parser, download_parser
from routes.compare import compare_image
from routes.encode import encode_image
from routes.search import search_images
from routes.download import download_qt
from routes.count import count_database
from routes.list import list_database
from utils.utils import (validate_image, validate_quadtree, convert_heic)

app = Flask(__name__)

app.config['DEBUG'] = True


upload_folder = './upload'
app.config['UPLOAD_FOLDER'] = upload_folder

output_folder = './output'
app.config['OUTPUT_FOLDER'] = output_folder

if not os.path.exists(upload_folder):
    os.makedirs(upload_folder)

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

api = Api(app, version='0.0.9', title='HOPrS', description='Open standard for content authentication')
ns = api.namespace('hoprs', description='Human oriented proof standard')

@ns.route('/download/')
class Download(Resource):
    @api.expect(download_parser)
    def post(self):
        args = download_parser.parse_args()
        qt_ref = args['qt_ref']
        if (qt_ref == ''):
            return "No qt_ref specified", 400
        return download_qt(qt_ref)

# Serves files from output folder
@app.route('/output/<path:folder>/<path:filename>')
def get_file(folder, filename):
    if folder is None or folder == "":
        return send_from_directory(app.config['OUTPUT_FOLDER'], filename)
    else:
        return send_from_directory(os.path.join(app.config['OUTPUT_FOLDER'], folder), filename)

# Returns version number
@ns.route('/version')
@ns.response(404, 'Task not found')
class Version(Resource):
    def get(self):
        return api.version

# Encode a file
# Return - a quad tree file to download 
@ns.route('/encode')
class Encode(Resource):
    @api.expect(encode_parser)
    def post(self):
        args = encode_parser.parse_args()
        file = args['file']
        depth = args['depth']
        algorithm = args['algorithm']
        resize = args['resize']
        crop = args['crop']
        note = args['note']

        if file.filename == '':
            return "No selected file", 400
        
        if depth < 0 or depth > 6:
            return "Compare depth must be between 1 and 6.", 400
        
        valid, message = validate_image(file)
        if not valid:
            return message

        filename = secure_filename(file.filename)
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        if resize:
            resize = tuple(map(int, resize.split(',')))

        if crop:
            crop = tuple(map(int, crop.split(',')))

        try:
            response = encode_image(filepath, depth, algorithm, resize, crop, note, file)
            return send_file(response, as_attachment=True)

        except Exception as e:
            return str(e), 500

# Compares an uploaded image to a quad tree file
# Returns json of images of similarity and stats
@ns.route('/compare')
@ns.response(404, 'Task not found')
class Compare(Resource):
    @api.expect(compare_parser)
    def post(self):
        args = compare_parser.parse_args()
        original_image_qt_file = args['original_image_qt']
        new_image_file = args['new_image']
        threshold = args['threshold']
        compare_depth = args['compare_depth']

        if threshold < 0 or threshold > 11:
            return "Threshold must be between 1 and 10.", 400
        
        if compare_depth < 0 or compare_depth > 6:
            return "Compare depth must be between 1 and 6.", 400

        if original_image_qt_file.filename == '' or new_image_file.filename == '':
            return "No selected file", 400
        
        valid, message = validate_image(new_image_file)
        if not valid:
            return message
        
        qt_valid, qt_message = validate_quadtree(original_image_qt_file)
        if not qt_valid:
            return qt_message

        original_image_qt_filename = secure_filename(original_image_qt_file.filename)
        new_image_filename = secure_filename(new_image_file.filename)
        original_image_qt_filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], original_image_qt_filename)
        new_image_filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], new_image_filename)

        root, ext = os.path.splitext(new_image_file.filename.lower())
        if ext == '.heic':
            image_array = convert_heic(new_image_file)
            current_app.logger.debug(image_array)
            new_image_filepath = new_image_filepath + '.png'
            new_image_filename = new_image_filename + '.png'
            img_cv = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            cv2.imwrite(new_image_filepath, img_cv)
        else: 
            new_image_file.save(new_image_filepath)

        original_image_qt_file.save(original_image_qt_filepath)
        output_folder = os.path.join(current_app.config['OUTPUT_FOLDER'], os.path.splitext(new_image_filename)[0])
        os.makedirs(output_folder, exist_ok=True)
        try:
            response_data = compare_image(original_image_qt_filepath, new_image_filepath, threshold, compare_depth, output_folder)
            return response_data

        except Exception as e:
            return e.message, 500

#BAsic db admin - counts the umber of records int eh DB and number of unique image_ids (quadtrees)
@ns.route('/count')
@ns.response(404, 'Task not found')
class Count_Database(Resource): 
    def get(self):
        return count_database()
    
@ns.route('/list')
@ns.response(404, 'Task not found')
class List_Database(Resource): 
    def get(self):
        return list_database()

# Searches if there is a particular image in our database
# Returns closest matches if any
@ns.route('/search')
@ns.response(404, 'Task not found')
class Search(Resource):
     @api.expect(search_parser)
     def post(self):
        args = search_parser.parse_args()
        image = args['image']
        threshold = args['threshold']
        compare_depth = args['compare_depth']

        if image.filename == '':
            return "No selected file"
        if threshold < 0 or threshold > 100:
            return "Threshold must be between 1 and 100.", 400
        if compare_depth < 1 or compare_depth > 10:
            return "Compare depth must be between 1 and 10.", 400
        try:
            valid, message = validate_image(image)
            if valid:
                result = search_images(image, threshold, compare_depth)
                return result
            else:
                return message
        except Exception as e:
            return str(e)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
