import os
import numpy as np
from flask import Flask, request, current_app, send_file, jsonify, send_from_directory
from flask_restx import Api, Resource
from werkzeug.utils import secure_filename
from parsers.parsers import encode_parser, compare_parser, search_parser
from routes.compare import compare_image
from routes.encode import encode_image

app = Flask(__name__)
upload_folder = './upload'
app.config['UPLOAD_FOLDER'] = upload_folder

output_folder = './output'
app.config['OUTPUT_FOLDER'] = output_folder

if not os.path.exists(upload_folder):
    os.makedirs(upload_folder)

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

api = Api(app, version='1.0', title='HOPrS', description='Open standard for content authentication')
ns = api.namespace('hoprs', description='Human oriented proof standard')

@app.route('/output/<path:folder>/<path:filename>')
def get_file(folder, filename):
    return send_from_directory(os.path.join(app.config['OUTPUT_FOLDER'], folder), filename)

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

        filename = secure_filename(file.filename)
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        if resize:
            resize = tuple(map(int, resize.split(',')))

        if crop:
            crop = tuple(map(int, crop.split(',')))

        try:
            response = encode_image(filepath, depth, algorithm, resize, crop, note)
            return send_file(response, as_attachment=True)

        except Exception as e:
            return str(e), 500

@ns.route('/compare')
@ns.response(404, 'Task not found')
class Compare(Resource):
    @api.expect(compare_parser)
    def post(self):
        args = compare_parser.parse_args()
        original_image_qt_file = args['original_image_qt']
        new_image_file = args['new_image']

        if original_image_qt_file.filename == '' or new_image_file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        original_image_qt_filename = secure_filename(original_image_qt_file.filename)
        new_image_filename = secure_filename(new_image_file.filename)
        original_image_qt_filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], original_image_qt_filename)
        new_image_filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], new_image_filename)

        original_image_qt_file.save(original_image_qt_filepath)
        new_image_file.save(new_image_filepath)

        threshold = int(request.form.get('threshold', 10))
        compare_depth = int(request.form.get('compare_depth', 5))
        output_folder = os.path.join(current_app.config['OUTPUT_FOLDER'], os.path.splitext(new_image_filename)[0])
        os.makedirs(output_folder, exist_ok=True)

        original_image = new_image_filepath  # Reuse the uploaded new image as the base image
        new_image_qt_filepath = new_image_filepath + ".qt"

        try:
            response_data = compare_image(original_image, original_image_qt_filepath, new_image_filepath, new_image_filename, new_image_qt_filepath, threshold, compare_depth, output_folder)
            return response_data

        except Exception as e:
            return jsonify({"error": str(e)}), 500

@ns.route('/search')
@ns.response(404, 'Task not found')
class Search(Resource):
     @api.expect(search_parser)
     def post(self):
        args = search_parser.parse_args()
        image = args['image']

        if image.filename == '':
            return jsonify({"error": "No selected file"}), 400
        
        try:
            return
        except Exception as e:
            return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
