from flask_restx import reqparse
from werkzeug.datastructures import FileStorage

encode_parser = reqparse.RequestParser()
encode_parser.add_argument('file', location='files', type=FileStorage, required=True, help='Original file to encode')
encode_parser.add_argument('depth', type=int, required=False, help='Depth for encoding', default=5)
encode_parser.add_argument('algorithm', type=str, required=False, help='Perceptual algorithm to use', default='pdq')
encode_parser.add_argument('resize', type=str, required=False, help='Resize dimensions (comma-separated)', default=None)
encode_parser.add_argument('crop', type=str, required=False, help='Crop coordinates (comma-separated)', default=None)
encode_parser.add_argument('note', type=str, required=False, help='Comment or note', default="Need a meaningful comment in here at some point")

compare_parser = reqparse.RequestParser()
compare_parser.add_argument('original_image_qt', location='files', type=FileStorage, required=True, help='Original QT image to be compared')
compare_parser.add_argument('new_image', location='files', type=FileStorage, required=True, help='New image to compare')
compare_parser.add_argument('threshold', type=int, required=False, help='Threshold for comparison', default=10)
compare_parser.add_argument('compare_depth', type=int, required=False, help='Depth for comparison', default=5)

search_parser = reqparse.RequestParser()
search_parser.add_argument('image', location='files', type=FileStorage, required=True, help='Image to search')