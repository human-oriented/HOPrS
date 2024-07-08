from flask import Flask, render_template, request, redirect, url_for,Response
from flask.helpers import flash
from dotenv import load_dotenv
from io import StringIO

from astrapy.client import DataAPIClient
import astrapy
from utils import (
    collection, sort_csv_by_first_field)
from . import quadtrees_bp


# Route to display all QuadTrees
@quadtrees_bp.route('/quadtrees')
def list_quadtrees():
    unique_qt_refs = collection.distinct("qt_ref")
    
    qt_refs_with_counts = []
    for qt_ref in unique_qt_refs:
        count = collection.count_documents({"qt_ref": qt_ref}, upper_bound=50)
        qt_refs_with_counts.append((qt_ref, count))
    
    return render_template('list_quadtrees.html', qt_refs_with_counts=qt_refs_with_counts)

# Route to delete a QuadTree
@quadtrees_bp.route('/quadtrees/<qt_ref>/delete', methods=['POST'])
def delete_quadtrees(qt_ref):
    try:
        result = collection.delete_many({"qt_ref": qt_ref})
        return f"Deleted {result.deleted_count} records for qt_ref: {qt_ref}. <a href='/quadtrees'>Back to QuadTrees list</a>"
    except Exception as e:
        return f"Error deleting records: {str(e)}"
    
    
#Download from database a quadtree in CSV format
@quadtrees_bp.route('/quadtrees/<qt_ref>/download_csv', methods=['GET'])
def download_csv(qt_ref):
    print(f"download_csv called: {qt_ref}")
    # Retrieve all QuadTrees with the specified qt_ref from the database
    query = {
        "qt_ref": {"$eq": qt_ref}
    }
    csv_str = ""
    results = collection.find(query)
    for record in results:
        csv_str += f'{record["path"]},{record["level"]},{record["x0"]},{record["y0"]},{record["x1"]},{record["y1"]},{record["width"]},{record["height"]},{record["hash_algorithm"]},{record["perceptual_hash_hex"]},{record["quality"]}\n'

    sorted_csv_str = sort_csv_by_first_field(csv_str)
    
    # Set up response headers for CSV download
    response = Response(
        sorted_csv_str,
        mimetype='text/csv',
        headers={"Content-Disposition": f"attachment;filename={qt_ref}_quadtrees.csv"}
    )

    return response
    