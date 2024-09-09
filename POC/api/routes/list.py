#This is a utility that will list some basic stats. 
    
from utils.utils import (
    session, ASTRA_DB_KEYSPACE, ASTRA_DB_TABLE
)
from flask import jsonify
import json

def list_database():
    #Number of records. 

    try:
        query = "SELECT DISTINCT image_id FROM " + ASTRA_DB_KEYSPACE + "."+ ASTRA_DB_TABLE + " ;"    
        prepared_stmt = session.prepare(query)
        results = session.execute(prepared_stmt,[] )
        output_list = []
        for doc in results:
            print(f"image_id: {doc[0]}")
            individualquery = "SELECT image_id, misc FROM " + ASTRA_DB_KEYSPACE + "."+ ASTRA_DB_TABLE + " WHERE image_id = ?;"    
            individualprepared_stmt = session.prepare(individualquery)
            individualresults = session.execute(individualprepared_stmt,[doc[0]] )
    
            output_list.append({
                    "image_id": str(individualresults[0][0]),
                    "misc": individualresults[0][1]
            })        
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
        return "Error querying database to list entries", 500
    
    print(str(output_list))
    return json.dumps(output_list),200
