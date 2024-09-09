#This is a utility that will return some basic stats. 
#1) The number of rows in the database 
#2) The number of unique image_ids (quad trees)
    
from utils.utils import (
    session, ASTRA_DB_KEYSPACE, ASTRA_DB_TABLE
)

def count_database():
    #Number of records. 
    
    try: 
        query = "SELECT COUNT(*) FROM " + ASTRA_DB_KEYSPACE + "."+ ASTRA_DB_TABLE + " ;"    
        prepared_stmt = session.prepare(query)
        print(f"prepared_stmt1 {prepared_stmt}")
        results = session.execute(prepared_stmt,[] )
        total_number_of_rows = results[0][0]
        print(f"Total number of rows in database: {total_number_of_rows}")
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
        return "Error querying database for total rows", 500

    try:
        #query = "SELECT COUNT(*) FROM (SELECT DISTINCT image_id FROM " + ASTRA_DB_KEYSPACE + "."+ ASTRA_DB_TABLE + ") ;"    
        query = "SELECT DISTINCT image_id FROM " + ASTRA_DB_KEYSPACE + "."+ ASTRA_DB_TABLE + " ;"    
        prepared_stmt = session.prepare(query)
        results = session.execute(prepared_stmt,[] )

#TODO Tremendously inefficient in cassandra.  But will do for today until we split out a relational record for each quadtree
        total_quadtrees = 0
        for doc in results:
           total_quadtrees+=1 
        print(f"Total number of quadtrees in database: {total_quadtrees}")
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
        return "Error querying database for distinct entries", 500
    return {
            "total_number_of_rows": total_number_of_rows,
            "total_quadtrees": total_quadtrees            
         },200
