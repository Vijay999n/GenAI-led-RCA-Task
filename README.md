**rca_lookup_LLM** - This program accepts the description it can be a statement or just a keyword
          
         python .\rca_lookup_LLM.py --description "on the remote desktop the memory  is full" 


  **rca_lookup_batch** - This will process all the incidents which are present in the incidets.csv file taking the summary column as input and do the search in KEDB if not found get the New RCA lookup


           python .\rca_lookup_batch.py --incidents_csv incidents.csv --kedb_csv kedb.csv --summary_column summary
