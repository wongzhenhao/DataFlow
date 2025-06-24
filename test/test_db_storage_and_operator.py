from dataflow.operators.db.db_operator import DBOperator

from dataflow.utils.Storage import DBStorage

class DBShowCasePipeline():
    def __init__(self):
        self.storage = DBStorage()
        self.operator = DBOperator("SELECT * FROM example_table")
    def run(self):
        """
        Run the DBOperator with the DBStorage.
        """

        output_keys_step1 = self.operator.run(
            storage=self.storage, 
            input_key="example_input")
        
        print(f"Operation finished. Output keys from DBOperator: {output_keys_step1}")
        
    