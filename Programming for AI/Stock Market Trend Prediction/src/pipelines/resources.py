from dagster import ConfigurableResource
import psycopg2
from pymongo import MongoClient
import os


class PostgreSQLResource(ConfigurableResource):
    
    dbname: str = "postgres"
    user: str = "dap"
    password: str = "dap"
    host: str = "localhost"
    port: str = "5432"
    
    def get_connection(self):
        
        return psycopg2.connect(
            dbname=self.dbname,
            user=self.user,
            password=self.password,
            host=self.host,
            port=self.port
        )


class MongoDBResource(ConfigurableResource):
    
    connection_string: str = "mongodb://localhost:27017/"
    database_name: str = "stock_market_etl"
    
    def get_client(self):
        return MongoClient(self.connection_string)
    
    def get_database(self):
        client = self.get_client()
        return client[self.database_name]
    
    def get_collection(self, collection_name: str):
        db = self.get_database()
        return db[collection_name]


class DataStorageResource(ConfigurableResource):
    
    base_dir: str = "c:/Users/Kenshin/Documents/Programming for AI/Project CA2 - ETL"
    
    @property
    def raw_dir(self):
        return os.path.join(self.base_dir, "data", "raw")
    
    @property
    def processed_dir(self):
        return os.path.join(self.base_dir, "data", "processed")
    
    @property
    def model_dir(self):
        return os.path.join(self.base_dir, "models")
    
    def get_raw_path(self, filename: str = "") -> str:
        return os.path.join(self.raw_dir, filename)
    
    def get_processed_path(self, filename: str = "") -> str:
        return os.path.join(self.processed_dir, filename)
    
    def get_model_path(self, filename: str = "") -> str:
        return os.path.join(self.model_dir, filename)


# Resource instances
postgres_connection = PostgreSQLResource()
mongodb_connection = MongoDBResource()
data_storage = DataStorageResource()
