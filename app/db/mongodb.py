# app/db/mongodb.py
from motor.motor_asyncio import AsyncIOMotorClient
from ..core.config import settings
from ..core.logger import logger
from typing import Optional

class MongoDBClient:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MongoDBClient, cls).__new__(cls)
            cls._instance.client = None
            cls._instance.db = None
        return cls._instance
    
    async def connect(self):
        try:
            logger.info("Connecting to MongoDB...")
            self.client = AsyncIOMotorClient(settings.MONGODB_URL)
            self.db = self.client[settings.MONGODB_DB_NAME]
            logger.info(f"Connected to MongoDB: {settings.MONGODB_DB_NAME}")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
    
    async def close(self):
        if self.client is not None:
            logger.info("Closing MongoDB connection...")
            self.client.close()
    
    def get_collection(self, collection_name: str):
        if self.db is None:
            raise ValueError("Database connection not established")
        return self.db[collection_name]
    
    async def insert_one(self, collection_name: str, document: dict) -> Optional[str]:
        try:
            collection = self.get_collection(collection_name)
            result = await collection.insert_one(document)
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Error inserting document into {collection_name}: {e}")
            return None
    
    async def find_one(self, collection_name: str, query: dict) -> Optional[dict]:
        try:
            collection = self.get_collection(collection_name)
            return await collection.find_one(query)
        except Exception as e:
            logger.error(f"Error finding document in {collection_name}: {e}")
            return None

# Global MongoDB instance
mongodb = MongoDBClient()
