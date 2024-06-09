import asyncio
import logging
from kademlia.network import Server

logging.basicConfig(level=logging.INFO)

class KademliaNode:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.server = Server()
        self.local_db_manager = None

    async def start(self):
        await self.server.listen(self.port)
        print(f"Node started at {self.host}:{self.port}")

    async def stop(self):
        self.server.stop()
        print("Node has been stopped")

    async def bootstrap(self, bootstrap_host, bootstrap_port):
        await self.server.bootstrap([(bootstrap_host, bootstrap_port)])
        print(f"Node bootstrapped to {bootstrap_host}:{bootstrap_port}")

    async def set_value(self, key, value):
        if isinstance(value, tuple):
            value = f"{value[0]}:{value[1]}"
        await self.server.set(key, value)
        print(f"Set key {key} to value {value}")

    async def get_value(self, key):
        value = await self.server.get(key)
        if value and ":" in value:
            host, port = value.split(":")
            print(f"Get key {key} returned value {value}")
            return host, int(port)
        print(f"Get key {key} returned value {value}")
        return value

    async def query_vector(self, db_id, vector_id):
        print(f"Querying vector with db_id: {db_id}, vector_id: {vector_id}")

        # Check local storage first
        if self.local_db_manager:
            try:
                db = self.local_db_manager.get_database(db_id)
                vector = db.get(vector_id)
                print(f"Vector found locally: {vector}")
                return vector
            except ValueError as e:
                print(f"Vector not found locally: {e}")

        # If not found locally, query the DHT
        host_port = await self.get_value(db_id)
        if host_port:
            host, port = host_port
            print(f"Host: {host}, Port: {port} for db_id: {db_id}")
            if (host, port) == (self.host, self.port):
                print(f"Vector is local for db_id {db_id}")
                return "Local"
            else:
                print(f"Vector should be remote for db_id {db_id}, at {host}:{port}")
                remote_node = KademliaNode(host, port)
                await remote_node.start()
                try:
                    vector = await remote_node.query_vector(db_id, vector_id)
                finally:
                    await remote_node.stop()
                return vector
        print(f"Host and port not found for db_id {db_id}")
        return None

    def set_local_db_manager(self, db_manager):
        self.local_db_manager = db_manager

    async def add_vector(self, db_id, vector_id, vector, metadata=None):
        print(f"Adding vector with db_id: {db_id}, vector_id: {vector_id}")
        
        # Add vector to local database
        if self.local_db_manager:
            db = self.local_db_manager.get_database(db_id)
            db.add(vector, vector_id)
            if metadata:
                db.add_metadata(vector_id, metadata)
            print(f"Vector added locally: {vector_id}")
        
        # Propagate vector information to the DHT
        await self.set_value(db_id, (self.host, self.port))
        print(f"Vector metadata added to DHT: {vector_id}")

    async def update_vector(self, db_id, vector_id, vector, metadata=None):
        print(f"Updating vector with db_id: {db_id}, vector_id: {vector_id}")
        
        # Update vector in local database
        if self.local_db_manager:
            db = self.local_db_manager.get_database(db_id)
            db.update(vector_id, vector)
            if metadata:
                db.update_metadata(vector_id, metadata)
            print(f"Vector updated locally: {vector_id}")
        
        # Update vector information in the DHT
        await self.set_value(db_id, (self.host, self.port))
        print(f"Vector metadata updated in DHT: {vector_id}")

    async def delete_vector(self, db_id, vector_id):
        print(f"Deleting vector with db_id: {db_id}, vector_id: {vector_id}")
        
        # Delete vector from local database
        if self.local_db_manager:
            db = self.local_db_manager.get_database(db_id)
            db.delete(vector_id)
            print(f"Vector deleted locally: {vector_id}")
        
        # Update DHT to reflect deletion
        await self.set_value(db_id, (self.host, self.port))
        print(f"Vector metadata deleted from DHT: {vector_id}")
