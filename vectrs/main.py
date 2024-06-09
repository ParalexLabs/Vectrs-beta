import argparse
import asyncio
import numpy as np
import logging
from network import KademliaNode
from database import VectorDBManager

logging.basicConfig(level=logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser(description="P2P Vector Database Node")
    parser.add_argument("mode", choices=["start-node", "create-db", "add-vector", "query-vector", "view-log", "stop-node"], help="Mode of operation.")
    parser.add_argument("--host", default="0.0.0.0", help="Host address for the node.")
    parser.add_argument("--port", type=int, default=8468, help="Port number for the node.")
    parser.add_argument("--bootstrap_host", default=None, help="Bootstrap node host address.")
    parser.add_argument("--bootstrap_port", type=int, default=8468, help="Bootstrap node port number.")
    parser.add_argument("--dim", type=int, help="Dimension of the vector space for the database.")
    parser.add_argument("--space", default="l2", help="Metric space type (e.g., l2, cosine).")
    parser.add_argument("--max_elements", type=int, default=10000, help="Maximum number of elements in the database.")
    parser.add_argument("--db_id", help="ID of the database.")
    parser.add_argument("--vector_id", help="ID of the vector.")
    parser.add_argument("--vector", help="Vector data as a comma-separated string.")
    parser.add_argument("--metadata", help="Metadata for the vector.")
    return parser.parse_args()

async def start_node(host, port, bootstrap_host, bootstrap_port):
    node = KademliaNode(host=host, port=port)
    await node.start()
    if bootstrap_host:
        await node.bootstrap(bootstrap_host, bootstrap_port)
    await asyncio.Future()  # Keeps the node running indefinitely

async def create_vector_database(host, port, dim, space, max_elements, bootstrap_host, bootstrap_port):
    node = KademliaNode(host=host, port=port)
    await node.start()
    if bootstrap_host:
        await node.bootstrap(bootstrap_host, bootstrap_port)
    
    db_manager = VectorDBManager()
    node.set_local_db_manager(db_manager)
    db_id = db_manager.create_database(dim, space, max_elements)
    await node.set_value(db_id, (host, port))
    print(f"Database created with ID: {db_id}")

    await node.stop()

async def add_vector(host, port, db_id, vector_id, vector, metadata, bootstrap_host, bootstrap_port):
    node = KademliaNode(host=host, port=port)
    await node.start()
    if bootstrap_host:
        await node.bootstrap(bootstrap_host, bootstrap_port)

    db_manager = VectorDBManager()
    node.set_local_db_manager(db_manager)
    vector = np.array([float(x) for x in vector.split(',')], dtype=np.float32)
    await node.add_vector(db_id, vector_id, vector, metadata)
    print(f"Vector added with ID: {vector_id}")

    await node.stop()

async def query_vector(host, port, db_id, vector_id, bootstrap_host, bootstrap_port):
    node = KademliaNode(host=host, port=port)
    await node.start()
    if bootstrap_host:
        await node.bootstrap(bootstrap_host, bootstrap_port)

    db_manager = VectorDBManager()
    node.set_local_db_manager(db_manager)
    result = await node.query_vector(db_id, vector_id)
    if result == "Local":
        vector = db_manager.get_vector(db_id, vector_id)
        print(f"Retrieved Vector: {vector}")
    else:
        print(f"Retrieved Vector from remote node: {result}")

    await node.stop()

async def view_log(db_id):
    db_manager = VectorDBManager()
    log = db_manager.get_log(db_id)
    print(f"Log for database {db_id}:")
    for entry in log:
        print(entry)

async def stop_node(host, port):
    node = KademliaNode(host=host, port=port)
    await node.stop()
    print("Node has been stopped")

def main():
    args = parse_args()

    if args.mode == "start-node":
        asyncio.run(start_node(args.host, args.port, args.bootstrap_host, args.bootstrap_port))
    elif args.mode == "create-db":
        if not all([args.dim]):
            print("Missing parameters for creating database. Please provide all necessary information.")
        else:
            asyncio.run(create_vector_database(args.host, args.port, args.dim, args.space, args.max_elements, args.bootstrap_host, args.bootstrap_port))
    elif args.mode == "add-vector":
        if not all([args.db_id, args.vector_id, args.vector]):
            print("Missing parameters for adding vector. Please provide all necessary information.")
        else:
            asyncio.run(add_vector(args.host, args.port, args.db_id, args.vector_id, args.vector, args.metadata, args.bootstrap_host, args.bootstrap_port))
    elif args.mode == "query-vector":
        if not all([args.db_id, args.vector_id]):
            print("Missing parameters for querying vector. Please provide all necessary information.")
        else:
            asyncio.run(query_vector(args.host, args.port, args.db_id, args.vector_id, args.bootstrap_host, args.bootstrap_port))
    elif args.mode == "view-log":
        if not args.db_id:
            print("Missing parameters for viewing log. Please provide the database ID.")
        else:
            asyncio.run(view_log(args.db_id))
    elif args.mode == "stop-node":
        asyncio.run(stop_node(args.host, args.port))

if __name__ == "__main__":
    main()
