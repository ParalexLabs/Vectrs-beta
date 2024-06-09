# readme   
# Vectrs - Decentralized & Distributed Vector Database   
## Overview   
**Vectrs** is a decentralized & distributed vector database designed for efficient storage and retrieval of vector embeddings. By utilizing commodity hardware and scaling horizontally, Vectrs offers a cost-effective solution compared to traditional centralized databases. It leverages a distributed hash table (DHT) for decentralized data management, ensuring scalability and fault tolerance.   
## Features   
- **Distributed Storage**: Data is distributed across multiple nodes for scalability and fault tolerance.   
- **Cost-Effective**: Utilizes commodity hardware to reduce costs.   
- **Horizontal Scalability**: Easily add more nodes to handle increased data load.   
- **Efficient Vector Operations**: Optimized for storing and querying vector embeddings.   
- **OpenAI Integration**: Supports storing and retrieving vector embeddings generated by OpenAI models.   
   
## Installation   
You can install Vectrs from PyPI using pip:   
```

pip install vectrs


```
## Usage   
### Initializing a Vectrs Node   
To initialize a Vectrs node, import the `KademliaNode` class and start the node:   
```
import asyncio
from vectrs.network import KademliaNode
from vectrs.database import VectorDBManager

async def start_node():
    db_manager = VectorDBManager()
    node = KademliaNode(host='127.0.0.1', port=8468)
    node.set_local_db_manager(db_manager)
    await node.start()
    return node

if __name__ == "__main__":
    asyncio.run(start_node())


```
###    
### Adding and Querying Vectors   
### Adding Vectors   
You can add vectors to the database by using the `add\_vector` method:   
```
import numpy as np

async def add_vectors(node, db_id):
    vectors = {
        "vec1": np.random.rand(1024).astype(np.float32),
        "vec2": np.random.rand(1024).astype(np.float32),
    }
    metadata = "Example metadata"
    for vector_id, vector in vectors.items():
        await node.add_vector(db_id, vector_id, vector, metadata)
        print(f"Added vector with ID: {vector_id} and metadata: {metadata}")

if __name__ == "__main__":
    node = asyncio.run(start_node())
    db_id = node.local_db_manager.create_database(dim=1024)
    print(f"Created database with ID: {db_id}")
    asyncio.run(add_vectors(node, db_id))


```
### Querying Vectors   
You can query vectors from the database using the `query\_vector` method:   
```
async def query_vector(node, db_id, vector_id):
    vector = await node.query_vector(db_id, vector_id)
    print(f"Retrieved vector: {vector}")

if __name__ == "__main__":
    node = asyncio.run(start_node())
    db_id = node.local_db_manager.create_database(dim=1024)
    print(f"Created database with ID: {db_id}")
    asyncio.run(query_vector(node, db_id, "vec1"))


```
### Example with OpenAI Embeddings   
You can store OpenAI-generated vector embeddings in Vectrs:   
```
import openai
import numpy as np

openai.api_key = 'your_openai_api_key'

async def store_openai_embedding(node, db_id, text):
    response = openai.Embedding.create(input=[text], model="text-embedding-ada-002")
    vector = np.array(response['data'][0]['embedding'], dtype=np.float32)
    await node.add_vector(db_id, "openai_vec", vector, "OpenAI generated embedding")
    print("Stored OpenAI embedding")

if __name__ == "__main__":
    node = asyncio.run(start_node())
    db_id = node.local_db_manager.create_database(dim=1024)
    print(f"Created database with ID: {db_id}")
    asyncio.run(store_openai_embedding(node, db_id, "Example text for embedding"))


```
### Retrieving Vector and Log Hashes   
After adding a vector, you can retrieve its hash and log hash as follows:   
```
async def add_vector_and_get_hashes(node, db_id):
    vector = np.random.rand(1024).astype(np.float32)
    vector_id = "vec1"
    metadata = "Example metadata"

    # Add the vector
    await node.add_vector(db_id, vector_id, vector, metadata)

    # Retrieve vector hash and log hash
    vector_hash = node.local_db_manager.get_vector_hash(db_id, vector_id)
    log_hash = node.local_db_manager.get_log_hash(db_id, vector_id)

    print(f"Added vector with ID: {vector_id} and metadata: {metadata}")
    print(f"Vector hash: {vector_hash}")
    print(f"Log hash: {log_hash}")

if __name__ == "__main__":
    node = asyncio.run(start_node())
    db_id = node.local_db_manager.create_database(dim=1024)
    print(f"Created database with ID: {db_id}")
    asyncio.run(add_vector_and_get_hashes(node, db_id))


```
## API Reference   
### KademliaNode   
### Methods   
- `start()`: Starts the node and listens for connections.   
- `stop()`: Stops the node.   
- `bootstrap(bootstrap\_host, bootstrap\_port)`: Bootstraps the node to an existing network.   
- `add\_vector(db\_id, vector\_id, vector, metadata=None)`: Adds a vector to the database.   
- `query\_vector(db\_id, vector\_id)`: Queries a vector from the database.   
- `set\_local\_db\_manager(db\_manager)`: Sets the local database manager.   
- `get\_value(key)`: Retrieves a value from the DHT by key.   
   
### VectorDBManager   
### Methods   
- `create\_database(dim)`: Creates a new vector database with the specified dimensions and returns the database ID.   
- `get\_database(db\_id)`: Retrieves a database by its ID.   
- `get\_vector\_hash(db\_id, vector\_id)`: Retrieves the hash of a vector by its ID.   
- `get\_log\_hash(db\_id, vector\_id)`: Retrieves the log hash of a vector by its ID.   
   
## Contribution   
Contributions are welcome! Please open an issue or submit a pull request on [GitHub](https://github.com/yourusername/vectrs).   
## License   
This project is licensed under the MIT License. See the [LICENSE](https://github.com/yourusername/vectrs/blob/main/LICENSE) file for more details.   
## Support   
For support or inquiries, please contact your.email@example.com.   