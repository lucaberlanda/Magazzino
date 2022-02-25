import json
import hashlib
import datetime
from flask import Flask, jsonify


# Part 1 >> Build Blockchain
class Blockchain:
    def __init__(self):
        self.chain = []
        self.create_block(proof=1, previous_hash='0000')  # used right after mining a block

    def create_block(self, proof, previous_hash):
        # index, datetime,`````````````````````````````` proof (Nonce?) & previous hash
        block = {'index': len(self.chain) + 1,
                 'timestamp': str(datetime.datetime.now()),
                 'proof': proof,
                 'previous_hash': previous_hash
                 }

        self.chain.append(block)
        return block

    def get_previous_block(self):
        return self.chain[-1]

    def PoW(self, block):
        new_proof = 1
        check_proof = False

        encoded_block = json.dumps(block, sort_keys=True).encode()
        while check_proof is False:
            hash_op = hashlib.sha256(encoded_block + str(new_proof).encode()).hexdigest()
            if hash_op[:4] == '0000':
                check_proof = True
            else:
                new_proof += 1

        return hash_op, new_proof

    def hash(self, block):
        encoded_block = json.dumps(block, sort_keys=True).encode()
        return hashlib.sha256(encoded_block).hexdigest()

    def is_chain_valid(self):

        previous_block = self.chain[0]
        block_index = 1

        while block_index < len(self.chain):

            block = self.chain[block_index]

            # todo: the check should not do the PoW from scratch
            #   the self.PoW(previous_block) should go on the previous block (new field "hash")
            #   https://andersbrownworth.com/blockchain/distributed

            hash_operation, _ = self.PoW(previous_block)
            if block['previous_hash'] != hash_operation:
                return False

            hash_operation, _ = self.PoW(block)
            if hash_operation[:4] != '0000':
                return False

            previous_block = block
            block_index += 1

        return True


# Part 2 - Mining our Blockchain

# Creating a Web App
app = Flask(__name__)

# Creating a Blockchain
blockchain = Blockchain()


# Mining a new block
@app.route('/mine_block', methods=['GET'])
def mine_block():
    previous_block = blockchain.get_previous_block()
    previous_proof = previous_block['proof']
    previous_block = blockchain.get_previous_block()
    previous_hash, proof = blockchain.PoW(previous_block)
    block = blockchain.create_block(proof, previous_hash)
    response = {'message': 'Congratulations, you just mined a block!',
                'index': block['index'],
                'timestamp': block['timestamp'],
                'proof': block['proof'],
                'previous_hash': block['previous_hash']}
    return jsonify(response), 200


# Getting the full Blockchain
@app.route('/get_chain', methods=['GET'])
def get_chain():
    response = {'chain': blockchain.chain,
                'length': len(blockchain.chain)}
    return jsonify(response), 200


# Checking if the Blockchain is valid
@app.route('/is_valid', methods=['GET'])
def is_valid():
    is_valid = blockchain.is_chain_valid()
    if is_valid:
        response = {'message': 'All good. The Blockchain is valid.'}
    else:
        response = {'message': 'Houston, we have a problem. The Blockchain is not valid.'}
    return jsonify(response), 200


# Running the app
app.run(host='0.0.0.0', port=5000)

# Testing
# blockchain = Blockchain()

# def mine_block():
#     previous_block = blockchain.get_previous_block()
#     previous_hash, proof = blockchain.PoW(previous_block)
#     block = blockchain.create_block(proof, previous_hash)

# mine_block()
# mine_block()
# mine_block()
# mine_block()
# print(blockchain.chain)
# print(blockchain.is_chain_valid())
