import hashlib

string = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."
encoded = string.encode()
result = hashlib.sha256(encoded)
print("String : ", end="")
print(string)
print("Hash Value : ", end="")
print(result)
print("Hexadecimal equivalent: ", result.hexdigest())
print("Digest Size : ", end="")
print(result.digest_size)
print("Block Size : ", end="")
print(result.block_size)
