import socket

# Create a TCP/IP socket
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Replace 'YOUR_SERVER_IP' with the server's IP address
client.connect(('172.20.10.2', 8080))

# Send a message to the server
client.sendall('I am CLIENT\n'.encode('utf-8'))

# Receive data from the server
from_server = client.recv(4096)

# Close the connection
client.close()

# Print the server's response
print(from_server.decode('utf-8'))
