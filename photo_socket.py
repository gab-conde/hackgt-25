#!/usr/bin/env python3
"""
Server Photo Receiver
Requests photos from Raspberry Pi and saves them for processing
"""
import socket
import time
import os
from datetime import datetime

class PhotoServer:
    def __init__(self, host='0.0.0.0', port=8888):
        self.host = host
        self.port = port
        self.current_image = None
    
    def receive_file(self, sock):
        """Receive file from client"""
        try:
            # Receive filename
            filename = sock.recv(1024).decode().strip()
            print(f"Receiving file: {filename}")
            
            # Receive file size
            file_size = int(sock.recv(1024).decode().strip())
            print(f"File size: {file_size} bytes")
            
            # Save to current working directory with simple name
            filepath = "current_image.jpg"
            
            # Receive file data
            received = 0
            with open(filepath, 'wb') as f:
                while received < file_size:
                    data = sock.recv(min(4096, file_size - received))
                    if not data:
                        break
                    f.write(data)
                    received += len(data)
            
            if received == file_size:
                print(f"Image saved: {filepath}")
                self.current_image = filepath
                return filepath
            else:
                print(f"Error: Received {received} bytes, expected {file_size}")
                return None
                
        except Exception as e:
            print(f"Error receiving file: {e}")
            return None
    
    def handle_client(self, sock, addr):
        """Handle client connection"""
        print(f"Pi connected from {addr}")
        
        try:
            # Send photo request
            sock.sendall(b"TAKE_PHOTO")
            print("Photo request sent")
            
            # Receive photo
            filepath = self.receive_file(sock)
            
            if filepath:
                print(f"Successfully received image: {filepath}")
                return True
            else:
                print("Failed to receive photo")
                return False
                
        except Exception as e:
            print(f"Error handling client {addr}: {e}")
            return False
        finally:
            sock.close()
    
    def request_photo(self):
        """Request a single photo from Pi"""
        try:
            server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_sock.bind((self.host, self.port))
            server_sock.listen(1)
            
            print(f"Waiting for Pi connection...")
            
            # Wait for Pi to connect
            client_sock, addr = server_sock.accept()
            
            # Handle the connection
            success = self.handle_client(client_sock, addr)
            
            server_sock.close()
            return success
            
        except Exception as e:
            print(f"Error requesting photo: {e}")
            return False
    
    def get_current_image_path(self):
        """Get path to current image"""
        return self.current_image

# Global photo server instance
photo_server = PhotoServer()

def get_new_image():
    """Request a new image from Pi and return the path"""
    print("\n[PHOTO REQUEST] Requesting new image from Pi...")
    if photo_server.request_photo():
        return photo_server.get_current_image_path()
    else:
        print("Failed to get image from Pi")
        return None