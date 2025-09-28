#!/usr/bin/env python3
"""
Server Photo Receiver
Requests photos from Raspberry Pi and saves them for processing
"""
import socket
import struct
import os


class PhotoServer:
    def __init__(self, host="172.20.10.6", port=8888):
        self.host = host
        self.port = port
        self.current_image = None

    def receive_all(self, sock, length):
        """Helper to receive exactly `length` bytes"""
        data = b""
        while len(data) < length:
            packet = sock.recv(length - len(data))
            if not packet:
                return None
            data += packet
        return data

    def receive_file(self, sock):
        """Receive file from client with header"""
        try:
            # Header format: filename length (I), file size (Q)
            header = self.receive_all(sock, 12)
            if not header:
                print("Failed to receive header")
                return None

            filename_len, file_size = struct.unpack("!IQ", header)

            # Receive filename
            filename_bytes = self.receive_all(sock, filename_len)
            if not filename_bytes:
                print("Failed to receive filename")
                return None
            filename = filename_bytes.decode()
            print(f"Receiving file: {filename} ({file_size} bytes)")

            # Save file as current_image.jpg
            filepath = "current_image.jpg"
            received = 0
            with open(filepath, "wb") as f:
                while received < file_size:
                    chunk = sock.recv(min(4096, file_size - received))
                    if not chunk:
                        break
                    f.write(chunk)
                    received += len(chunk)

            if received == file_size:
                print(f"Image saved: {filepath}")
                self.current_image = filepath
                return filepath
            else:
                print(f"Error: Received {received} of {file_size} bytes")
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
            server_sock.listen(5)

            print("Waiting for Pi connection...")
            client_sock, addr = server_sock.accept()

            success = self.handle_client(client_sock, addr)

            server_sock.close()
            return success
        except Exception as e:
            print(f"Error requesting photo: {e}")
            return False

    def get_current_image_path(self):
        """Get path to current image"""
        return self.current_image


# Global instance
photo_server = PhotoServer()


def get_new_image():
    """Request a new image from Pi and return its path"""
    print("\n[PHOTO REQUEST] Requesting new image from Pi...")
    if photo_server.request_photo():
        return photo_server.get_current_image_path()
    else:
        print("Failed to get image from Pi")
        return None
