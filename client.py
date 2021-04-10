import socket
import tqdm
import os
import argparse
s = socket.socket()
s.connect(('2.tcp.ngrok.io',12056))
SEPARATOR = "<SEPARATOR>"
BUFFER_SIZE = 1024 * 4 #4KB
host='2.tcp.ngrok.io'
port=12056



while True:
    str = input("S: ")
    try:
        s.send(str.encode())
    except:
        break
    if(str == "Bye" or str == "bye"):
        break
    try:
        returned_message=s.recv(1024).decode()
    except:
        break
    print ("N:",returned_message)
    if returned_message=="please send file":
        filename=input("input file directory: ")
        # get the file size
        filesize = os.path.getsize(filename)
        # create the client socket

        # send the filename and filesize
        s.send(f"{filename}{SEPARATOR}{filesize}".encode())

        # start sending the file
        progress = tqdm.tqdm(range(filesize), f"Sending {filename}", unit="B", unit_scale=True, unit_divisor=1024)
        with open(filename, "rb") as f:
            while True:
                # read the bytes from the file
                bytes_read = f.read(BUFFER_SIZE)
                if not bytes_read:
                    # file transmitting is done
                    break
                # we use sendall to assure transimission in
                # busy networks
                s.sendall(bytes_read)
                # update the progress bar
                progress.update(len(bytes_read))
        s.close()
        
s.close()





