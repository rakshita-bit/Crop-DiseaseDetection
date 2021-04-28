__version__ = "0.1"
__all__ = ["SimpleHTTPRequestHandler"]
__author__ = "bones7456"
__home_page__ = "http://li2z.cn/"

import os
import posixpath
import http.server
import os
import threading
import urllib.request, urllib.parse, urllib.error
import cgi
import shutil
import mimetypes
import re
from io import BytesIO

class SimpleHTTPRequestHandler(http.server.BaseHTTPRequestHandler):

    """Simple HTTP request handler with GET/HEAD/POST commands.
    This serves files from the current directory and any of its
    subdirectories.  The MIME type for files is determined by
    calling the .guess_type() method. And can reveive file uploaded
    by client.
    The GET/HEAD/POST requests are identical except that the HEAD
    request omits the actual contents of the file.
    """

    server_version = "SimpleHTTPWithUpload/" + __version__
    path_to_image = 'messi.jpeg'
    img = open(path_to_image, 'rb')
    statinfo = os.stat(path_to_image)
    img_size = statinfo.st_size
    print(img_size)
    
    def do_HEAD(self):
        self.send_response(200)
        self.send_header("Content-type", "image/jpg")
        self.send_header("Content-length", self.img_size)
        self.end_headers()
    
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "image/jpg")
        self.send_header("Content-length", self.img_size)
        self.end_headers() 
        f = open(self.path_to_image, 'rb')
        self.wfile.write(f.read())
        f.close()         
def test(HandlerClass = SimpleHTTPRequestHandler,
        ServerClass = http.server.HTTPServer):
    http.server.test(HandlerClass, ServerClass)

if __name__ == '__main__':
    test()


