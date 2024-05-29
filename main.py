import pywebio as peb
from pywebio import start_server
from inference_tens import Master
import time as t
m = Master()

def app():
    peb.output.put_markdown(r""" # Neural Style Transfer 
    Style Base
    """)
    imgs = peb.input.file_upload("Select Content Image:", accept="data/*", multiple=True)
    addr1,addr2 = "",""
    for img in imgs:
        #peb.output.put_image(img['content'])
        addr1 = "data/input_image/" + img['filename']
    imgs = peb.input.file_upload("Select Style Image:", accept="data/*", multiple=True)
    for img in imgs:
        #peb.output.put_image(img['content'])
        addr2 = "data/target_style/" + img['filename']
   
    org = open(addr1, 'rb').read()  
    peb.output.put_image(org, width='300px')
    peb.output.put_text("Content Image")
    style = open(addr2, 'rb').read()  
    peb.output.put_image(style, width='300px')
    peb.output.put_text("Style Image")
    
    with peb.output.put_loading():
            peb.output.put_text("Please Wait...")
            m.put_to(addr1,addr2)
            t.sleep(4)
    

    out = open('style.jpg', 'rb').read()  
    peb.output.put_image(out, width='300px')
    peb.output.put_text("Style Transfer Image")


if __name__ == '__main__':
    start_server(app, port=8085, debug=True)