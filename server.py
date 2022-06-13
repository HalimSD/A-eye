import argparse
from a_eye import caption_from_device, main, load_checkpoint, caption_live

from flask import Flask, render_template


parser = argparse.ArgumentParser()
parser.add_argument('--project', dest='project', action="store_true")
parser.add_argument('--live', default='l')
parser.add_argument('--pretrained', dest='pretrained', action="store_true")
parser.add_argument('--ClipCaptionModel', dest='ccm', action="store_true")
parser.add_argument('--prefix_length', type=int, default=10)
parser.add_argument('--clip_length', type=int, default=10)
parser.add_argument('--prefix_size', type=int, default=512)
parser.add_argument('--coco', dest='coco', action="store_true")
parser.add_argument('--conceptual', dest='conceptual', action="store_true")
parser.add_argument('--transformer', dest='transformer', action="store_true")

args = parser.parse_args()
app = Flask(__name__)

@app.route('/')
def index():
  print("Halim")
  return render_template('index.html')
  


# @app.route('/load_checkpoint/<pretrained>', methods=['GET'])
# @app.route(methods=['GET']'/load_checkpoint/'/<pretrained>')
@app.route('/my-link/')
def my_link():

  # print ('I got clicked!')
  # something = args["pretrained"] 
  m = load_checkpoint(args)
  caption_from_device(m)
 

if __name__ == '__main__':
  app.run(debug=True)