from flask import Blueprint, redirect, render_template , Flask, request, url_for , flash
from flask_login import login_required, current_user
import argparse
from matplotlib import image
from numpy import imag
from werkzeug.utils import secure_filename
from matplotlib.pyplot import get
import login
import a_eye
import os

views = Blueprint('views', __name__)

@views.route('/', methods=['GET', 'POST'])
@login_required
def home():
    if request.method == 'POST':
        if request.form.get("parms") == "cameracon":
            return redirect(url_for('views.camera'))
        elif request.form.get("parms") == "localcon":
            return redirect(url_for('views.from_folder')) 
        elif "file" in request.form:
            image = request.files['file']
            if image.filename == '':
                flash('no selected image', category='error')
                return redirect(request.url)
            elif image :
                imagename = secure_filename(image.filename)
                print(imagename)
                path = os.path.join(os.getcwd(),'login/static',imagename)
                print(path)
                image.save(path)
                return redirect(url_for('views.upload',path=path)) 

    return render_template("home.html", user=current_user)

@views.route('/upload/')
def upload():
    model = a_eye.load_checkpoint(argparse.Namespace(ccm=False, clip_length=10, coco=False, conceptual=True, live='l', prefix_length=10, prefix_size=512, pretrained=False, project=True, transformer=False))
    image_caption = a_eye.caption_upload(model,request.args["path"])
    return render_template("upload.html", image_caption=image_caption, user=current_user)    

@views.route('/from_folder/')
def from_folder():
    model = a_eye.load_checkpoint(argparse.Namespace(ccm=False, clip_length=10, coco=False, conceptual=True, live='l', prefix_length=10, prefix_size=512, pretrained=False, project=True, transformer=False))
    image_caption = a_eye.caption_from_device(model)
    return render_template("from_folder.html", image_caption=image_caption, user=current_user)

@views.route('/camera/')
def camera():
    model = a_eye.load_checkpoint(argparse.Namespace(ccm=False, clip_length=10, coco=False, conceptual=True, live='l', prefix_length=10, prefix_size=512, pretrained=False, project=True, transformer=False))
    image_caption = a_eye.caption_live(model)
    return render_template("camera.html", image_caption=image_caption, user=current_user)