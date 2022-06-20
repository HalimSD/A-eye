from flask import Blueprint, redirect, render_template , Flask, request, url_for , flash , Response
from flask_login import login_required, current_user
import argparse
from matplotlib import image
from numpy import imag
from sqlalchemy import false, true
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
            return redirect(url_for('views.camera', name_model = "project"))
        elif request.form.get("parms") == "cameraconpretrained":
            return redirect(url_for('views.camera', name_model = "pretrained"))
        elif request.form.get("parms") == "localcon":
            return redirect(url_for('views.from_folder', name_model = "project"))
        elif request.form.get("parms") == "localconpretrained":
            return redirect(url_for('views.from_folder', name_model = "pretrained")) 
        elif "file" in request.form:
            image = request.files['file']
            if image.filename == '':
                flash('no selected image', category='error')
                return redirect(request.url)
            elif image :
                imagename = secure_filename(image.filename)
                print(imagename)
                path = os.path.join(os.getcwd(),'login/static/images',imagename)
                image.save(path)
                return redirect(url_for('views.upload',path=path)) 

    return render_template("home.html", user=current_user)

@views.route('/upload/')
@login_required
def upload():
    project_model = a_eye.load_checkpoint(argparse.Namespace(ccm=False, clip_length=10, coco=False, conceptual=True, live='l', prefix_length=10, prefix_size=512, pretrained=False, project=True, transformer=False))
    pretrained_model = a_eye.load_checkpoint(argparse.Namespace(ccm=False, clip_length=10, coco=False, conceptual=True, live='l', prefix_length=10, prefix_size=512, pretrained=True, project=False, transformer=False))
    image_caption_project = a_eye.caption_upload(project_model,request.args["path"],False)
    image_caption_pretrained = a_eye.caption_upload(pretrained_model,request.args["path"],True)
    return render_template("upload.html", image_caption=image_caption_project, image_caption_pretrained=image_caption_pretrained, user=current_user)    

@views.route('/from_folder/')
@login_required
def from_folder():
    name_model = request.args["name_model"]
    if name_model == "project" :
        model = a_eye.load_checkpoint(argparse.Namespace(ccm=False, clip_length=10, coco=False, conceptual=True, live='l', prefix_length=10, prefix_size=512, pretrained=False, project=True, transformer=False))
    elif name_model == "pretrained":
        model = a_eye.load_checkpoint(argparse.Namespace(ccm=False, clip_length=10, coco=False, conceptual=True, live='l', prefix_length=10, prefix_size=512, pretrained=True, project=False, transformer=False))
    image_caption = a_eye.caption_from_device(model)
    return render_template("from_folder.html", image_caption=image_caption, name_model=name_model, user=current_user)

@views.route('/camera/')
@login_required
def camera():
    name_model = request.args["name_model"]
    if name_model == "project" :
        model = a_eye.load_checkpoint(argparse.Namespace(ccm=False, clip_length=10, coco=False, conceptual=True, live='l', prefix_length=10, prefix_size=512, pretrained=False, project=True, transformer=False))
    elif name_model == "pretrained":
                model = a_eye.load_checkpoint(argparse.Namespace(ccm=False, clip_length=10, coco=False, conceptual=True, live='l', prefix_length=10, prefix_size=512, pretrained=True, project=False, transformer=False))
    image_caption = a_eye.caption_live(model)
    return Response(image_caption, mimetype='multipart/x-mixed-replace; boundary=frame')
