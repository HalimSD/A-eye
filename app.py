import time
import cv2
import torch
import wavio as wv
from flask import Flask, Response, render_template,request,flash,redirect,url_for,session
import sqlite3
import argparse
from a_eye import Image, caption_live, caption_from_device, generate_caption, load_checkpoint,os
from utils.clip_synthesized import hps, get_text, net_g
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key="123"

app.config["IMAGE_UPLOADS"] = "./static/test"
app.config["ALLOWED_IMAGE_EXTENSIONS"] = ["PNG","JPG"]

con=sqlite3.connect("database.db")
con.execute("create table if not exists user(pid integer primary key,name text,mail text)")
con.close()

@app.route('/')
def index():
    return render_template('index.html')
# Run a_eye.py --pretrained --conceptual
@app.route('/pretrained/')
def pretrained():
    model = load_checkpoint(argparse.Namespace(ccm=False, clip_length=10, coco=False, conceptual=True, live='l', prefix_length=10, prefix_size=512, pretrained=True, project=False, transformer=False))
#   caption_from_device(m)
    test_data_path = os.path.join(os.getcwd(),'./static/test/')
    wav_data_path = os.path.join(os.getcwd(),'./static/wav/')
    image_names = [name for name in os.listdir(test_data_path) if name[-4] == '.']
    results = []
    for image in image_names:
            caption = generate_caption(Image.open(os.path.join(test_data_path, image)), model )
            with torch.no_grad():
                stn_tst = get_text(caption, hps)
                x_tst = stn_tst.unsqueeze(0)
                x_tst_lengths = torch.LongTensor([stn_tst.size(0)])
                audio = net_g.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.float().numpy()
                wav_name = image.rsplit(".",1)[0] +'.wav'
                print(wav_name)
                wv.write(os.path.join(wav_data_path, wav_name), audio ,rate=hps.data.sampling_rate, sampwidth=1)
            results.append([image,caption, wav_name])
    return render_template('view.html',result=results)


@app.route('/project/')
def project():
    model = load_checkpoint(argparse.Namespace(ccm=False, clip_length=10, coco=False, conceptual=True, live='l', prefix_length=10, prefix_size=512, pretrained=False, project=True, transformer=False))
#   caption_from_device(m)
    test_data_path = os.path.join(os.getcwd(),'./static/test/')
    wav_data_path = os.path.join(os.getcwd(),'./static/wav/')
    image_names = [name for name in os.listdir(test_data_path) if name[-4] == '.']
    results = []
    for image in image_names:
            caption = generate_caption(Image.open(os.path.join(test_data_path, image)), model )
            with torch.no_grad():
                stn_tst = get_text(caption, hps)
                x_tst = stn_tst.unsqueeze(0)
                x_tst_lengths = torch.LongTensor([stn_tst.size(0)])
                audio = net_g.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.float().numpy()
                wav_name = image.rsplit(".",1)[0] +'.wav'
                print(wav_name)
                wv.write(os.path.join(wav_data_path, wav_name), audio ,rate=hps.data.sampling_rate, sampwidth=1)
            results.append([image,caption, wav_name])
    return render_template('view.html',result=results)


# Run a_eye.py --project --conceptual
@app.route('/live/')
def live ():
  model = load_checkpoint(argparse.Namespace(ccm=False, clip_length=10, coco=False, conceptual=True, live='l', prefix_length=10, prefix_size=512, pretrained=False, project=True, transformer=False)) 
  return Response(caption_live(model), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/login',methods=["GET","POST"])
def login():
    if request.method=='POST':
        name=request.form['name']
        password=request.form['password']
        con=sqlite3.connect("database.db")
        con.row_factory=sqlite3.Row
        cur=con.cursor()
        cur.execute("select * from user where name=? and mail=?",(name,password))
        data=cur.fetchone()

        if data:
            session["name"]=data["name"]
            session["mail"]=data["mail"]
            return redirect("user")
        else:
            flash("Username and Password Mismatch","danger")
    return redirect(url_for("index"))


@app.route('/user',methods=["GET","POST"])
def user():
    return render_template("user.html")


def allowed_image(filename):
    if not "." in filename:
        return False
    ext = filename.rsplit(".",1)[1]
    if ext.upper() in app.config["ALLOWED_IMAGE_EXTENSIONS"]:
        return True
    else:
        return False
@app.route('/upload',methods=["GET","POST"])
def upload_image():
	if request.method == "POST":
		image = request.files['file']

		if not allowed_image(image.filename):
			print("Sorry, but only png extension is allowed. Please change the format")
			return redirect(url_for("user"))
		filename = secure_filename(image.filename)
		basedir = os.path.abspath(os.path.dirname(__file__))
		image.save(os.path.join(basedir,app.config["IMAGE_UPLOADS"],filename))
        
		return redirect('/project')
	return render_template('user.html')


@app.route('/display/<filename>')
def display_image(filename):
	return redirect(url_for('static',filename = "/Images" + filename), code=301)


@app.route('/register',methods=['GET','POST'])
def register():
    if request.method=='POST':
        try:
            name=request.form['name']
            mail=request.form['mail']
            con=sqlite3.connect("database.db")
            cur=con.cursor()
            cur.execute("insert into user(name,mail)values(?,?)",(name,mail))
            con.commit()
            flash("Record Added  Successfully","success")
        except:
            flash("Error in Insert Operation","danger")
        finally:
            return redirect(url_for("index"))
            con.close()

    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for("index"))

if __name__ == '__main__':
    app.run(debug=False)
