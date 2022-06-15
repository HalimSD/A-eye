import time
import cv2
from flask import Flask, Response, render_template,request,flash,redirect,url_for,session
import sqlite3
import argparse
from a_eye import Image, caption_live, caption_from_device, generate_caption, load_checkpoint,os


app = Flask(__name__)
app.secret_key="123"

con=sqlite3.connect("database.db")
con.execute("create table if not exists user(pid integer primary key,name text,mail text)")
con.close()
# # camera = cv2.VideoCapture(0)
# def gen_frames():  # generate frame by frame from camera
#     while True:
#         # Capture frame-by-frame
#         success, frame = camera.read()  # read the camera frame
#         if not success:
#             break
#         else:
#             ret, buffer = cv2.imencode('.jpg', frame)
#             frame = buffer.tobytes()
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
@app.route('/')
def index():
    return render_template('index.html')
# Run a_eye.py --pretrained --conceptual
@app.route('/pretrained/')
def pretrained():
    model = load_checkpoint(argparse.Namespace(ccm=False, clip_length=10, coco=False, conceptual=True, live='l', prefix_length=10, prefix_size=512, pretrained=True, project=False, transformer=False))
#   caption_from_device(m)
    test_data_path = os.path.join(os.getcwd(),'./data/test')
    image_paths = [os.path.join(test_data_path, name) for name in os.listdir(test_data_path) if name[-4] == '.']
    img_list = [Image.open(image) for image in image_paths]  
    res = ""
    results = ""
    for image in img_list:
            caption = generate_caption(image, model )
            res = res+caption+'\n'
            results = res.replace('\n', '<br>')
    return render_template('view.html', result=results)

@app.route('/project/')
def project():
    model = load_checkpoint(argparse.Namespace(ccm=False, clip_length=10, coco=False, conceptual=True, live='l', prefix_length=10, prefix_size=512, pretrained=False, project=True, transformer=False))
#   caption_from_device(m)
    test_data_path = os.path.join(os.getcwd(),'./data/test')
    image_paths = [os.path.join(test_data_path, name) for name in os.listdir(test_data_path) if name[-4] == '.']
    img_list = [Image.open(image) for image in image_paths]  
    res = ""
    results = ""
    for image in img_list:
            caption = generate_caption(image, model )
            res = res+caption+'\n'
            results = res.replace('\n', '<br>')
    return render_template('view.html', result=results)
    
#   return 'project conceptual'
# @app.route('/video_feed')
# def video_feed():
#     #Video streaming route. Put this in the src attribute of an img tag
#     return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

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
