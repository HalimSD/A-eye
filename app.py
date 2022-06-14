from flask import Flask, render_template,request,flash,redirect,url_for,session
import sqlite3
import argparse
from a_eye import WEIGHTS_PATHS, caption_live,device, caption_from_device, torch, load_checkpoint, ClipCaptionModel,os


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
app.secret_key="123"

con=sqlite3.connect("database.db")
con.execute("create table if not exists user(pid integer primary key,name text,mail text)")
con.close()

@app.route('/')
def index():
    return render_template('index.html')
# Run a_eye.py --pretrained --conceptual
@app.route('/pretrained/')
def pretrained():
  m = load_checkpoint(argparse.Namespace(ccm=False, clip_length=10, coco=False, conceptual=True, live='l', prefix_length=10, prefix_size=512, pretrained=True, project=False, transformer=False))
  caption_from_device(m)
#   print('Conceptual pretrained model')
#   model_path = os.path.join(WEIGHTS_PATHS.get('pretrained_conceptual'), 'conceptual_weights.pt')
#   checkpoint = torch.load(model_path, map_location=device)
#   model = ClipCaptionModel(prefix_length = 10, prefix_size = 512)
#   model.load_state_dict(checkpoint)
#   model.eval()
#   model.to(device=device)
  return 'Conceptual pretrained model'
@app.route('/project/')
def project():
  m = load_checkpoint(argparse.Namespace(ccm=False, clip_length=10, coco=False, conceptual=True, live='l', prefix_length=10, prefix_size=512, pretrained=False, project=True, transformer=False))
  caption_from_device(m)
  return 'project conceptual'

# Run a_eye.py --project --conceptual
@app.route('/live/')
def live ():
  m = load_checkpoint(argparse.Namespace(ccm=False, clip_length=10, coco=False, conceptual=True, live='l', prefix_length=10, prefix_size=512, pretrained=False, project=True, transformer=False))
  caption_live(m)
  return 'Live conceptual'

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
    app.run(debug=True)